# Copyright (c) Facebook, Inc. and its affiliates.
# Modified for Protein Data Loading

import logging
import os
import struct
import time
import json
import numpy
import torch
from typing import Dict, Any, Union, List, Optional, Tuple, Type
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_INDEX_HEADER = b"MMIDIDX\x00\x00"

# --- Copy of DType, _IndexWriter, _IndexReader, _BinReader, etc. ---
# (Keeping these unchanged as they are essential)
class DType(Enum):
    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[numpy.number]) -> int:
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[numpy.number]:
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[numpy.number]]) -> int:
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> Type[numpy.number]:
        if cardinality is not None and cardinality < 65500:
            return numpy.uint16
        else:
            return numpy.int32

class _IndexReader(object):
    def __init__(self, idx_path: str, multimodal: bool) -> None:
        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"
            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"
            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)
            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )
        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int64, count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        self.document_indices = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int64, count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )

    def __del__(self) -> None:
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()
            del self.bin_buffer_mmap

    def __len__(self) -> int:
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]:
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            None,
        )

class _BinReader(ABC):
    @abstractmethod
    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        pass

class _MMapBinReader(_BinReader):
    def __init__(self, bin_path: str) -> None:
        self._bin_buffer_mmap = numpy.memmap(bin_path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        return numpy.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)

    def __del__(self) -> None:
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap

# --- Main Classes ---

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, path_prefix: str, multimodal: bool = False, mmap: bool = True) -> None:
        super().__init__()
        self.path_prefix = path_prefix
        self.mmap = mmap
        
        self.initialize(path_prefix, multimodal, mmap)

    def initialize(self, path_prefix: str, multimodal: bool, mmap: bool) -> None:
        idx_path = path_prefix + ".idx"
        bin_path = path_prefix + ".bin"
        
        assert os.path.exists(idx_path) and os.path.exists(bin_path), \
            f"Files not found: {idx_path} or {bin_path}"

        if mmap:
            self.bin_reader = _MMapBinReader(bin_path)
        else:
            raise NotImplementedError("Only mmap supported for this simplified version")
            
        self.index = _IndexReader(idx_path, multimodal)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: Union[int, numpy.integer]) -> numpy.ndarray:
        if isinstance(idx, (int, numpy.integer)):
            sequence_pointer, sequence_length, _ = self.index[idx]
            sequence = self.bin_reader.read(
                dtype=self.index.dtype, count=sequence_length, offset=sequence_pointer
            )
            return sequence
        else:
            raise TypeError("Only integer indexing supported")

class ProteinDataset(torch.utils.data.Dataset):
    """
    A wrapper dataset that loads protein data from binary files (mmap).
    Everything (Seq, ID, Domain) is stored in IndexedDatasets to save RAM.
    
    Structure:
      - prefix_seq_document.bin/.idx (Ints)
      - prefix_entry_id_document.bin/.idx (Uint8 Bytes)
      - prefix_domain_info_document.bin/.idx (Uint8 Bytes)
    """
    def __init__(
        self,
        path_prefix: str,
        seq_key: str = "seq",
        domain_key: str = "domain_info",
        id_key: str = "entry_id",
        **kwargs
    ):
        super().__init__()
        # 1. Sequence Dataset
        self.seq_prefix = f"{path_prefix}_{seq_key}_document"
        if os.path.exists(self.seq_prefix + ".idx"):
            self.seq_dataset = IndexedDataset(self.seq_prefix)
        else:
            raise FileNotFoundError(f"Sequence dataset not found at {self.seq_prefix}")

        # 2. ID Dataset (Optional)
        self.id_prefix = f"{path_prefix}_{id_key}_document"
        self.id_dataset = None
        if os.path.exists(self.id_prefix + ".idx"):
            self.id_dataset = IndexedDataset(self.id_prefix)
            
        # 3. Domain Dataset (Optional)
        self.domain_prefix = f"{path_prefix}_{domain_key}_document"
        self.domain_dataset = None
        if os.path.exists(self.domain_prefix + ".idx"):
            self.domain_dataset = IndexedDataset(self.domain_prefix)

    def __len__(self) -> int:
        return len(self.seq_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1. Sequence (Int Tensor)
        seq_tokens = self.seq_dataset[idx]
        
        # 2. Entry ID (Bytes -> String)
        entry_id = str(idx)
        if self.id_dataset:
            # Retrieve bytes (uint8 array)
            id_bytes = self.id_dataset[idx]
            # Decode to string
            entry_id = id_bytes.tobytes().decode('utf-8')
            
        # 3. Domain Info (Bytes -> JSON String -> Object)
        domain_positions = []
        if self.domain_dataset:
            domain_bytes = self.domain_dataset[idx]
            json_str = domain_bytes.tobytes().decode('utf-8')
            try:
                domain_positions = json.loads(json_str)
            except:
                domain_positions = []

        return {
            "entry_id": entry_id,
            "seq": torch.tensor(seq_tokens, dtype=torch.long),
            "domain_positions": domain_positions, # This is now the nested list [ [[1,100]], ... ]
        }

if __name__ == "__main__":
    # Test Code
    path_prefix = "/storage/yuanfajieLab/yuanfajie/datasets/AFDB/Processed_TED_data/"
    print(f"Loading ProteinDataset from {path_prefix}...")

    # Note: id_key matches the --seq-name-key arg in preprocess
    dataset = ProteinDataset(
        path_prefix=path_prefix, 
        seq_key="seq", 
        domain_key="domain_info",
        id_key="entry_id"
    )
    
    print(f"Dataset successfully loaded.")
    print(f"Total samples: {len(dataset)}")
    
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, collate_fn=lambda x:torch.stack([i['seq'] for i in x]))
    for batch in tqdm(dataloader):
        continue
    # if len(dataset) > 0:
    #     num_samples_to_show = min(3, len(dataset))
    #     print(f"\n--- Checking first {num_samples_to_show} samples ---")
        
    #     for i in range(num_samples_to_show):
    #         sample = dataset[i]
    #         print(f"\n[Sample {i}]")
    #         print(f"  Entry ID: {sample['entry_id']}")
    #         print(f"  Seq (Token IDs, 10 tokens): {sample['seq'][:10]}")
    #         print(f"  Seq Length: {len(sample['seq'])}")
    #         print(f"  Domain Positions: {sample['domain_positions']}")
            