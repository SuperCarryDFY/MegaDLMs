# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing protein data for pretraining."""
import argparse
import json
import os
import sys
import time
import gzip
import glob
import multiprocessing
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def _parse_domains_to_structure(self, domain_info_list):
        """
        Parses raw domain strings into the nested list structure.
        We do this here to validate, then dump back to JSON string for storage.
        Input: ["1-100", "150-200_210-250"]
        Output: [ [[1, 100]], [[150, 200], [210, 250]] ]
        """
        parsed_structure = []
        try:
            for domain_info in domain_info_list:
                group_intervals = []
                parts = domain_info.split("_")
                for part in parts:
                    start, end = map(int, part.split("-"))
                    group_intervals.append([start, end])
                parsed_structure.append(group_intervals)
        except Exception as e:
            # Fallback or error logging
            return []
        return parsed_structure

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        
        # 1. Sequence (Token IDs -> Int32/Uint16)
        seq_key = self.args.seq_key
        text = data[seq_key]
        doc_ids = Encoder.tokenizer.tokenize(text)
        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids.append(Encoder.tokenizer.eod)
        ids[seq_key] = doc_ids
        lens[seq_key] = len(doc_ids)
        # 2. Domain Info (Nested List -> JSON String -> Bytes -> Uint8)
        domain_key = self.args.domain_key
        if domain_key in data:
            domain_info_list = data[domain_key]
            # Convert to the desired nested structure first
            nested_domains = self._parse_domains_to_structure(domain_info_list)
            # Dump to JSON string, then encode to bytes (list of ints)
            json_bytes = list(json.dumps(nested_domains).encode('utf-8'))
            
            ids[domain_key] = json_bytes
            lens[domain_key] = len(json_bytes)
        
        # 3. Sequence Name (String -> Bytes -> Uint8)
        seq_name_key = self.args.seq_name_key
        if seq_name_key in data:
            name_str = str(data[seq_name_key]).strip()
            name_bytes = list(name_str.encode('utf-8'))
            
            ids[seq_name_key] = name_bytes
            lens[seq_name_key] = len(name_bytes)

        return ids, lens, len(json_line)

class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')

        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer, maxtasksperchild=1000)
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        # Define keys and their types
        # Seq -> Tokenizer Vocab Size (usually int32 or uint16)
        # Domain/ID -> Uint8 (Bytes)
        
        keys_config = {}
        keys_config[self.args.seq_key] = indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
        
        if self.args.domain_key:
            keys_config[self.args.domain_key] = np.uint8
        
        if self.args.seq_name_key:
            keys_config[self.args.seq_name_key] = np.uint8

        for key, dtype in keys_config.items():
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix, key, "document")
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix, key, "document")
            
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=dtype,
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        
        for i, (doc, lengths, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            
            for key in keys_config.keys():
                if key in doc:
                    builders[key].add_document(doc[key], [lengths[key]])
            
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        
        for key in keys_config.keys():
            builders[key].finalize(output_idx_files[key])

def get_args():
    parser = argparse.ArgumentParser()
    parser = _add_tokenizer_args(parser)
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON/JSONL')
    group.add_argument('--seq-key', type=str, default='seq',
                       help='Key for the protein sequence in json')
    group.add_argument('--domain-key', type=str, default=None,
                       help='Key for the domain info list in json')
    group.add_argument('--seq-name-key', type=str, default=None,
                       help='Key for the sequence name in json')
    group = parser.add_argument_group(title='tokenization process')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help='Number of worker processes to launch.')
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is preserved.')
    
    args = parser.parse_args()
    args.keep_empty = False
    
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'output_prefix': output_prefix}
    return file_names

def check_files_exist(in_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_names[i][key]):
            return False
    return True

def main():
    args = get_args()

    in_names = []
    if args.partitions == 1:
        file_names = {
            'partition': args.input,
            'output_prefix': args.output_prefix}
        in_names.append(file_names)
    else:
        in_file_names = glob.glob(args.input)
        
        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename, "r") as fin:
                    for fc, _ in enumerate(fin):
                        pass
                total_sample_count += (fc + 1)
            partition_size = (total_sample_count + args.partitions - 1) // args.partitions

        for idx in range(args.partitions):
            in_name = get_file_name(args, idx)
            in_names.append(in_name)

        partitions_present = check_files_exist(in_names, 'partition', args.partitions)

        if not partitions_present:
            partitioned_input_files = []
            for idx in range(args.partitions):
                partitioned_input_file = open(in_names[idx]['partition'], 'w')
                partitioned_input_files.append(partitioned_input_file)

            index = 0
            if args.keep_sequential_samples: line_count = 0
            for in_file_name in in_file_names:
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, 'rt')
                else:
                    fin = open(in_file_name, 'r', encoding='utf-8')

                for line in fin:
                    partitioned_input_files[index].write(line)
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0:
                            index += 1
                    else:
                        index = (index + 1) % args.partitions
                fin.close()
            
            for f in partitioned_input_files:
                f.close()

    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers//args.partitions)

    processes = []
    for name in in_names:
        p = multiprocessing.Process(target=partition.process_json_file,
                                    args=((name['partition'], name['output_prefix']),))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if args.partitions == 1:
        return

    # Merge partitions
    tokenizer = build_tokenizer(args)
    
    keys_config = {}
    keys_config[args.seq_key] = indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
    if args.domain_key:
        keys_config[args.domain_key] = np.uint8
    if args.seq_name_key:
        keys_config[args.seq_name_key] = np.uint8

    for key, dtype in keys_config.items():
        output_bin = "{}_{}_{}.bin".format(args.output_prefix, key, "document")
        output_idx = "{}_{}_{}.idx".format(args.output_prefix, key, "document")
        
        builder = indexed_dataset.IndexedDatasetBuilder(
            output_bin,
            dtype=dtype,
        )

        for name in in_names:
            p_prefix = name['output_prefix']
            full_p_prefix = "{}_{}_{}".format(p_prefix, key, "document")
            builder.add_index(full_p_prefix)
        
        builder.finalize(output_idx)

if __name__ == '__main__':
    main()