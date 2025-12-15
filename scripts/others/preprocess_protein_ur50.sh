# python tools/preprocess_protein.py \
#     --input /storage/yuanfajieLab/yuanfajie/datasets/AFDB/TED_data_tmp.jsonl \
#     --output-prefix /storage/yuanfajieLab/yuanfajie/datasets/AFDB/Processed_TED_data_tmp/ \
#     --seq-key seq \
#     --seq-name-key entry_id \
#     --domain-key domain_info \
#     --tokenizer-type HuggingFaceTokenizer --tokenizer-model airkingbd/dplm_150m \
#     --workers 32


python tools/preprocess_protein_ur50.py \
    --input /storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/data_processing/uniref50/uniref50_train.jsonl \
    --output-prefix /storage/yuanfajieLab/yuanfajie/datasets/dplm_ur50_index/train \
    --seq-key seq \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model airkingbd/dplm_150m \
    --workers 96

python tools/preprocess_protein_ur50.py \
    --input /storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/data_processing/uniref50/uniref50_val.jsonl \
    --output-prefix /storage/yuanfajieLab/yuanfajie/datasets/dplm_ur50_index/val \
    --seq-key seq \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model airkingbd/dplm_150m \
    --workers 96

    # --partitions 8  # <--- 关键修改：将大文件切分成 8 份并行处理