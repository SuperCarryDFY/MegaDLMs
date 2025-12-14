# python tools/preprocess_data.py \
#     --input tools/example_data/data.jsonl \
#     --output-prefix tools/example_data/processed_data \
#     --tokenizer-type HuggingFaceTokenizer --tokenizer-model google/t5-v1_1-xxl \
#     --workers 4 --append-eod

# python tools/preprocess_protein.py \
#     --input /storage/yuanfajieLab/yuanfajie/datasets/AFDB/TED_data_tmp.jsonl \
#     --output-prefix /storage/yuanfajieLab/yuanfajie/datasets/AFDB/Processed_TED_data_tmp/ \
#     --seq-key seq \
#     --seq-name-key entry_id \
#     --domain-key domain_info \
#     --tokenizer-type HuggingFaceTokenizer --tokenizer-model airkingbd/dplm_150m \
#     --workers 32

python tools/preprocess_protein.py \
    --input /storage/yuanfajieLab/yuanfajie/datasets/AFDB/TED_data.jsonl \
    --output-prefix /storage/yuanfajieLab/yuanfajie/datasets/AFDB/Processed_TED_data/ \
    --seq-key seq \
    --seq-name-key entry_id \
    --domain-key domain_info \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model airkingbd/dplm_150m \
    --workers 96\
    --partitions 8  # <--- 关键修改：将大文件切分成 8 份并行处理