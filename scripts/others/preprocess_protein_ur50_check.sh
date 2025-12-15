

python tools/preprocess_protein_ur50_check.py \
    --input /storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/data_processing/uniref50/uniref50_val.jsonl \
    --output-prefix /storage/yuanfajieLab/yuanfajie/fengyuan/MegaDLMs/scripts/others/tmp \
    --seq-key seq \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model airkingbd/dplm_150m \
    --workers 1

    # --partitions 8  # <--- 关键修改：将大文件切分成 8 份并行处理