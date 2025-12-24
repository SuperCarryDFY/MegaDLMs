cd /storage/yuanfajieLab/yuanfajie/fengyuan/MegaDLMs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate megadlms

source envs/.env; bash examples/dlm_training/dlm_pretrain_protein_1B.sh
# source envs/.env; bash examples/dlm_training/dlm_pretrain_protein.sh convert_ckpt
