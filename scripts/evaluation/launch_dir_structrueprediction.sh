source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldflow-env
DIR=$1

for file_path in $DIR/generation_output/*/*.fasta; do
    echo "Processing $file_path"
    python tools/esmfold/structure_prediction.py --fasta_path $file_path
done