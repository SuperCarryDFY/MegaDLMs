source ~/miniconda3/etc/profile.d/conda.sh

DIR=$1
remasking="random low_confidence"

for file_path in $DIR/*; do
    echo "Processing $file_path"
    # get the file name 
    file_name=$(basename $file_path)
    for remasking_type in $remasking; do
        python examples/dlm_generation/dlm_inference_dist.py --model_dir_path $file_path/hf --remasking $remasking_type
    done
done