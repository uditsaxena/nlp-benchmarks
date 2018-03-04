#!/bin/bash
#
#SBATCH --job-name=mixed-005-us-vdcnn
#SBATCH --partition=m40-short
#SBATCH --output=5-test-root-%A.out
#SBATCH --error=5-test-root-%A.err
#SBATCH --gres=gpu:1

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

dataset="amazon_polarity"
test_dataset="yelp_polarity"
combined_datasets="amazon_polarity---yelp_polarity"
depth=29
model_folder="models/VDCNN/VDCNN_${combined_datasets}_depth@${depth}/5"
epoch_size=5000
batch_size=128
iterations=$(($epoch_size*10))
halving=$((3*$epoch_size))
test_interval=1000

module purge
module load python/3.5.2
module load cuda80/blas/8.0.44
module load cuda80/fft/8.0.44
module load cuda80/nsight/8.0.44
module load cuda80/profiler/8.0.44
module load cuda80/toolkit/8.0.44
pip install --user ipython h5py numpy scikit-learn pandas scipy torchvision requests
pip install  --user http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl

cd /home/usaxena/work/s18/lex/code/vdcnn/

# test
python -m src.VDCNN --dataset "${dataset}" \
                    --test_dataset "${test_dataset}" \
                    --model_folder "${model_folder}" \
                    --model_save_path "${model_folder}" \
                    --depth ${depth} \
                    --maxlen 1024 \
                    --chunk_size 2048 \
                    --batch_size ${batch_size} \
                    --test_batch_size ${batch_size} \
                    --test_interval ${test_interval} \
                    --iterations ${iterations} \
                    --lr 0.01 \
                    --lr_halve_interval ${halving} \
                    --seed 1337 \
                    --joint_training True \
                    --joint_test 1 \
                    --joint_ratio 0.05 \
                    --gpu \
                    --num_embedding_features 100 \
                    --model_load_path "${model_folder}/best_model.pt"


