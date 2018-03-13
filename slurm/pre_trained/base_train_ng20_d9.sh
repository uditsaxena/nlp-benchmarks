#!/bin/bash
#
#SBATCH --job-name=ng-transfer-us-vdcnn
#SBATCH --partition=m40-short
#SBATCH --output=ng-transfer-us-vdcnn-100f-emb%A.out
#SBATCH --error=ng-transfer-us-vdcnn-100f-emb%A.err
#SBATCH --gres=gpu:1

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

dataset="ng20"
depth=9
model_folder="models/VDCNN/VDCNN_transfer_${dataset}_depth@${depth}_100_emb"
epoch_size=5000
batch_size=128
iterations=$(($epoch_size*10))
halving=$((3*$epoch_size))

module purge
module load python/3.5.2
module load cuda80/blas/8.0.44
module load cuda80/fft/8.0.44
module load cuda80/nsight/8.0.44
module load cuda80/profiler/8.0.44
module load cuda80/toolkit/8.0.44
pip install --user ipython h5py numpy scikit-learn pandas scipy torchvision requests
pip install --user http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl

cd /home/usaxena/work/s18/lex/code/vdcnn/
python -m src.main --dataset "${dataset}" \
                    --model_folder "${model_folder}" \
                    --model_save_path "${model_folder}" \
                    --depth ${depth} \
                    --maxlen 1024 \
                    --chunk_size 2048 \
                    --batch_size ${batch_size} \
                    --test_batch_size ${batch_size} \
                    --test_interval ${epoch_size} \
                    --iterations ${iterations} \
                    --lr 0.01 \
                    --lr_halve_interval ${halving} \
                    --seed 1337 \
                    --gpu \
                    --num_embedding_features 100

