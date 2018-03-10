#!/bin/bash
#
#SBATCH --job-name=mixed-005-us-vdcnn
#SBATCH --partition=m40-short
#SBATCH --output=5-train-%A.out
#SBATCH --error=5-train-%A.err
#SBATCH --gres=gpu:1

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

dataset="ag_news"
test_dataset="ng20"
combined_datasets="ng20---ag_news"
depth=9
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

# train joint, 
python -m src.main --dataset "${dataset}" \
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
                    --num_embedding_features 100 \
                    --lr_halve_interval ${halving} \
                    --seed 1337 \
                    --test_only 0 \
                    --model_load_path "models/VDCNN/ag_news-2000_model.pt" \
                    --joint_training True \
                    --joint_ratio 0.05 \
                    --combined_datasets "${combined_datasets}" \
                    --shuffle \
                    --gpu


