#!/bin/bash
#
#SBATCH --job-name=us_rnn_2
#SBATCH --mem=10000
#SBATCH --partition=longq
#SBATCH --output=us_experiment_%A.out
#SBATCH --error=us_experiment_%A.err

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-lstm-jobs.txt

module purge
module load python/3.5.2
pip install --user ipython h5py numpy scikit-learn pandas scipy torchvision requests gensim nltk matplotlib
cd /home/usaxena/work/lex

dataset="ng20"
depth=9
model_folder="models/VDCNN/VDCNN_${dataset}_depth@${depth}"
epoch_size=5000
batch_size=128
iterations=$(($epoch_size*10))
halving=$((3*$epoch_size))

python -m src.construct_graph --dataset "${dataset}" \
                    --model_folder "${model_folder}" \
                    --model_save_path "${model_folder}" \
                    --model_load_path "${model_folder}/best_model.pt" \
                    --depth ${depth} \
                    --maxlen 1024 \
                    --chunk_size 2048 \
                    --batch_size ${batch_size} \
                    --test_batch_size ${batch_size} \
                    --test_interval 1000 \
                    --iterations ${iterations} \
                    --lr 0.01 \
                    --lr_halve_interval ${halving} \
                    --seed 1337 \
                    --gpu

