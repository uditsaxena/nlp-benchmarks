#!/usr/bin/env bash
cd ../../..
dataset="lex1"
test_dataset="ng20"
combined_datasets="ag_news---lex1"
depth=9
model_folder="models/VDCNN/VDCNN_${dataset}_depth@${depth}"
epoch_size=5000
batch_size=128
iterations=$(($epoch_size*10))
halving=$((3*$epoch_size))

python -m src.main --dataset "${dataset}" \
					--test_dataset "${test_dataset}" \
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
                    --test_only 0 \
					--model_load_path "models/VDCNN/ag_news-2000_model.pt" \
					--joint_training True \
					--joint_ratio 0.02 \
					--combined_datasets "${combined_datasets}"