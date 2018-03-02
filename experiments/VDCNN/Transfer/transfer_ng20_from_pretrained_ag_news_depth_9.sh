#!/usr/bin/env bash
cd ../../../
dataset="ng20"
depth=9
model_folder="models/VDCNN/VDCNN_transfer_${dataset}_depth@${depth}"
epoch_size=5000
batch_size=128
iterations=$(($epoch_size*10))
halving=$((3*$epoch_size))

python -m src.VDCNN --dataset "${dataset}" \
					--transfer_weights True \
					--target_transfer_ratio 1.0 \
					--num_prev_classes 4 \
                    --model_folder "${model_folder}" \
                    --depth ${depth} \
                    --maxlen 1024 \
                    --chunk_size 2048 \
                    --batch_size ${batch_size} \
                    --test_batch_size ${batch_size} \
                    --test_interval 4 \
                    --iterations ${iterations} \
                    --lr 0.01 \
                    --lr_halve_interval ${halving} \
                    --seed 1337 \
                    --num_embedding_features 100 \
                    --model_load_path "models/VDCNN/AgNews_15000_model.pt"