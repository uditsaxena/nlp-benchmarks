#!/usr/bin/env bash

#combined_datasets="ag_news---lex1"
#depth=49
#num_embedding_features=100

ratios=( 'c1' 'c2' 'c5' 'c10' 'c25' 'c50' 'c100' )
#ratios=( 'c1' ) #'c2' 'c5' 'c10' 'c25' 'c50' 'c100' )

for ratio in "${ratios[@]}"
do
# "./$ratio/train-c1.sh"
 sbatch "./$ratio/train-c1.sh"
done