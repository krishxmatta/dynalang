#! /bin/bash

task=$1
name=$2
device=$3
seed=$4

shift
shift
shift
shift

python dynalang/sample.py \
  --run.script sample \
  --logdir ~/logdir/homegrid/$name \
  --use_wandb False \
  --task $task \
  --envs.amount 1 \
  --seed $seed \
  --encoder.mlp_keys token$ \
  --decoder.mlp_keys token$ \
  --decoder.vector_dist onehot \
  --batch_size 16 \
  --batch_length 256 \
  --run.train_ratio 32 \
  "$@"
