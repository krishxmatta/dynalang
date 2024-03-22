#! /bin/bash

task=$1
name=$2
device=$3
seed=$4

shift
shift
shift
shift

python dynalang/train.py \
  --run.script train_offline \
  --logdir ~/logdir/androidenv/$name \
  --offline_datadir /home/rmendonc/logdir/androidenv/test_0/episodes \
  --use_wandb False \
  --task $task \
  --envs.amount 1 \
  --seed $seed \
  --encoder.mlp_keys orientation,timedelta \
  --decoder.mlp_keys orientation,timedelta \
  --decoder.vector_dist onehot \
  --batch_size 16 \
  --batch_length 256 \
  --run.train_ratio 32 \
  "$@"
