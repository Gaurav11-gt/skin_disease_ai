#!/usr/bin/env bash
# Full experiment grid for the paper: 5 models x 3 seeds (+ ablations). ~ a few GPU-hours on a Colab T4/A100.
set -e
DATA=${1:-data/processed}      # use data/splits if you skipped preprocess.py
FLAG=${2:---preprocessed}      # pass "" if you skipped preprocess.py
for SEED in 42 43 44; do
  for M in simple_cnn resnet50 effnet_b0 hybrid_resnet50 hybrid_effnet; do
    python train.py --model $M --data_dir $DATA $FLAG --seed $SEED --epochs 30
  done
done
# Ablations on the proposed model (seed 42 is enough, 3 seeds is better)
mkdir -p outputs_ablation
python train.py --model hybrid_effnet --data_dir $DATA $FLAG --seed 42 --no_class_weights --out_dir outputs_ablation/no_class_weights
python train.py --model hybrid_effnet --data_dir $DATA $FLAG --seed 42 --freeze_epochs 0 --out_dir outputs_ablation/no_freeze_stage
python train.py --model hybrid_effnet --data_dir data/splits --seed 42 --out_dir outputs_ablation/no_preprocessing
python train.py --model hybrid_effnet --data_dir $DATA $FLAG --seed 42 --no_pretrained --out_dir outputs_ablation/no_pretraining
python aggregate.py
