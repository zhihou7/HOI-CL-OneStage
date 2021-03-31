#!/bin/bash -e

# Download our pretrained model on HICO-DET dataset.
mkdir ../output
echo "Downloading our pretrained models ..."
gdown "https://drive.google.com/uc?id=1J-C2z9ZhJCJd3e3MwpgdXEqgvxL5stue" -O hico_det_pretrained.pkl
gdown "https://drive.google.com/uc?id=13aytw34aNUYlSp9_ASMBcqX3nrvC-Olf" -O ../output/hico_det_pretrained_agnostic.pkl


python train_net.py --eval-only --num-gpus 1 \
  --config-file configs/HICO-DET/interacting_objects_R_50_FPN.yaml \
  MODEL.WEIGHTS hico_det_pretrained.pkl \
  OUTPUT_DIR ./output/HICO_interacting_objects_zs

python train_net.py --eval-only --num-gpus 1 \
  --config-file configs/HICO-DET/interaction_zero_shot_R_50_FPN.yaml \
  MODEL.WEIGHTS ../output/hico_det_pretrained_agnostic.pkl \
  OUTPUT_DIR ./output/HICO_interacting_objects_zs_ag

python train_net.py --config-file configs/HICO-DET/interaction_zero_shot_R_50_FPN.yaml