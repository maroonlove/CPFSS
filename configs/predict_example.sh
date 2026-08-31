#!/usr/bin/env bash
set -euo pipefail

python scripts/ion_classify/predict.py \
  --input-csv data/ion_classify/processed/predict.csv \
  --checkpoint outputs/train_run/checkpoints/best_model.pt \
  --class0-seq-root data/ion_classify/embeddings/class0/esm2_sequence \
  --class0-seq-att-root data/ion_classify/embeddings/class0/esm3_function \
  --class0-struc-att-root data/ion_classify/embeddings/class0/saprot_structure \
  --class1-seq-root data/ion_classify/embeddings/class1/esm2_sequence \
  --class1-seq-att-root data/ion_classify/embeddings/class1/esm3_function \
  --class1-struc-att-root data/ion_classify/embeddings/class1/saprot_structure \
  --class0-subdirs 10 \
  --class1-subdirs 41 \
  --output-dir outputs/predict_run \
  --batch-size 64 \
  --num-workers 8 \
  --device cuda:1
