#!/usr/bin/env bash
set -euo pipefail

python scripts/ion_classify/train.py \
  --train-csv data/ion_classify/processed/train.csv \
  --test-csv data/ion_classify/processed/test.csv \
  --class0-seq-root data/ion_classify/embeddings/class0/esm2_sequence \
  --class0-seq-att-root data/ion_classify/embeddings/class0/esm3_function \
  --class0-struc-att-root data/ion_classify/embeddings/class0/saprot_structure \
  --class1-seq-root data/ion_classify/embeddings/class1/esm2_sequence \
  --class1-seq-att-root data/ion_classify/embeddings/class1/esm3_function \
  --class1-struc-att-root data/ion_classify/embeddings/class1/saprot_structure \
  --class0-subdirs 10 \
  --class1-subdirs 41 \
  --output-dir outputs/train_run \
  --epochs 10 \
  --batch-size 64 \
  --num-workers 8 \
  --device cuda:1
