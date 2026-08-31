#!/usr/bin/env bash
set -euo pipefail

python scripts/go_predict/predict_go.py \
  --namespace bp \
  --input-pids data/go_predict/processed/seqid_10/bp/bp_predict_ids.txt \
  --train-prop data/go_predict/processed/seqid_10/bp/bp_train_final.txt \
  --merged-emb-pt data/go_predict/embeddings/merge_3131_embeddings.pt \
  --obo-file data/go_predict/external/go.obo \
  --checkpoint-dir outputs/go_prediction/seqid_10/bp/checkpoints \
  --epochs 100 \
  --output-dir outputs/go_prediction/seqid_10/bp/predict \
  --threshold 0.5 \
  --top-k 0 \
  --batch-size 64 \
  --num-workers 0 \
  --device cuda:1
