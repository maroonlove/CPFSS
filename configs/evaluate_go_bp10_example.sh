#!/usr/bin/env bash
set -euo pipefail

python scripts/go_predict/evaluate_go.py \
  --namespace bp \
  --train-prop data/go_predict/processed/seqid_10/bp/bp_train_final.txt \
  --val-prop data/go_predict/processed/seqid_10/bp/bp_val_final.txt \
  --test-prop data/go_predict/processed/seqid_10/bp/bp_test_final.txt \
  --merged-emb-pt data/go_predict/embeddings/merge_3131_embeddings.pt \
  --obo-file data/go_predict/external/go.obo \
  --val-diamond-res data/go_predict/diamond/res/seqid10_bp_val_diamond.res \
  --test-diamond-res data/go_predict/diamond/res/seqid10_bp_test_diamond.res \
  --checkpoint-dir outputs/go_prediction/seqid_10/bp/checkpoints \
  --output-dir outputs/go_prediction/seqid_10/bp/final \
  --epochs 100 \
  --batch-size 64 \
  --device cuda:1
