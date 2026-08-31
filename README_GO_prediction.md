# Downstream GO prediction with GO propagation and DIAMOND fusion

This module implements the downstream multi-label GO prediction task after sequence/function/structure embeddings are prepared.

The final result is intentionally restricted to the requested logic:

1. Train or load the last three checkpoints, normally `epoch_098.pth`, `epoch_099.pth`, and `epoch_100.pth`.
2. Average the prediction probabilities of these three models on both validation and test sets.
3. Apply GO score propagation to model probabilities: each ancestor term score is forced to be at least the child term score, while BP/MF/CC root terms are excluded.
4. Build DIAMOND GO probability matrices from the training annotations and DIAMOND `.res` files.
5. Search alpha on validation:

```text
fused_probs = alpha * model_probs + (1 - alpha) * diamond_probs
```

The validation alpha is selected by maximum Fmax, then smaller Smin, then larger Aupr.

If the validation-selected alpha is `0.00`, the namespace default is used:

```text
bp = 0.57
cc = 0.91
mf = 0.64
```

If the validation-selected alpha is not `0.00`, the validation-selected value is used.

Only DIAMOND-fused final outputs are saved.

## Expected input layout

```text
data/go_predict
├── external/
│   └── go.obo
├── embeddings/
│   └── merge_3131_embeddings.pt
├── processed/
│   └── seqid_10/
│       └── bp/
│           ├── bp_train_final.txt
│           ├── bp_val_final.txt
│           └── bp_test_final.txt
└── diamond/
    └── res/
        ├── seqid10_bp_val_diamond.res
        └── seqid10_bp_test_diamond.res
```

The propagated GO label files are whitespace-separated and must contain at least:

```text
protein_id GO:xxxxxxx bp
```

The merged embedding file must contain:

```python
data["embeddings"][pid]["seq_token"]  # [1280]
data["embeddings"][pid]["sequence"]   # [L, 1536]
data["embeddings"][pid]["struc"]      # [L, 1280]
```

## Train and evaluate

```bash
python scripts/go_predict/train_go.py \
  --namespace bp \
  --seq-identity 10 \
  --train-prop data/go_predict/processed/seqid_10/bp/bp_train_final.txt \
  --val-prop data/go_predict/processed/seqid_10/bp/bp_val_final.txt \
  --test-prop data/go_predict/processed/seqid_10/bp/bp_test_final.txt \
  --merged-emb-pt data/go_predict/embeddings/merge_3131_embeddings.pt \
  --obo-file data/go_predict/external/go.obo \
  --val-diamond-res data/go_predict/diamond/res/seqid10_bp_val_diamond.res \
  --test-diamond-res data/go_predict/diamond/res/seqid10_bp_test_diamond.res \
  --output-dir outputs/go_prediction/seqid_10/bp \
  --epochs 100 \
  --batch-size 64 \
  --device cuda:1
```

The final files are:

```text
outputs/go_prediction/seqid_10/bp/
├── checkpoints/
│   ├── epoch_098.pth
│   ├── epoch_099.pth
│   └── epoch_100.pth
└── final/
    ├── diamond_fused_test_probs.pt
    ├── diamond_fused_test_go_scores.csv
    ├── predictions_test.csv
    ├── metrics.csv
    ├── alpha_selection.csv
    └── run_config.json
```

## Evaluate from existing checkpoints only

```bash
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
```
