# Preprocessing Instructions

All preprocessing scripts are in `scripts/preprocess/`. They use command-line arguments.

## 1. Generate ESM2 Sequence Embeddings

```bash
python scripts/preprocess/esm2.py \
  --input-dir data/raw/fasta \
  --output-dir data/ion_classify/embeddings/class0/esm2_sequence \
  --model-path /path/to/esm2_t33_650M_UR50D.pt \
  --repr-layer 33 \
  --device cuda:0
```

## 2. Generate ESM3 Sequence/Function Embeddings

```bash
python scripts/preprocess/esm3.py \
  --input-dir data/raw/fasta \
  --interpro-dir data/raw/interpro \
  --output-dir data/ion_classify/embeddings/class0/esm3_function \
  --interpro-map /path/to/interpro_29026_to_keywords_58641.csv \
  --hf-token "$HF_TOKEN" \
  --hf-endpoint https://hf-mirror.com \
  --device cuda:0
```

## 3. Generate SaProt Structure Embeddings

```bash
python scripts/preprocess/saprot.py \
  --input-dir data/raw/pdb \
  --output-dir data/ion_classify/embeddings/class0/saprot_structure \
  --model-path /path/to/SaProt_650M_AF2 \
  --foldseek-path /path/to/foldseek \
  --saprot-code-dir /path/to/SaProt \
  --chain A \
  --device cuda:0
```

## 4. Generate ESM3-predicted Structures

```bash
python scripts/preprocess/esm3_structure.py \
  --input-dir data/raw/fasta \
  --output-dir data/raw/pdb \
  --hf-token "$HF_TOKEN" \
  --hf-endpoint https://hf-mirror.com \
  --num-steps 8 \
  --device cuda:0
```

## 5. Run InterProScan

```bash
python scripts/preprocess/interpro.py \
  --input-dir data/raw/fasta \
  --output-dir data/raw/interpro \
  --interproscan-bin /path/to/interproscan.sh \
  --cpu 32
```

If InterProScan has already been run, split an existing TSV only:

```bash
python scripts/preprocess/interpro.py \
  --input-dir data/raw/fasta \
  --output-dir data/raw/interpro \
  --interproscan-bin /path/to/interproscan.sh \
  --tsv-output data/raw/interpro/interproscan.tsv \
  --split-only
```

## Notes

- Input directories should contain raw FASTA or PDB files.
- Output directories will store embeddings for downstream tasks.
- Adjust `--device` according to available GPU.
