# IonFusion

This repository implements protein-function prediction using sequence, function, and structure embeddings.

## Tasks

1. **Ion-channel / transporter classification**  
2. **Multi-label Gene Ontology (GO) prediction**  

## Dataset and Model

Dataset and pretrained IonFusion model weights (used in the IonFusion paper):
[Google Drive Link](https://drive.google.com/file/d/1TNcsLS6Qwrc1TKEwZRpi1ENrWz15VAc3/view?usp=drive_link)

## Repository Layout

```text
.
├── scripts/
│   ├── ion_classify/
│   ├── go_predict/
│   └── preprocess/  # see preprocess/README.md
├── src/
├── configs/
├── data/
├── outputs/
├── README_GO_prediction.md
├── requirements.txt
├── environment.yml
└── .gitignore
```

## Environment

```bash
# Conda
conda env create -f environment.yml
conda activate <env_name>

# or pip
pip install -r requirements.txt
```

## Task 1: Ion Classification

Train a two-stage hierarchical classifier using sequence and structure embeddings.

```bash
bash configs/train_example.sh
```

Test and Evaluate:

```bash
bash configs/predict_example.sh
bash configs/evaluate_example.sh
```

More details on inputs/outputs in `data/README.md`.

## Task 2: GO Prediction

Perform multi-label GO term prediction with DIAMOND fusion.

```bash
bash configs/train_go_bp10_example.sh
bash configs/predict_go_bp10_example.sh
bash configs/evaluate_go_bp10_example.sh
```

Detailed workflow is in [`README_GO_prediction.md`](README_GO_prediction.md).

## Notes

- Write experiment outputs under `outputs/`.
- Preprocessing instructions are in `scripts/preprocess/README.md`.
