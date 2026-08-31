# Ion Classify Data directory
Recommended layout:

```text
data/ion_classify/
├── processed/
│   ├── train.csv
│   └── test.csv
└── embeddings/
    ├── class0/
    │   ├── esm2_sequence/
    │   ├── esm3_function/
    │   └── saprot_structure/
    └── class1/
        ├── esm2_sequence/
        ├── esm3_function/
        └── saprot_structure/
```

The metadata CSV files must contain at least:

```csv
protein_id,class,label
P00001,0,0
P00002,1,6
```

Feature files should be named `<protein_id>.pt`. Both flat layouts, such as
`root/P00001.pt`, and subdirectory layouts, such as `root/0/P00001.pt`, are supported.
# GO Predict Data directory
Recommended layout:

```text
data/go_predict/
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

Each propagated GO label file is expected to contain at least three whitespace-separated columns:

```text
protein_id GO:xxxxxxx bp
```

The DIAMOND `.res` file is expected to contain at least:

```text
query_protein_id subject_train_protein_id score
```

The merged embedding file must contain:

```python
data["embeddings"][pid]["seq_token"]
data["embeddings"][pid]["sequence"]
data["embeddings"][pid]["struc"]
```