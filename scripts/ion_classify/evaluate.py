#!/usr/bin/env python
"""Evaluate a trained hierarchical classifier on a test set."""

from __future__ import annotations

import sys
from pathlib import Path

import click as ck
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ion_classifier.data import (  # noqa: E402
    FeatureRoots,
    build_dataloader,
    build_items_from_csv,
    load_items_from_pickle,
    save_items_to_pickle,
)
from ion_classifier.engine import evaluate_stages, load_model_weights, predict_final  # noqa: E402
from ion_classifier.model import ProteinTransformer  # noqa: E402
from ion_classifier.utils import ensure_dir, get_device, save_dataframe, set_seed  # noqa: E402


@ck.command()
@ck.option("--test-csv", type=ck.Path(exists=True, dir_okay=False), default=None, help="Test CSV with columns protein_id,class,label.")
@ck.option("--test-items-pkl", type=ck.Path(exists=True, dir_okay=False), default=None, help="Optional cached test items pickle.")
@ck.option("--cache-test-items", type=ck.Path(dir_okay=False), default=None, help="Optional path to save built test items pickle.")
@ck.option("--checkpoint", "-ckpt", type=ck.Path(exists=True, dir_okay=False), required=True, help="Trained checkpoint path.")
@ck.option("--class0-seq-root", type=ck.Path(file_okay=False), default=None, help="Class 0 ESM2 token embedding root.")
@ck.option("--class0-seq-att-root", type=ck.Path(file_okay=False), default=None, help="Class 0 ESM3 function embedding root.")
@ck.option("--class0-struc-att-root", type=ck.Path(file_okay=False), default=None, help="Class 0 SaProt structure embedding root.")
@ck.option("--class1-seq-root", type=ck.Path(file_okay=False), default=None, help="Class 1 ESM2 token embedding root.")
@ck.option("--class1-seq-att-root", type=ck.Path(file_okay=False), default=None, help="Class 1 ESM3 function embedding root.")
@ck.option("--class1-struc-att-root", type=ck.Path(file_okay=False), default=None, help="Class 1 SaProt structure embedding root.")
@ck.option("--class0-subdirs", type=int, default=10, show_default=True)
@ck.option("--class1-subdirs", type=int, default=41, show_default=True)
@ck.option("--output-dir", "-o", type=ck.Path(file_okay=False), default="outputs/eval_run", show_default=True)
@ck.option("--batch-size", type=int, default=64, show_default=True)
@ck.option("--num-workers", type=int, default=4, show_default=True)
@ck.option("--prefetch-factor", type=int, default=4, show_default=True)
@ck.option("--device", type=str, default="cuda:0", show_default=True)
@ck.option("--seed", type=int, default=0, show_default=True)
@ck.option("--sample-every", type=int, default=1, show_default=True, help="If >1, randomly keep one row in every N CSV rows.")
@ck.option("--input-dim", type=int, default=2560, show_default=True)
@ck.option("--model-dim", type=int, default=512, show_default=True)
@ck.option("--num-heads", type=int, default=8, show_default=True)
@ck.option("--num-layers", type=int, default=4, show_default=True)
@ck.option("--dropout", type=float, default=0.1, show_default=True)
@ck.option("--strict-checkpoint/--non-strict-checkpoint", default=True, show_default=True)
@ck.option("--strict-missing/--skip-missing", default=False, show_default=True)
def main(
    test_csv,
    test_items_pkl,
    cache_test_items,
    checkpoint,
    class0_seq_root,
    class0_seq_att_root,
    class0_struc_att_root,
    class1_seq_root,
    class1_seq_att_root,
    class1_struc_att_root,
    class0_subdirs,
    class1_subdirs,
    output_dir,
    batch_size,
    num_workers,
    prefetch_factor,
    device,
    seed,
    sample_every,
    input_dim,
    model_dim,
    num_heads,
    num_layers,
    dropout,
    strict_checkpoint,
    strict_missing,
):
    """Evaluate a checkpoint and export prediction/metric CSV files."""

    set_seed(seed)
    device = get_device(device)
    output_dir = ensure_dir(output_dir)

    if test_items_pkl:
        test_items = load_items_from_pickle(test_items_pkl)
    else:
        if not test_csv:
            raise ck.UsageError("Either --test-csv or --test-items-pkl must be provided.")
        required_roots = {
            "--class0-seq-root": class0_seq_root,
            "--class0-seq-att-root": class0_seq_att_root,
            "--class0-struc-att-root": class0_struc_att_root,
            "--class1-seq-root": class1_seq_root,
            "--class1-seq-att-root": class1_seq_att_root,
            "--class1-struc-att-root": class1_struc_att_root,
        }
        missing = [name for name, value in required_roots.items() if value is None]
        if missing:
            raise ck.UsageError("Feature roots are required when --test-csv is used: " + ", ".join(missing))
        class0_roots = FeatureRoots(Path(class0_seq_root), Path(class0_seq_att_root), Path(class0_struc_att_root), class0_subdirs)
        class1_roots = FeatureRoots(Path(class1_seq_root), Path(class1_seq_att_root), Path(class1_struc_att_root), class1_subdirs)
        test_items = build_items_from_csv(
            csv_path=test_csv,
            class0_roots=class0_roots,
            class1_roots=class1_roots,
            seed=seed,
            sample_every=sample_every,
            strict=strict_missing,
        )
        if cache_test_items:
            save_items_to_pickle(test_items, cache_test_items)

    test_loader = build_dataloader(
        test_items,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=prefetch_factor,
    )

    model = ProteinTransformer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    load_model_weights(model, checkpoint, device=device, strict=strict_checkpoint)
    criterion = nn.CrossEntropyLoss()

    stage_metrics = evaluate_stages(model, test_loader, criterion, device)
    save_dataframe(pd.DataFrame([stage_metrics]), output_dir / "stage_metrics.csv")

    predictions_df, overall_df, per_class_df = predict_final(model, test_loader, device)
    save_dataframe(predictions_df, output_dir / "test_predictions.csv")
    save_dataframe(overall_df, output_dir / "overall_metrics.csv")
    save_dataframe(per_class_df, output_dir / "per_class_metrics.csv")

    print("Overall metrics:")
    print(overall_df.to_string(index=False))
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
