#!/usr/bin/env python
"""Train the hierarchical transporter / ion-channel classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import click as ck
import pandas as pd
import torch
import torch.nn as nn
from tqdm import trange

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ion_classifier.data import (  # noqa: E402
    FeatureRoots,
    build_dataloader,
    build_items_from_csv,
    load_items_from_pickle,
    save_items_to_pickle,
)
from ion_classifier.engine import (  # noqa: E402
    evaluate_stages,
    load_model_weights,
    predict_final,
    save_checkpoint,
    train_one_epoch,
)
from ion_classifier.model import ProteinTransformer  # noqa: E402
from ion_classifier.utils import ensure_dir, get_device, save_dataframe, set_seed  # noqa: E402


def _build_roots(
    class0_seq_root,
    class0_seq_att_root,
    class0_struc_att_root,
    class0_subdirs,
    class1_seq_root,
    class1_seq_att_root,
    class1_struc_att_root,
    class1_subdirs,
):
    class0_roots = FeatureRoots(
        seq_root=Path(class0_seq_root),
        seq_att_root=Path(class0_seq_att_root),
        struc_att_root=Path(class0_struc_att_root),
        num_subdirs=class0_subdirs,
    )
    class1_roots = FeatureRoots(
        seq_root=Path(class1_seq_root),
        seq_att_root=Path(class1_seq_att_root),
        struc_att_root=Path(class1_struc_att_root),
        num_subdirs=class1_subdirs,
    )
    return class0_roots, class1_roots


def _load_or_build_items(
    csv_path,
    items_pkl,
    cache_pkl,
    class0_roots,
    class1_roots,
    seed,
    sample_every,
    strict_missing,
):
    if items_pkl:
        return load_items_from_pickle(items_pkl)
    if not csv_path:
        raise ck.UsageError("Either CSV input or items PKL input must be provided.")

    items = build_items_from_csv(
        csv_path=csv_path,
        class0_roots=class0_roots,
        class1_roots=class1_roots,
        seed=seed,
        sample_every=sample_every,
        strict=strict_missing,
    )
    if cache_pkl:
        save_items_to_pickle(items, cache_pkl)
    return items


@ck.command()
@ck.option("--train-csv", type=ck.Path(exists=True, dir_okay=False), default=None, help="Training CSV with columns protein_id,class,label.")
@ck.option("--test-csv", type=ck.Path(exists=True, dir_okay=False), default=None, help="Test CSV with columns protein_id,class,label.")
@ck.option("--train-items-pkl", type=ck.Path(exists=True, dir_okay=False), default=None, help="Optional cached training items pickle.")
@ck.option("--test-items-pkl", type=ck.Path(exists=True, dir_okay=False), default=None, help="Optional cached test items pickle.")
@ck.option("--cache-train-items", type=ck.Path(dir_okay=False), default=None, help="Optional path to save built training items pickle.")
@ck.option("--cache-test-items", type=ck.Path(dir_okay=False), default=None, help="Optional path to save built test items pickle.")
@ck.option("--class0-seq-root", type=ck.Path(file_okay=False), default=None, help="Class 0 ESM2 token embedding root.")
@ck.option("--class0-seq-att-root", type=ck.Path(file_okay=False), default=None, help="Class 0 ESM3 function embedding root.")
@ck.option("--class0-struc-att-root", type=ck.Path(file_okay=False), default=None, help="Class 0 SaProt structure embedding root.")
@ck.option("--class1-seq-root", type=ck.Path(file_okay=False), default=None, help="Class 1 ESM2 token embedding root.")
@ck.option("--class1-seq-att-root", type=ck.Path(file_okay=False), default=None, help="Class 1 ESM3 function embedding root.")
@ck.option("--class1-struc-att-root", type=ck.Path(file_okay=False), default=None, help="Class 1 SaProt structure embedding root.")
@ck.option("--class0-subdirs", type=int, default=10, show_default=True, help="Number of subdirectories searched for class 0 features.")
@ck.option("--class1-subdirs", type=int, default=41, show_default=True, help="Number of subdirectories searched for class 1 features.")
@ck.option("--output-dir", "-o", type=ck.Path(file_okay=False), default="outputs/train_run", show_default=True, help="Directory for checkpoints and result CSV files.")
@ck.option("--epochs", type=int, default=10, show_default=True)
@ck.option("--batch-size", type=int, default=64, show_default=True)
@ck.option("--num-workers", type=int, default=4, show_default=True)
@ck.option("--prefetch-factor", type=int, default=4, show_default=True)
@ck.option("--lr", type=float, default=1e-4, show_default=True)
@ck.option("--weight-decay", type=float, default=0.01, show_default=True)
@ck.option("--grad-clip", type=float, default=1.0, show_default=True)
@ck.option("--device", type=str, default="cuda:0", show_default=True)
@ck.option("--seed", type=int, default=0, show_default=True)
@ck.option("--sample-every", type=int, default=1, show_default=True, help="If >1, randomly keep one row in every N CSV rows.")
@ck.option("--input-dim", type=int, default=2560, show_default=True)
@ck.option("--model-dim", type=int, default=512, show_default=True)
@ck.option("--num-heads", type=int, default=8, show_default=True)
@ck.option("--num-layers", type=int, default=4, show_default=True)
@ck.option("--dropout", type=float, default=0.1, show_default=True)
@ck.option("--resume-checkpoint", type=ck.Path(exists=True, dir_okay=False), default=None, help="Optional checkpoint for continuing training.")
@ck.option("--amp/--no-amp", default=True, show_default=True, help="Use automatic mixed precision on CUDA.")
@ck.option("--strict-missing/--skip-missing", default=False, show_default=True, help="Raise error for missing feature files instead of skipping them.")
def main(
    train_csv,
    test_csv,
    train_items_pkl,
    test_items_pkl,
    cache_train_items,
    cache_test_items,
    class0_seq_root,
    class0_seq_att_root,
    class0_struc_att_root,
    class1_seq_root,
    class1_seq_att_root,
    class1_struc_att_root,
    class0_subdirs,
    class1_subdirs,
    output_dir,
    epochs,
    batch_size,
    num_workers,
    prefetch_factor,
    lr,
    weight_decay,
    grad_clip,
    device,
    seed,
    sample_every,
    input_dim,
    model_dim,
    num_heads,
    num_layers,
    dropout,
    resume_checkpoint,
    amp,
    strict_missing,
):
    """Train and evaluate the two-stage classifier."""

    set_seed(seed)
    device = get_device(device)
    output_dir = ensure_dir(output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")

    class0_roots = class1_roots = None
    if train_csv or test_csv:
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
            raise ck.UsageError("Feature roots are required when CSV inputs are used: " + ", ".join(missing))
        class0_roots, class1_roots = _build_roots(
            class0_seq_root,
            class0_seq_att_root,
            class0_struc_att_root,
            class0_subdirs,
            class1_seq_root,
            class1_seq_att_root,
            class1_struc_att_root,
            class1_subdirs,
        )

    train_items = _load_or_build_items(
        train_csv,
        train_items_pkl,
        cache_train_items,
        class0_roots,
        class1_roots,
        seed,
        sample_every,
        strict_missing,
    )
    test_items = _load_or_build_items(
        test_csv,
        test_items_pkl,
        cache_test_items,
        class0_roots,
        class1_roots,
        seed,
        sample_every,
        strict_missing,
    )

    train_loader = build_dataloader(
        train_items,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=prefetch_factor,
    )
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_epoch = 1
    best_loss = float("inf")
    if resume_checkpoint:
        ckpt = load_model_weights(model, resume_checkpoint, device=device, strict=True)
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_loss = float(ckpt.get("train_loss", best_loss))
        print(f"Loaded checkpoint: {resume_checkpoint}")

    history = []
    for epoch in trange(start_epoch, epochs + 1, desc="[Epoch]"):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=amp,
            grad_clip=grad_clip,
            epoch=epoch,
        )
        history.append({"epoch": epoch, "train_loss": train_loss})
        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.6f}")

        last_ckpt = checkpoint_dir / "last_model.pt"
        save_checkpoint(model, optimizer, last_ckpt, epoch=epoch, train_loss=train_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            best_ckpt = checkpoint_dir / "best_model.pt"
            save_checkpoint(model, optimizer, best_ckpt, epoch=epoch, train_loss=train_loss)
            print(f"Saved new best checkpoint: {best_ckpt}")

    history_df = pd.DataFrame(history)
    save_dataframe(history_df, output_dir / "train_history.csv")

    # Use the best checkpoint for final test evaluation.
    best_ckpt = checkpoint_dir / "best_model.pt"
    load_model_weights(model, best_ckpt, device=device, strict=True)
    model.eval()

    stage_metrics = evaluate_stages(model, test_loader, criterion, device)
    save_dataframe(pd.DataFrame([stage_metrics]), output_dir / "stage_metrics.csv")

    predictions_df, overall_df, per_class_df = predict_final(model, test_loader, device)
    save_dataframe(predictions_df, output_dir / "test_predictions.csv")
    save_dataframe(overall_df, output_dir / "overall_metrics.csv")
    save_dataframe(per_class_df, output_dir / "per_class_metrics.csv")

    print("Final test overall metrics:")
    print(overall_df.to_string(index=False))
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
