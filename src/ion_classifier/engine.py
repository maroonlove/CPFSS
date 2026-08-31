"""Training, evaluation, and prediction loops."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .labels import combine_stage_predictions, make_labels
from .metrics import calculate_per_class_metrics, overall_metrics_df


def _make_scaler(enabled: bool):
    try:
        return GradScaler("cuda", enabled=enabled)
    except TypeError:
        return GradScaler(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
    epoch: int,
) -> float:
    model.train()
    scaler = _make_scaler(enabled=use_amp and device.type == "cuda")
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"[Train E{epoch}]", leave=False)
    for batch in pbar:
        if batch is None:
            continue

        feat_batch, labels, pids, cls_list = batch
        labels = labels.to(device, non_blocking=True)
        cls_list = cls_list.to(device, non_blocking=True)

        if (labels < 0).any() or (labels >= 9).any():
            raise ValueError(f"Labels outside [0, 8]: {labels.cpu().tolist()}")

        s1_labels, s2_labels, s2_mask = make_labels(cls_list, labels, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            outputs_stage1, outputs_stage2 = model(feat_batch)
            loss_stage1 = criterion(outputs_stage1, s1_labels)
            if s2_mask.any():
                loss_stage2 = criterion(outputs_stage2[s2_mask], s2_labels[s2_mask])
                loss = loss_stage1 + loss_stage2
            else:
                loss = loss_stage1

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss detected. "
                f"pids={pids}, labels={labels.cpu().tolist()}, cls_list={cls_list.cpu().tolist()}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if n_batches == 0:
        raise ValueError("No valid batch was seen during training.")
    return running_loss / n_batches


@torch.no_grad()
def evaluate_stages(model: nn.Module, dataloader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_stage1_logits: List[torch.Tensor] = []
    all_stage2_logits: List[torch.Tensor] = []
    all_s1_labels: List[torch.Tensor] = []
    all_s2_labels: List[torch.Tensor] = []
    all_s2_masks: List[torch.Tensor] = []

    for batch in tqdm(dataloader, desc="[Evaluate]"):
        if batch is None:
            continue
        feat_batch, labels, pids, cls_list = batch
        labels = labels.to(device, non_blocking=True)
        cls_list = cls_list.to(device, non_blocking=True)

        outputs_stage1, outputs_stage2 = model(feat_batch)
        s1_labels, s2_labels, s2_mask = make_labels(cls_list, labels, device)

        loss_stage1 = criterion(outputs_stage1, s1_labels)
        loss = loss_stage1
        if s2_mask.any():
            loss = loss + criterion(outputs_stage2[s2_mask], s2_labels[s2_mask])

        total_loss += float(loss.item())
        n_batches += 1
        all_stage1_logits.append(outputs_stage1.detach().cpu())
        all_stage2_logits.append(outputs_stage2.detach().cpu())
        all_s1_labels.append(s1_labels.detach().cpu())
        all_s2_labels.append(s2_labels.detach().cpu())
        all_s2_masks.append(s2_mask.detach().cpu())

    if n_batches == 0:
        raise ValueError("No valid batch was seen during evaluation.")

    stage1_logits = torch.cat(all_stage1_logits, dim=0)
    stage2_logits = torch.cat(all_stage2_logits, dim=0)
    s1_labels = torch.cat(all_s1_labels, dim=0)
    s2_labels = torch.cat(all_s2_labels, dim=0)
    s2_masks = torch.cat(all_s2_masks, dim=0)

    s1_preds = torch.argmax(stage1_logits, dim=1).numpy()
    s1_true = s1_labels.numpy()

    metrics = {
        "loss": total_loss / n_batches,
        "stage1_accuracy": accuracy_score(s1_true, s1_preds),
        "stage1_precision_macro": precision_score(s1_true, s1_preds, average="macro", zero_division=0),
        "stage1_recall_macro": recall_score(s1_true, s1_preds, average="macro", zero_division=0),
        "stage1_f1_macro": f1_score(s1_true, s1_preds, average="macro", zero_division=0),
        "stage2_samples": int(s2_masks.sum().item()),
    }

    if s2_masks.any():
        s2_preds = torch.argmax(stage2_logits[s2_masks], dim=1).numpy()
        s2_true = s2_labels[s2_masks].numpy()
        metrics.update(
            {
                "stage2_accuracy": accuracy_score(s2_true, s2_preds),
                "stage2_precision_macro": precision_score(s2_true, s2_preds, average="macro", zero_division=0),
                "stage2_recall_macro": recall_score(s2_true, s2_preds, average="macro", zero_division=0),
                "stage2_f1_macro": f1_score(s2_true, s2_preds, average="macro", zero_division=0),
            }
        )
    else:
        metrics.update(
            {
                "stage2_accuracy": 0.0,
                "stage2_precision_macro": 0.0,
                "stage2_recall_macro": 0.0,
                "stage2_f1_macro": 0.0,
            }
        )
    return metrics


@torch.no_grad()
def predict_final(model: nn.Module, dataloader, device: torch.device):
    model.eval()
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_pids: List[str] = []

    for batch in tqdm(dataloader, desc="[Predict]"):
        if batch is None:
            continue
        feat_batch, labels, pids, cls_list = batch
        cls_list = cls_list.to(device, non_blocking=True)

        outputs_stage1, outputs_stage2 = model(feat_batch)
        final_preds = combine_stage_predictions(cls_list, outputs_stage1, outputs_stage2)

        all_preds.append(final_preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_pids.extend(pids)

    if not all_preds:
        raise ValueError("No predictions were generated.")

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    predictions_df = pd.DataFrame(
        {
            "protein_id": all_pids,
            "predicted_label": y_pred,
            "true_label": y_true,
        }
    )
    overall_df = overall_metrics_df(y_true, y_pred)
    per_class_df = calculate_per_class_metrics(y_true, y_pred, num_classes=9)
    return predictions_df, overall_df, per_class_df


def save_checkpoint(model, optimizer, path: str | Path, epoch: int, train_loss: float, extra: Dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "train_loss": train_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_model_weights(model, checkpoint_path: str | Path, device: torch.device, strict: bool = True) -> Dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        return checkpoint

    model.load_state_dict(checkpoint, strict=strict)
    return {"model_state_dict": checkpoint}
