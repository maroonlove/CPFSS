from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .alpha import build_diamond_prob_matrix, fuse_with_diamond, metric_selection_key, resolve_best_alpha, write_alpha_selection
from .metrics import metrics_from_probs, propagate_scores_with_go


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def train_one_epoch(model, loader, criterion, optimizer, scaler: GradScaler, device, amp: bool, grad_clip: float):
    model.train()
    running_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc="[Train]", leave=False)
    use_amp = amp and device.type == "cuda"
    for batch in pbar:
        if batch is None:
            continue
        feat_batch, targets, pids, *_rest = batch
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(feat_batch)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected. pids={pids}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.item())
        n_batches += 1
        pbar.set_postfix(loss=f"{float(loss.item()):.4f}")
    return running_loss / max(1, n_batches)


@torch.no_grad()
def collect_prediction_cache(
    model,
    loader,
    criterion,
    device,
    goic_vector,
    godp_vector,
    child_to_ancestors=None,
    fmax_steps: int = 101,
):
    """Collect model probabilities for one split.

    If child_to_ancestors is provided, model probabilities are GO-propagated before
    metric computation and before downstream DIAMOND fusion.
    """
    model.eval()
    all_pids, all_logits, all_targets = [], [], []
    all_oov_cnt, all_oov_ic, all_oov_dp = [], [], []
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="[CollectPred]", leave=False):
        if batch is None:
            continue
        feat_batch, targets, pids, oov_cnt, oov_ic, oov_dp = batch
        targets = targets.to(device, non_blocking=True)
        logits = model(feat_batch)
        loss = criterion(logits, targets)
        total_loss += float(loss.item())
        n_batches += 1
        all_pids.extend(pids)
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_oov_cnt.append(oov_cnt.detach().cpu())
        all_oov_ic.append(oov_ic.detach().cpu())
        all_oov_dp.append(oov_dp.detach().cpu())
    if n_batches == 0:
        raise ValueError("No valid batch found while collecting predictions.")

    raw_probs = torch.sigmoid(torch.cat(all_logits, dim=0)).numpy().astype(np.float32)
    probs = propagate_scores_with_go(raw_probs, child_to_ancestors) if child_to_ancestors is not None else raw_probs
    cache = {
        "pids": all_pids,
        "raw_probs": raw_probs,
        "probs": probs.astype(np.float32),
        "targets": torch.cat(all_targets, dim=0).numpy().astype(np.int32),
        "oov_cnt": torch.cat(all_oov_cnt, dim=0).numpy().astype(np.float32),
        "oov_ic": torch.cat(all_oov_ic, dim=0).numpy().astype(np.float32),
        "oov_dp": torch.cat(all_oov_dp, dim=0).numpy().astype(np.float32),
        "loss": total_loss / n_batches,
    }
    cache["metrics"] = metrics_from_probs(cache, goic_vector=goic_vector, godp_vector=godp_vector, fmax_steps=fmax_steps)
    return cache


def average_prediction_caches(caches: Sequence[Dict], goic_vector, godp_vector, fmax_steps: int = 101):
    if not caches:
        raise ValueError("No prediction caches to average.")
    base_pids = caches[0]["pids"]
    for i, cache in enumerate(caches[1:], start=1):
        if cache["pids"] != base_pids:
            raise ValueError(f"PID order mismatch between cache 0 and cache {i}.")
    avg_cache = {
        "pids": base_pids,
        "raw_probs": np.mean([c["raw_probs"] for c in caches], axis=0).astype(np.float32),
        "probs": np.mean([c["probs"] for c in caches], axis=0).astype(np.float32),
        "targets": caches[0]["targets"],
        "oov_cnt": caches[0]["oov_cnt"],
        "oov_ic": caches[0]["oov_ic"],
        "oov_dp": caches[0]["oov_dp"],
        "loss": float(np.mean([c["loss"] for c in caches])),
    }
    avg_cache["metrics"] = metrics_from_probs(avg_cache, goic_vector=goic_vector, godp_vector=godp_vector, fmax_steps=fmax_steps)
    return avg_cache


def search_alpha_on_validation(
    avg_val_cache: Dict,
    val_diamond_probs: np.ndarray,
    child_to_ancestors,
    goic_vector,
    godp_vector,
    namespace: str,
    alpha_step: float = 0.01,
    fmax_steps: int = 101,
    default_alpha: float | None = None,
):
    alpha_values = np.arange(0.0, 1.0 + 1e-9, alpha_step)
    best_alpha = None
    best_metrics = None
    best_key = None

    for alpha in alpha_values:
        alpha = float(round(alpha, 6))
        fused_val = fuse_with_diamond(avg_val_cache["probs"], val_diamond_probs, alpha)
        fused_val = propagate_scores_with_go(fused_val, child_to_ancestors)
        metrics = metrics_from_probs(avg_val_cache, fused_val, goic_vector=goic_vector, godp_vector=godp_vector, fmax_steps=fmax_steps)
        key = metric_selection_key(metrics)
        if best_key is None or key > best_key:
            best_key = key
            best_alpha = alpha
            best_metrics = metrics

    alpha_used = resolve_best_alpha(namespace, best_alpha=best_alpha, default_alpha=default_alpha)
    fused_val_at_used = fuse_with_diamond(avg_val_cache["probs"], val_diamond_probs, alpha_used)
    fused_val_at_used = propagate_scores_with_go(fused_val_at_used, child_to_ancestors)
    metrics_at_alpha_used = metrics_from_probs(
        avg_val_cache,
        fused_val_at_used,
        goic_vector=goic_vector,
        godp_vector=godp_vector,
        fmax_steps=fmax_steps,
    )
    return {
        "best_alpha_from_val": float(best_alpha),
        "alpha_used": float(alpha_used),
        "used_default_alpha": bool(abs(float(best_alpha)) < 1e-12),
        "best_val_metrics": best_metrics,
        "val_metrics_at_alpha_used": metrics_at_alpha_used,
    }


def run_final_diamond_fusion(
    avg_val_cache: Dict,
    avg_test_cache: Dict,
    idx2go: Sequence[str],
    train_prop: str | Path,
    val_diamond_res: str | Path,
    test_diamond_res: str | Path,
    namespace: str,
    child_to_ancestors,
    goic_vector,
    godp_vector,
    output_dir: str | Path,
    used_epochs: Sequence[int],
    alpha_step: float = 0.01,
    fmax_steps: int = 101,
    default_alpha: float | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_diamond_probs = build_diamond_prob_matrix(
        pids=avg_val_cache["pids"],
        idx2go=idx2go,
        train_prop=train_prop,
        diamond_res_path=val_diamond_res,
        namespace_filter=namespace,
    )
    test_diamond_probs = build_diamond_prob_matrix(
        pids=avg_test_cache["pids"],
        idx2go=idx2go,
        train_prop=train_prop,
        diamond_res_path=test_diamond_res,
        namespace_filter=namespace,
    )

    alpha_info = search_alpha_on_validation(
        avg_val_cache=avg_val_cache,
        val_diamond_probs=val_diamond_probs,
        child_to_ancestors=child_to_ancestors,
        goic_vector=goic_vector,
        godp_vector=godp_vector,
        namespace=namespace,
        alpha_step=alpha_step,
        fmax_steps=fmax_steps,
        default_alpha=default_alpha,
    )
    alpha_used = alpha_info["alpha_used"]

    fused_test_probs = fuse_with_diamond(avg_test_cache["probs"], test_diamond_probs, alpha_used)
    fused_test_probs = propagate_scores_with_go(fused_test_probs, child_to_ancestors)
    final_test_metrics = metrics_from_probs(
        avg_test_cache,
        fused_test_probs,
        goic_vector=goic_vector,
        godp_vector=godp_vector,
        fmax_steps=fmax_steps,
    )

    fused_cache = {
        "source": "last3_probability_average_plus_diamond",
        "formula": "fused_probs = alpha * model_probs + (1 - alpha) * diamond_probs",
        "alpha_selected_by": "validation_Fmax_then_Smin_then_Aupr",
        "used_epochs": list(used_epochs),
        "namespace": namespace,
        "best_alpha_from_val": alpha_info["best_alpha_from_val"],
        "alpha_used": alpha_used,
        "used_default_alpha": alpha_info["used_default_alpha"],
        "pids": avg_test_cache["pids"],
        "idx2go": list(idx2go),
        "model_probs": avg_test_cache["probs"],
        "model_raw_probs": avg_test_cache["raw_probs"],
        "diamond_probs": test_diamond_probs.astype(np.float32),
        "fused_probs": fused_test_probs.astype(np.float32),
        "probs": fused_test_probs.astype(np.float32),
        "targets": avg_test_cache["targets"],
        "oov_cnt": avg_test_cache["oov_cnt"],
        "oov_ic": avg_test_cache["oov_ic"],
        "oov_dp": avg_test_cache["oov_dp"],
        "loss": avg_test_cache["loss"],
        "best_val_metrics": alpha_info["best_val_metrics"],
        "val_metrics_at_alpha_used": alpha_info["val_metrics_at_alpha_used"],
        "final_test_metrics": final_test_metrics,
        "score_propagated_after_fusion": True,
    }

    torch.save(fused_cache, output_dir / "diamond_fused_test_probs.pt")

    metrics_row = {
        "namespace": namespace,
        "used_epochs": str(list(used_epochs)),
        "best_alpha_from_val": alpha_info["best_alpha_from_val"],
        "alpha_used": alpha_used,
        "used_default_alpha": alpha_info["used_default_alpha"],
        "formula": "fused_probs = alpha * model_probs + (1 - alpha) * diamond_probs",
    }
    for key, value in alpha_info["best_val_metrics"].items():
        metrics_row[f"val_best_{key}"] = value
    for key, value in alpha_info["val_metrics_at_alpha_used"].items():
        metrics_row[f"val_alpha_used_{key}"] = value
    for key, value in final_test_metrics.items():
        metrics_row[f"test_{key}"] = value
    pd.DataFrame([metrics_row]).to_csv(output_dir / "metrics.csv", index=False)

    alpha_row = {
        "best_alpha_from_val": alpha_info["best_alpha_from_val"],
        "alpha_used": alpha_used,
        "used_default_alpha": alpha_info["used_default_alpha"],
    }
    for key, value in alpha_info["best_val_metrics"].items():
        alpha_row[f"best_val_{key}"] = value
    for key, value in alpha_info["val_metrics_at_alpha_used"].items():
        alpha_row[f"val_alpha_used_{key}"] = value
    write_alpha_selection(output_dir / "alpha_selection.csv", alpha_row)

    save_go_score_csv(avg_test_cache, fused_test_probs, idx2go, output_dir / "diamond_fused_test_go_scores.csv")
    return fused_cache


def prediction_cache_to_dataframe(cache: Dict, idx2go: Sequence[str], threshold: float, pid2oov_terms=None, probs_key: str = "probs"):
    pid2oov_terms = pid2oov_terms or {}
    probs = cache[probs_key]
    pred_bin = (probs >= threshold).astype(int)
    true_bin = cache["targets"].astype(int)
    rows = []
    for i, pid in enumerate(cache["pids"]):
        pred_terms = [idx2go[j] for j in np.where(pred_bin[i] == 1)[0]]
        true_terms_in = [idx2go[j] for j in np.where(true_bin[i] == 1)[0]]
        true_terms_oov = list(pid2oov_terms.get(pid, []))
        rows.append(
            {
                "protein_id": pid,
                "pred_go": ";".join(pred_terms),
                "true_go_inpool": ";".join(true_terms_in),
                "true_go_oov": ";".join(true_terms_oov),
                "true_go_all": ";".join(true_terms_in + true_terms_oov),
            }
        )
    return pd.DataFrame(rows)


def save_go_score_csv(cache: Dict, probs: np.ndarray, idx2go: Sequence[str], out_csv: str | Path):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, pid in enumerate(cache["pids"]):
        for j, go_id in enumerate(idx2go):
            rows.append({"protein_id": pid, "go_id": go_id, "score": float(probs[i, j])})
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def save_fused_predictions_csv(fused_cache: Dict, idx2go: Sequence[str], pid2oov_terms, output_dir: str | Path):
    output_dir = Path(output_dir)
    threshold = fused_cache["final_test_metrics"]["thr_fmax"]
    df = prediction_cache_to_dataframe(fused_cache, idx2go, threshold=threshold, pid2oov_terms=pid2oov_terms, probs_key="fused_probs")
    df.to_csv(output_dir / "predictions_test.csv", index=False)


def save_checkpoint(path, epoch, model, optimizer, scaler, train_loss, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": float(train_loss),
            "train_config": config,
        },
        path,
    )


def load_checkpoint_to_model(path, model, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return ckpt
