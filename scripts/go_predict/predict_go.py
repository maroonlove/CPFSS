#!/usr/bin/env python
"""Predict GO terms for unlabeled proteins."""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import click as ck
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from go_prediction.alpha import DEFAULT_BEST_ALPHA, build_diamond_prob_matrix, fuse_with_diamond  # noqa: E402
from go_prediction.data import build_go_vocab_from_train, load_merged_embedding_dict  # noqa: E402
from go_prediction.engine import load_checkpoint_to_model, save_go_score_csv  # noqa: E402
from go_prediction.metrics import build_child_to_ancestor_indices, propagate_scores_with_go  # noqa: E402
from go_prediction.model import ProteinGoTransformer  # noqa: E402
from go_prediction.ontology import Ontology, ROOT_GO_TERMS  # noqa: E402
from go_prediction.utils import get_device, set_seed  # noqa: E402


class UnlabeledGoDataset(Dataset):
    def __init__(self, protein_ids: Sequence[str], emb_dict) -> None:
        self.protein_ids = list(protein_ids)
        self.emb_dict = emb_dict
        print(f"Initialized UnlabeledGoDataset with {len(self.protein_ids)} proteins")

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        try:
            emb = self.emb_dict[pid]
            return (
                {
                    "token": emb["seq_token"].detach().cpu().float(),
                    "seq": emb["sequence"].detach().cpu().float(),
                    "struc": emb["struc"].detach().cpu().float(),
                },
                pid,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] failed to load embedding for pid={pid}: {exc}")
            return None


def collate_unlabeled(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    features, protein_ids = zip(*batch)
    return features, list(protein_ids)


def load_protein_ids(path: str | Path, emb_dict, use_all_embeddings: bool) -> list[str]:
    if use_all_embeddings:
        return sorted(emb_dict.keys())

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return []
        if "protein_id" in rows[0]:
            return [row["protein_id"] for row in rows if row.get("protein_id")]
        first_column = next(iter(rows[0]))
        return [row[first_column] for row in rows if row.get(first_column)]

    protein_ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip().split()
            if value:
                protein_ids.append(value[0])
    return protein_ids


@torch.no_grad()
def collect_model_probs(model, loader, device):
    model.eval()
    all_pids = []
    all_logits = []
    for batch in tqdm(loader, desc="[Predict]"):
        if batch is None:
            continue
        feat_batch, pids = batch
        logits = model(feat_batch)
        all_pids.extend(pids)
        all_logits.append(logits.detach().cpu())

    if not all_logits:
        raise ValueError("No predictions were generated.")
    raw_probs = torch.sigmoid(torch.cat(all_logits, dim=0)).numpy().astype(np.float32)
    return all_pids, raw_probs


def average_checkpoint_predictions(checkpoint_paths, model, loader, device, child_to_ancestors):
    caches = []
    base_pids = None
    for ckpt_path in checkpoint_paths:
        load_checkpoint_to_model(ckpt_path, model, device)
        pids, raw_probs = collect_model_probs(model, loader, device)
        if base_pids is None:
            base_pids = pids
        elif pids != base_pids:
            raise ValueError(f"PID order mismatch while predicting checkpoint {ckpt_path}")
        probs = propagate_scores_with_go(raw_probs, child_to_ancestors)
        caches.append({"raw_probs": raw_probs, "probs": probs})

    return {
        "pids": base_pids,
        "raw_probs": np.mean([cache["raw_probs"] for cache in caches], axis=0).astype(np.float32),
        "probs": np.mean([cache["probs"] for cache in caches], axis=0).astype(np.float32),
    }


def checkpoint_paths_from_args(checkpoint, checkpoint_dir, epochs):
    if checkpoint:
        return [Path(checkpoint)]
    if not checkpoint_dir:
        raise ck.UsageError("Either --checkpoint or --checkpoint-dir must be provided.")
    used_epochs = list(range(max(1, epochs - 2), epochs + 1))
    paths = [Path(checkpoint_dir) / f"epoch_{epoch:03d}.pth" for epoch in used_epochs]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))
    return paths


def predictions_to_dataframe(pids: Sequence[str], probs: np.ndarray, idx2go: Sequence[str], threshold: float, top_k: int):
    rows = []
    for i, pid in enumerate(pids):
        scores = probs[i]
        passed = np.where(scores >= threshold)[0]
        if top_k > 0:
            top_indices = np.argsort(scores)[::-1][:top_k]
            selected = sorted(set(passed.tolist()) | set(top_indices.tolist()), key=lambda j: float(scores[j]), reverse=True)
        else:
            selected = sorted(passed.tolist(), key=lambda j: float(scores[j]), reverse=True)
        rows.append(
            {
                "protein_id": pid,
                "pred_go": ";".join(idx2go[j] for j in selected),
                "pred_score": ";".join(f"{float(scores[j]):.6g}" for j in selected),
                "num_pred": len(selected),
            }
        )
    return pd.DataFrame(rows)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


@ck.command()
@ck.option("--input-pids", type=ck.Path(exists=True), default=None, help="TXT/CSV with protein IDs. CSV may contain protein_id.")
@ck.option("--predict-all-embeddings", is_flag=True, help="Predict all proteins found in --merged-emb-pt.")
@ck.option("--train-prop", type=ck.Path(exists=True), required=True, help="Training propagated GO labels, used only to build GO vocabulary.")
@ck.option("--merged-emb-pt", type=ck.Path(exists=True), required=True, help="Merged embedding .pt containing data['embeddings'][pid].")
@ck.option("--obo-file", type=ck.Path(exists=True), required=True, help="GO ontology .obo file for ancestor score propagation.")
@ck.option("--checkpoint", type=ck.Path(exists=True), default=None, help="Single checkpoint path.")
@ck.option("--checkpoint-dir", type=ck.Path(exists=True), default=None, help="Directory containing epoch_XXX.pth checkpoints for last3 averaging.")
@ck.option("--epochs", type=int, default=100, show_default=True, help="With --checkpoint-dir, use [E-2,E-1,E].")
@ck.option("--diamond-res", type=ck.Path(exists=True), default=None, help="Optional DIAMOND .res for the prediction proteins.")
@ck.option("--alpha", type=float, default=None, help="Fusion alpha. Defaults to namespace default when --diamond-res is provided.")
@ck.option("--namespace", type=ck.Choice(["bp", "cc", "mf"]), default="bp", show_default=True)
@ck.option("--output-dir", type=ck.Path(), required=True)
@ck.option("--threshold", type=float, default=0.5, show_default=True, help="Threshold for pred_go list.")
@ck.option("--top-k", type=int, default=0, show_default=True, help="Also include top K terms per protein in pred_go list.")
@ck.option("--batch-size", type=int, default=64, show_default=True)
@ck.option("--num-workers", type=int, default=0, show_default=True)
@ck.option("--input-dim", type=int, default=2560, show_default=True)
@ck.option("--model-dim", type=int, default=512, show_default=True)
@ck.option("--num-heads", type=int, default=8, show_default=True)
@ck.option("--num-layers", type=int, default=4, show_default=True)
@ck.option("--dropout", type=float, default=0.1, show_default=True)
@ck.option("--min-count", type=int, default=0, show_default=True)
@ck.option("--device", type=str, default="cuda:1", show_default=True)
@ck.option("--seed", type=int, default=0, show_default=True)
def main(
    input_pids,
    predict_all_embeddings,
    train_prop,
    merged_emb_pt,
    obo_file,
    checkpoint,
    checkpoint_dir,
    epochs,
    diamond_res,
    alpha,
    namespace,
    output_dir,
    threshold,
    top_k,
    batch_size,
    num_workers,
    input_dim,
    model_dim,
    num_heads,
    num_layers,
    dropout,
    min_count,
    device,
    seed,
):
    set_seed(seed)
    device = get_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_dict = load_merged_embedding_dict(merged_emb_pt)
    protein_ids = load_protein_ids(input_pids, emb_dict, predict_all_embeddings)
    if not protein_ids:
        raise ValueError("No protein IDs were provided.")

    go2idx, idx2go = build_go_vocab_from_train(train_prop, namespace_filter=namespace, min_count=min_count)
    num_labels = len(idx2go)
    go = Ontology(obo_file, with_rels=True)
    child_to_ancestors = build_child_to_ancestor_indices(idx2go, go, root_terms=ROOT_GO_TERMS)

    loader = DataLoader(
        UnlabeledGoDataset(protein_ids, emb_dict),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_unlabeled,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = ProteinGoTransformer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_labels=num_labels,
        dropout=dropout,
    ).to(device)

    ckpt_paths = checkpoint_paths_from_args(checkpoint, checkpoint_dir, epochs)
    cache = average_checkpoint_predictions(ckpt_paths, model, loader, device, child_to_ancestors)
    probs = cache["probs"]

    source = "model"
    alpha_used = None
    diamond_probs = None
    if diamond_res:
        alpha_used = float(DEFAULT_BEST_ALPHA[namespace] if alpha is None else alpha)
        diamond_probs = build_diamond_prob_matrix(
            pids=cache["pids"],
            idx2go=idx2go,
            train_prop=train_prop,
            diamond_res_path=diamond_res,
            namespace_filter=namespace,
        )
        probs = fuse_with_diamond(probs, diamond_probs, alpha_used)
        probs = propagate_scores_with_go(probs, child_to_ancestors)
        source = "model_plus_diamond"

    result_cache = {
        "source": source,
        "namespace": namespace,
        "checkpoint_paths": [str(path) for path in ckpt_paths],
        "alpha_used": alpha_used,
        "pids": cache["pids"],
        "idx2go": list(idx2go),
        "model_raw_probs": cache["raw_probs"],
        "model_probs": cache["probs"],
        "diamond_probs": diamond_probs,
        "probs": probs.astype(np.float32),
        "threshold": float(threshold),
        "top_k": int(top_k),
    }
    torch.save(result_cache, output_dir / "go_prediction_probs.pt")
    save_go_score_csv(result_cache, result_cache["probs"], idx2go, output_dir / "go_scores.csv")
    predictions_df = predictions_to_dataframe(result_cache["pids"], result_cache["probs"], idx2go, threshold, top_k)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    config = {
        "task": "predict_unlabeled_go_terms",
        "namespace": namespace,
        "input_pids": input_pids,
        "predict_all_embeddings": predict_all_embeddings,
        "train_prop": train_prop,
        "merged_emb_pt": merged_emb_pt,
        "obo_file": obo_file,
        "checkpoint": checkpoint,
        "checkpoint_dir": checkpoint_dir,
        "epochs": epochs,
        "used_checkpoints": [str(path) for path in ckpt_paths],
        "diamond_res": diamond_res,
        "alpha_used": alpha_used,
        "threshold": threshold,
        "top_k": top_k,
        "output_dir": str(output_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(config, output_dir / "predict_config.json")
    print(f"[Final] saved GO predictions to: {output_dir}")


if __name__ == "__main__":
    main()
