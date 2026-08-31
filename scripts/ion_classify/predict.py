#!/usr/bin/env python
"""Run unlabeled prediction with a trained hierarchical classifier."""

from __future__ import annotations

import csv
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import click as ck
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ion_classifier.data import FeatureRoots  # noqa: E402
from ion_classifier.engine import load_model_weights  # noqa: E402
from ion_classifier.model import ProteinTransformer  # noqa: E402
from ion_classifier.utils import ensure_dir, get_device, save_dataframe, set_seed  # noqa: E402


@dataclass(frozen=True)
class PredictItem:
    protein_id: str
    feature_class: Optional[int]
    seq_path: str
    seq_att_path: str
    struc_att_path: str


def _safe_torch_load(path: str | Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class PredictFeatureDataset(Dataset):
    def __init__(self, items: Sequence[PredictItem]) -> None:
        self.items = list(items)
        print(f"Initialized PredictFeatureDataset with {len(self.items)} samples")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        try:
            seq_token_embedding = _safe_torch_load(item.seq_path)
            seq_token_embedding = seq_token_embedding[0, 0, :].contiguous()

            sequence_embedding = _safe_torch_load(item.seq_att_path)
            sequence_embedding = sequence_embedding.squeeze(0).contiguous()

            struc_embedding = _safe_torch_load(item.struc_att_path)
            struc_embedding = struc_embedding.squeeze(0).contiguous()

            features = {
                "token": seq_token_embedding,
                "seq": sequence_embedding,
                "struc": struc_embedding,
            }
            return features, item.protein_id, item.feature_class
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load sample pid={item.protein_id}: {exc}")
            return None, None, None


def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None
    features, protein_ids, feature_classes = zip(*batch)
    return features, protein_ids, feature_classes


def _candidate_paths(root: Path, protein_id: str, num_subdirs: int) -> Iterable[Path]:
    yield root / f"{protein_id}.pt"
    for subdir in range(num_subdirs):
        yield root / str(subdir) / f"{protein_id}.pt"


def _find_nonempty_file(root: Path, protein_id: str, num_subdirs: int) -> Optional[Path]:
    for path in _candidate_paths(root, protein_id, num_subdirs):
        if path.exists() and path.is_file() and os.path.getsize(path) > 0:
            return path
    return None


def _complete_direct_paths(row: dict[str, str]) -> Optional[tuple[str, str, str]]:
    path_columns = ("seq_path", "seq_att_path", "struc_att_path")
    if not all(row.get(column) for column in path_columns):
        return None
    paths = tuple(row[column] for column in path_columns)
    if all(Path(path).exists() and os.path.getsize(path) > 0 for path in paths):
        return paths
    return None


def _find_feature_triplet(
    protein_id: str,
    class_hint: Optional[int],
    class0_roots: Optional[FeatureRoots],
    class1_roots: Optional[FeatureRoots],
) -> Optional[tuple[int, Path, Path, Path]]:
    candidates: list[tuple[int, FeatureRoots]] = []
    if class_hint == 0 and class0_roots is not None:
        candidates.append((0, class0_roots))
    elif class_hint == 1 and class1_roots is not None:
        candidates.append((1, class1_roots))
    else:
        if class0_roots is not None:
            candidates.append((0, class0_roots))
        if class1_roots is not None:
            candidates.append((1, class1_roots))

    for feature_class, roots in candidates:
        seq_file = _find_nonempty_file(roots.seq_root, protein_id, roots.num_subdirs)
        seq_att_file = _find_nonempty_file(roots.seq_att_root, protein_id, roots.num_subdirs)
        struc_att_file = _find_nonempty_file(roots.struc_att_root, protein_id, roots.num_subdirs)
        if seq_file and seq_att_file and struc_att_file:
            return feature_class, seq_file, seq_att_file, struc_att_file
    return None


def build_predict_items_from_csv(
    csv_path: str | Path,
    class0_roots: Optional[FeatureRoots],
    class1_roots: Optional[FeatureRoots],
    strict: bool,
) -> list[PredictItem]:
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    items: list[PredictItem] = []
    missing = 0

    for row in rows:
        protein_id = row["protein_id"]
        class_hint = int(row["class"]) if row.get("class") not in (None, "") else None

        direct_paths = _complete_direct_paths(row)
        if direct_paths is not None:
            seq_path, seq_att_path, struc_att_path = direct_paths
            items.append(PredictItem(protein_id, class_hint, seq_path, seq_att_path, struc_att_path))
            continue

        triplet = _find_feature_triplet(protein_id, class_hint, class0_roots, class1_roots)
        if triplet is not None:
            feature_class, seq_path, seq_att_path, struc_att_path = triplet
            items.append(
                PredictItem(
                    protein_id=protein_id,
                    feature_class=feature_class,
                    seq_path=str(seq_path),
                    seq_att_path=str(seq_att_path),
                    struc_att_path=str(struc_att_path),
                )
            )
            continue

        missing += 1
        if strict:
            raise FileNotFoundError(f"Missing complete feature triplet for protein_id={protein_id}")

    print(f"Loaded {len(items)} prediction samples from {csv_path}; skipped {missing} rows with missing features.")
    return items


def load_predict_items_from_pickle(path: str | Path) -> list[PredictItem]:
    with Path(path).open("rb") as handle:
        loaded = pickle.load(handle)
    return [
        item
        if isinstance(item, PredictItem)
        else PredictItem(item[0], item[1], item[2], item[3], item[4])
        for item in loaded
    ]


def save_predict_items_to_pickle(items: Sequence[PredictItem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(list(items), handle)


def build_predict_dataloader(
    items: Sequence[PredictItem],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(PredictFeatureDataset(items), **kwargs)


def combine_stage_predictions_blind(stage1_logits: torch.Tensor, stage2_logits: torch.Tensor) -> torch.Tensor:
    preds_stage1 = torch.argmax(stage1_logits, dim=1)
    preds_stage2 = torch.argmax(stage2_logits, dim=1)
    final_preds = preds_stage1.clone()

    for idx, fine_pred_tensor in enumerate(preds_stage2):
        coarse_pred = int(preds_stage1[idx].item())
        fine_pred = int(fine_pred_tensor.item())
        if coarse_pred == 0:
            final_preds[idx] = 0
        elif coarse_pred == 1:
            final_preds[idx] = 6 if fine_pred == 1 else 7 if fine_pred == 2 else 1
        elif coarse_pred == 2:
            final_preds[idx] = 2
        elif coarse_pred == 3:
            final_preds[idx] = 8 if fine_pred == 1 else 3
        elif coarse_pred == 4:
            final_preds[idx] = 5 if fine_pred == 1 else 4
    return final_preds


@torch.no_grad()
def predict_unlabeled(model: torch.nn.Module, dataloader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []

    for batch in tqdm(dataloader, desc="[Predict]"):
        if batch is None:
            continue
        feat_batch, protein_ids, feature_classes = batch
        stage1_logits, stage2_logits = model(feat_batch)
        final_preds = combine_stage_predictions_blind(stage1_logits, stage2_logits)
        stage1_preds = torch.argmax(stage1_logits, dim=1)
        stage2_preds = torch.argmax(stage2_logits, dim=1)

        for protein_id, feature_class, predicted_label, stage1_pred, stage2_pred in zip(
            protein_ids,
            feature_classes,
            final_preds.detach().cpu().tolist(),
            stage1_preds.detach().cpu().tolist(),
            stage2_preds.detach().cpu().tolist(),
        ):
            rows.append(
                {
                    "protein_id": protein_id,
                    "predicted_label": predicted_label,
                    "stage1_pred": stage1_pred,
                    "stage2_pred": stage2_pred,
                    "feature_class": feature_class,
                }
            )

    if not rows:
        raise ValueError("No predictions were generated.")
    return pd.DataFrame(rows)


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
    class0_roots = None
    if class0_seq_root and class0_seq_att_root and class0_struc_att_root:
        class0_roots = FeatureRoots(
            Path(class0_seq_root),
            Path(class0_seq_att_root),
            Path(class0_struc_att_root),
            class0_subdirs,
        )

    class1_roots = None
    if class1_seq_root and class1_seq_att_root and class1_struc_att_root:
        class1_roots = FeatureRoots(
            Path(class1_seq_root),
            Path(class1_seq_att_root),
            Path(class1_struc_att_root),
            class1_subdirs,
        )

    return class0_roots, class1_roots


@ck.command()
@ck.option("--input-csv", type=ck.Path(exists=True, dir_okay=False), default=None, help="Unlabeled CSV. Requires protein_id; optional class or direct seq_path,seq_att_path,struc_att_path.")
@ck.option("--input-items-pkl", type=ck.Path(exists=True, dir_okay=False), default=None, help="Optional cached prediction items pickle.")
@ck.option("--cache-items", type=ck.Path(dir_okay=False), default=None, help="Optional path to save built prediction items pickle.")
@ck.option("--checkpoint", "-ckpt", type=ck.Path(exists=True, dir_okay=False), required=True, help="Trained checkpoint path.")
@ck.option("--class0-seq-root", type=ck.Path(file_okay=False), default=None, help="Class 0 ESM2 token embedding root.")
@ck.option("--class0-seq-att-root", type=ck.Path(file_okay=False), default=None, help="Class 0 ESM3 function embedding root.")
@ck.option("--class0-struc-att-root", type=ck.Path(file_okay=False), default=None, help="Class 0 SaProt structure embedding root.")
@ck.option("--class1-seq-root", type=ck.Path(file_okay=False), default=None, help="Class 1 ESM2 token embedding root.")
@ck.option("--class1-seq-att-root", type=ck.Path(file_okay=False), default=None, help="Class 1 ESM3 function embedding root.")
@ck.option("--class1-struc-att-root", type=ck.Path(file_okay=False), default=None, help="Class 1 SaProt structure embedding root.")
@ck.option("--class0-subdirs", type=int, default=10, show_default=True)
@ck.option("--class1-subdirs", type=int, default=41, show_default=True)
@ck.option("--output-dir", "-o", type=ck.Path(file_okay=False), default="outputs/predict_run", show_default=True)
@ck.option("--output-file", type=str, default="predictions.csv", show_default=True)
@ck.option("--batch-size", type=int, default=64, show_default=True)
@ck.option("--num-workers", type=int, default=4, show_default=True)
@ck.option("--prefetch-factor", type=int, default=4, show_default=True)
@ck.option("--device", type=str, default="cuda:0", show_default=True)
@ck.option("--seed", type=int, default=0, show_default=True)
@ck.option("--input-dim", type=int, default=2560, show_default=True)
@ck.option("--model-dim", type=int, default=512, show_default=True)
@ck.option("--num-heads", type=int, default=8, show_default=True)
@ck.option("--num-layers", type=int, default=4, show_default=True)
@ck.option("--dropout", type=float, default=0.1, show_default=True)
@ck.option("--strict-checkpoint/--non-strict-checkpoint", default=True, show_default=True)
@ck.option("--strict-missing/--skip-missing", default=False, show_default=True)
def main(
    input_csv,
    input_items_pkl,
    cache_items,
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
    output_file,
    batch_size,
    num_workers,
    prefetch_factor,
    device,
    seed,
    input_dim,
    model_dim,
    num_heads,
    num_layers,
    dropout,
    strict_checkpoint,
    strict_missing,
):
    """Predict final labels 0-8 without requiring ground-truth labels."""

    set_seed(seed)
    device = get_device(device)
    output_dir = ensure_dir(output_dir)

    if input_items_pkl:
        items = load_predict_items_from_pickle(input_items_pkl)
    else:
        if not input_csv:
            raise ck.UsageError("Either --input-csv or --input-items-pkl must be provided.")
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
        items = build_predict_items_from_csv(input_csv, class0_roots, class1_roots, strict_missing)
        if cache_items:
            save_predict_items_to_pickle(items, cache_items)

    dataloader = build_predict_dataloader(
        items,
        batch_size=batch_size,
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

    predictions_df = predict_unlabeled(model, dataloader, device)
    output_path = output_dir / output_file
    save_dataframe(predictions_df, output_path)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
