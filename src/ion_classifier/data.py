"""Dataset and feature-path utilities for transporter / ion-channel classification.

Expected item format:
    (protein_id, class_id, label, esm2_token_path, esm3_function_path, saprot_structure_path)

The CSV input used to build this format should contain at least three columns:
    protein_id,class,label

class_id convention:
    0: transporter protein, mapped to coarse label 0
    1: ion-channel protein, mapped hierarchically to labels 1-8
"""

from __future__ import annotations

import csv
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

Item = Tuple[str, int, int, str, str, str]


@dataclass(frozen=True)
class FeatureRoots:
    """Root directories for three feature types of one protein class."""

    seq_root: Path
    seq_att_root: Path
    struc_att_root: Path
    num_subdirs: int


def _safe_torch_load(path: str | Path) -> torch.Tensor:
    """Load tensor files saved by torch.save.

    weights_only=True is preferred for safer deserialization in modern PyTorch.
    A fallback is kept for older PyTorch versions.
    """

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ProteinFeatureDataset(Dataset):
    """Dataset that lazily loads ESM2 token, ESM3 function, and SaProt structure tensors."""

    def __init__(self, items: Sequence[Item]) -> None:
        self.items = list(items)
        print(f"Initialized ProteinFeatureDataset with {len(self.items)} samples")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        pid, cls, label, seq_path, seq_att_path, struc_att_path = self.items[idx]

        if not (os.path.exists(seq_path) and os.path.exists(seq_att_path) and os.path.exists(struc_att_path)):
            return None, None

        try:
            # ESM2 sequence token embedding: [1, L, 1280] -> [1280]
            seq_token_embedding = _safe_torch_load(seq_path)
            seq_token_embedding = seq_token_embedding[0, 0, :].contiguous()

            # ESM3 sequence+InterPro/function embedding: [1, L, 1536] -> [L, 1536]
            sequence_embedding = _safe_torch_load(seq_att_path)
            sequence_embedding = sequence_embedding.squeeze(0).contiguous()

            # SaProt structure embedding: [1, L, 1280] -> [L, 1280]
            struc_embedding = _safe_torch_load(struc_att_path)
            struc_embedding = struc_embedding.squeeze(0).contiguous()

            label = int(label)
            if label < 0 or label >= 9:
                raise ValueError(f"Label {label} is outside valid range [0, 8].")

            features = {
                "token": seq_token_embedding,
                "seq": sequence_embedding,
                "struc": struc_embedding,
            }
            return features, torch.tensor(label, dtype=torch.long), pid, int(cls)

        except Exception as exc:  # noqa: BLE001: keep sample-level failure tolerant for large feature sets
            print(f"Failed to load sample pid={pid}: {exc}")
            return None, None


def collate_fn(batch):
    """Skip failed samples and keep variable-length tensors as a list."""

    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if len(batch) == 0:
        return None

    inputs, labels, pids, cls_list = zip(*batch)
    labels = torch.stack(labels)
    cls_list = torch.tensor(cls_list, dtype=torch.long)
    return inputs, labels, pids, cls_list


def _candidate_paths(root: Path, protein_id: str, num_subdirs: int) -> Iterable[Path]:
    """Yield possible tensor paths.

    The original project stored features under root/0/*.pt, root/1/*.pt, ... .
    This helper also supports a flat layout root/*.pt for cleaner GitHub examples.
    """

    yield root / f"{protein_id}.pt"
    for subdir in range(num_subdirs):
        yield root / str(subdir) / f"{protein_id}.pt"


def _find_nonempty_file(root: Path, protein_id: str, num_subdirs: int) -> Optional[Path]:
    for path in _candidate_paths(root, protein_id, num_subdirs):
        if path.exists() and path.is_file() and os.path.getsize(path) > 0:
            return path
    return None


def build_items_from_csv(
    csv_path: str | Path,
    class0_roots: FeatureRoots,
    class1_roots: FeatureRoots,
    seed: int = 0,
    sample_every: Optional[int] = None,
    strict: bool = False,
) -> List[Item]:
    """Build tensor-path items from a metadata CSV.

    Parameters
    ----------
    csv_path:
        CSV with columns protein_id,class,label.
    class0_roots:
        Feature roots for transporter samples (class=0).
    class1_roots:
        Feature roots for ion-channel samples (class=1).
    sample_every:
        If >1, randomly keeps one row in every consecutive group of this size.
        This reproduces the original quick subsampling behavior.
    strict:
        If True, raise an error when any row has missing feature files.
    """

    csv_path = Path(csv_path)
    rng = random.Random(seed)
    with csv_path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    if sample_every is not None and sample_every > 1:
        selected_rows = [rng.choice(raw_rows[i : i + sample_every]) for i in range(0, len(raw_rows), sample_every)]
    else:
        selected_rows = raw_rows

    items: List[Item] = []
    missing = 0

    for row in selected_rows:
        pid = row["protein_id"]
        cls = int(row["class"])
        label = int(row["label"])

        if cls == 0:
            roots = class0_roots
        elif cls == 1:
            roots = class1_roots
        else:
            if strict:
                raise ValueError(f"Unsupported class={cls} for protein_id={pid}")
            continue

        seq_file = _find_nonempty_file(roots.seq_root, pid, roots.num_subdirs)
        seq_att_file = _find_nonempty_file(roots.seq_att_root, pid, roots.num_subdirs)
        struc_att_file = _find_nonempty_file(roots.struc_att_root, pid, roots.num_subdirs)

        if seq_file and seq_att_file and struc_att_file:
            items.append((pid, cls, label, str(seq_file), str(seq_att_file), str(struc_att_file)))
        else:
            missing += 1
            if strict:
                raise FileNotFoundError(
                    f"Missing feature file for protein_id={pid}: "
                    f"seq={seq_file}, seq_att={seq_att_file}, struc_att={struc_att_file}"
                )

    print(f"Loaded {len(items)} valid samples from {csv_path}; skipped {missing} rows with missing features.")
    return items


def load_items_from_pickle(path: str | Path) -> List[Item]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_items_to_pickle(items: Sequence[Item], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(list(items), handle)


def build_dataloader(
    items: Sequence[Item],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    dataset = ProteinFeatureDataset(items)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)
