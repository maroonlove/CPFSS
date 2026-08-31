from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .ontology import Ontology

Item = Tuple[str, List[int], float, float, float]


def build_go_vocab_from_train(train_prop_txt: str | Path, namespace_filter: str, min_count: int = 0):
    pid2gos: Dict[str, set[str]] = defaultdict(set)
    go_freq: Dict[str, int] = defaultdict(int)

    with open(train_prop_txt, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            pid, go_id, namespace = parts[0], parts[1], parts[2]
            if namespace_filter and namespace != namespace_filter:
                continue
            pid2gos[pid].add(go_id)

    for gos in pid2gos.values():
        for go_id in gos:
            go_freq[go_id] += 1

    idx2go = sorted(go_id for go_id, count in go_freq.items() if count >= min_count)
    go2idx = {go_id: idx for idx, go_id in enumerate(idx2go)}
    print(f"[Vocab] train proteins={len(pid2gos)} labels={len(idx2go)} min_count={min_count}")
    return go2idx, idx2go


def load_split_labels_with_oov(prop_txt: str | Path, go2idx: Dict[str, int], namespace_filter: str):
    pid2in: Dict[str, set[int]] = defaultdict(set)
    pid2oov: Dict[str, set[str]] = defaultdict(set)
    all_pids = set()

    with open(prop_txt, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            pid, go_id, namespace = parts[0], parts[1], parts[2]
            if namespace_filter and namespace != namespace_filter:
                continue
            all_pids.add(pid)
            if go_id in go2idx:
                pid2in[pid].add(go2idx[go_id])
            else:
                pid2oov[pid].add(go_id)

    pid2in_final = {pid: sorted(pid2in.get(pid, [])) for pid in all_pids}
    pid2oov_final = {pid: sorted(pid2oov.get(pid, [])) for pid in all_pids}
    print(
        f"[Labels] {Path(prop_txt).name}: pids={len(all_pids)} "
        f"inpool_nonempty={sum(len(v) > 0 for v in pid2in_final.values())} "
        f"oov_nonempty={sum(len(v) > 0 for v in pid2oov_final.values())}"
    )
    return pid2in_final, pid2oov_final


def load_merged_embedding_dict(merged_pt_path: str | Path):
    print(f"[Load merged embedding] {merged_pt_path}")
    data = torch.load(merged_pt_path, map_location="cpu", weights_only=False)
    if "embeddings" not in data:
        raise KeyError(f"'embeddings' not found in {merged_pt_path}")
    emb_dict = data["embeddings"]
    print(f"[Merged embedding] proteins={len(emb_dict)}")
    if "meta" in data:
        print(f"[Merged embedding meta] {data['meta']}")
    return emb_dict


def build_background_annots_from_items(items: Sequence[Item], idx2go: Sequence[str], go: Ontology):
    all_annots = []
    for _pid, go_idxs, *_rest in items:
        term_set = set()
        for j in go_idxs:
            go_id = idx2go[j]
            if go.has_term(go_id):
                term_set |= go.get_anchestors(go_id)
        all_annots.append(list(term_set))
    return all_annots


def precompute_oov_stats(pid2oov_terms: Dict[str, Sequence[str]], go: Ontology):
    pid2stat = {}
    for pid, terms in pid2oov_terms.items():
        if not terms:
            pid2stat[pid] = (0.0, 0.0, 0.0)
            continue
        ic_sum = sum(float(go.get_ic(term)) for term in terms)
        dp_sum = sum(float(go.get_icdepth(term)) for term in terms)
        pid2stat[pid] = (float(len(terms)), ic_sum, dp_sum)
    return pid2stat


def build_items_from_pid_labels(pid2in: Dict[str, List[int]], pid2oov_stat, emb_dict):
    items: List[Item] = []
    missing_embed = 0
    for pid, go_idxs in pid2in.items():
        if pid not in emb_dict:
            missing_embed += 1
            continue
        oov_cnt, oov_ic, oov_dp = pid2oov_stat.get(pid, (0.0, 0.0, 0.0))
        items.append((pid, go_idxs, oov_cnt, oov_ic, oov_dp))
    print(f"[Items] built={len(items)} missing_embed={missing_embed}")
    return items


def compute_pos_weight_from_items(items: Sequence[Item], num_labels: int):
    n_samples = len(items)
    pos = np.zeros(num_labels, dtype=np.float64)
    for _pid, go_idxs, *_rest in items:
        if go_idxs:
            pos[go_idxs] += 1.0
    pos = np.clip(pos, 1.0, None)
    neg = n_samples - pos
    return torch.tensor(neg / pos, dtype=torch.float32)


class ProteinGoDataset(Dataset):
    def __init__(self, items: Sequence[Item], num_labels: int, emb_dict) -> None:
        self.items = list(items)
        self.num_labels = num_labels
        self.emb_dict = emb_dict
        print(f"Initialized ProteinGoDataset with {len(self.items)} samples, num_labels={num_labels}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pid, go_idxs, oov_cnt, oov_ic, oov_dp = self.items[idx]
        try:
            emb = self.emb_dict[pid]
            y = torch.zeros(self.num_labels, dtype=torch.float32)
            if go_idxs:
                y[torch.tensor(go_idxs, dtype=torch.long)] = 1.0
            return (
                {
                    "token": emb["seq_token"].detach().cpu().float(),
                    "seq": emb["sequence"].detach().cpu().float(),
                    "struc": emb["struc"].detach().cpu().float(),
                },
                y,
                pid,
                float(oov_cnt),
                float(oov_ic),
                float(oov_dp),
            )
        except Exception as exc:
            print(f"[WARN] failed to load embedding for pid={pid}: {exc}")
            return None


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    inputs, labels, pids, oov_cnt, oov_ic, oov_dp = zip(*batch)
    return (
        inputs,
        torch.stack(labels),
        list(pids),
        torch.tensor(oov_cnt, dtype=torch.float32),
        torch.tensor(oov_ic, dtype=torch.float32),
        torch.tensor(oov_dp, dtype=torch.float32),
    )
