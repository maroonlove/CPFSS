from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

DEFAULT_BEST_ALPHA = {"bp": 0.57, "cc": 0.91, "mf": 0.64}


def resolve_best_alpha(namespace: str, best_alpha: float, default_alpha: float | None = None) -> float:
    """Use namespace default when the validation-selected alpha is exactly 0.00."""
    if default_alpha is None:
        default_alpha = DEFAULT_BEST_ALPHA[namespace]
    return float(default_alpha if abs(float(best_alpha)) < 1e-12 else best_alpha)


def read_prop_labels(prop_txt: str | Path, namespace_filter: str, go2idx: Dict[str, int] | None = None):
    if go2idx is None:
        pid2gos = defaultdict(set)
        with open(prop_txt, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                pid, go_id, namespace = parts[0], parts[1], parts[2]
                if namespace_filter and namespace != namespace_filter:
                    continue
                pid2gos[pid].add(go_id)
        return pid2gos

    pid2in = defaultdict(set)
    pid2oov = defaultdict(set)
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
    return ({pid: sorted(pid2in[pid]) for pid in all_pids}, {pid: sorted(pid2oov[pid]) for pid in all_pids})


def load_diamond_scores(diamond_res_path: str | Path):
    """Read a DIAMOND result file with at least qid, sid, score columns."""
    q2hits = defaultdict(list)
    with open(diamond_res_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qid, sid = parts[0], parts[1]
            try:
                score = float(parts[2])
            except ValueError:
                continue
            if score > 0:
                q2hits[qid].append((sid, score))
    return q2hits


def build_diamond_prob_matrix(
    pids: Sequence[str],
    idx2go: Sequence[str],
    train_prop: str | Path,
    diamond_res_path: str | Path,
    namespace_filter: str,
) -> np.ndarray:
    """Convert DIAMOND hits to GO probability vectors using training annotations."""
    go2idx = {go_id: i for i, go_id in enumerate(idx2go)}
    train_pid2gos = read_prop_labels(train_prop, namespace_filter=namespace_filter, go2idx=None)
    q2hits = load_diamond_scores(diamond_res_path)

    diamond_probs = np.zeros((len(pids), len(idx2go)), dtype=np.float32)
    n_query_with_hit = 0
    n_query_with_annot_hit = 0

    for i, qid in enumerate(pids):
        hits = q2hits.get(qid, [])
        if hits:
            n_query_with_hit += 1

        total_score = 0.0
        for sid, score in hits:
            gos = train_pid2gos.get(sid)
            if not gos:
                continue
            has_any_in_pool = False
            for go_id in gos:
                j = go2idx.get(go_id)
                if j is None:
                    continue
                diamond_probs[i, j] += score
                has_any_in_pool = True
            if has_any_in_pool:
                total_score += score

        if total_score > 0:
            diamond_probs[i, :] /= total_score
            n_query_with_annot_hit += 1

    print(f"[DIAMOND] {diamond_res_path}")
    print(f"[DIAMOND] queries={len(pids)} any_hit={n_query_with_hit} annotated_hit={n_query_with_annot_hit}")
    return diamond_probs


def fuse_with_diamond(model_probs: np.ndarray, diamond_probs: np.ndarray, alpha: float) -> np.ndarray:
    return (float(alpha) * model_probs + (1.0 - float(alpha)) * diamond_probs).astype(np.float32)


def metric_selection_key(metrics: Dict[str, float]):
    """Validation alpha selection: maximize Fmax, then minimize Smin, then maximize Aupr."""
    return (metrics["Fmax"], -metrics["Smin"], metrics["Aupr"])


def write_alpha_selection(path: str | Path, row: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
