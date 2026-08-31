from __future__ import annotations

import warnings
from typing import Dict, Sequence

import numpy as np
import scipy.sparse as ssp

from .ontology import Ontology, ROOT_GO_TERMS


def prepare_ic_vectors(go: Ontology, idx2go: Sequence[str]):
    goic_vector = np.array([go.get_ic(go_id) for go_id in idx2go], dtype=np.float32).reshape(-1, 1)
    godp_vector = np.array([go.get_icdepth(go_id) for go_id in idx2go], dtype=np.float32).reshape(-1, 1)
    return goic_vector, godp_vector


def trapz_aupr(precisions, recalls):
    precisions = np.asarray(precisions, dtype=np.float64)
    recalls = np.asarray(recalls, dtype=np.float64)
    order = np.argsort(recalls)
    return float(np.trapz(precisions[order], recalls[order]))


def build_child_to_ancestor_indices(idx2go: Sequence[str], go: Ontology, root_terms=None):
    """Build index lists used for GO score propagation.

    For each child term, this returns all in-vocabulary ancestors except the BP/MF/CC
    root terms. The child itself is included because Ontology.get_anchestors() includes
    the queried term.
    """
    root_terms = set(ROOT_GO_TERMS if root_terms is None else root_terms)
    go2idx = {go_id: i for i, go_id in enumerate(idx2go)}
    child_to_ancestors = []
    for child_go in idx2go:
        ancestors = go.get_anchestors(child_go) if go.has_term(child_go) else {child_go}
        anc_indices = sorted({go2idx[a] for a in ancestors if a not in root_terms and a in go2idx})
        child_to_ancestors.append(anc_indices)
    return child_to_ancestors


def propagate_scores_with_go(scores: np.ndarray, child_to_ancestors) -> np.ndarray:
    """Guarantee ancestor scores are not lower than child scores."""
    prop_scores = scores.astype(np.float32, copy=True)
    source_scores = scores.astype(np.float32, copy=False)
    for child_idx, ancestor_indices in enumerate(child_to_ancestors):
        if not ancestor_indices:
            continue
        child_scores = source_scores[:, child_idx]
        for anc_idx in ancestor_indices:
            prop_scores[:, anc_idx] = np.maximum(prop_scores[:, anc_idx], child_scores)
    return prop_scores.astype(np.float32)


def evalpy_curve_metrics_with_oov(
    y_true_in,
    scores,
    goic_vector,
    godp_vector,
    oov_cnt,
    oov_ic_sum,
    oov_dp_sum,
    steps: int = 101,
):
    """Protein-centric GO metrics with OOV targets kept in recall denominators.

    This follows the evaluation.py/CPSS convention: Fmax is selected by maximum F1;
    ties choose the smaller S value.
    """
    targets = ssp.csr_matrix(y_true_in.astype(np.int32))
    n_samples = targets.shape[0]

    best_f = 0.0
    best_s = float("inf")
    best_thr = 0.0
    precisions, recalls = [], []
    icprecisions, icrecalls = [], []
    dpprecisions, dprecalls = [], []

    true_in_cnt = np.asarray(targets.sum(axis=1)).reshape(-1).astype(np.float32)
    true_all_cnt = true_in_cnt + oov_cnt.astype(np.float32)

    for cut in (c / (steps - 1) for c in range(steps)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct_sc = cut_sc.multiply(targets)
        fp_sc = cut_sc - correct_sc
        fn_sc = targets - correct_sc

        correct = np.asarray(correct_sc.sum(axis=1)).reshape(-1).astype(np.float32)
        pred_cnt = np.asarray(cut_sc.sum(axis=1)).reshape(-1).astype(np.float32)

        correct_ic = np.asarray(correct_sc.dot(goic_vector)).reshape(-1).astype(np.float32)
        cut_ic = np.asarray(cut_sc.dot(goic_vector)).reshape(-1).astype(np.float32)
        targets_ic = np.asarray(targets.dot(goic_vector)).reshape(-1).astype(np.float32)

        correct_dp = np.asarray(correct_sc.dot(godp_vector)).reshape(-1).astype(np.float32)
        cut_dp = np.asarray(cut_sc.dot(godp_vector)).reshape(-1).astype(np.float32)
        targets_dp = np.asarray(targets.dot(godp_vector)).reshape(-1).astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_i = correct / pred_cnt
            p = float(np.average(p_i[~np.isnan(p_i)])) if np.any(~np.isnan(p_i)) else 0.0

            r_i = np.divide(correct, true_all_cnt, out=np.zeros_like(correct), where=(true_all_cnt > 0))
            r = float(np.average(r_i))

            mi = float(fp_sc.dot(goic_vector).sum(axis=0)) / n_samples
            ru_in = float(fn_sc.dot(goic_vector).sum(axis=0)) / n_samples
            ru = ru_in + float(np.mean(oov_ic_sum))

            icp_i = correct_ic / cut_ic
            icp = float(np.average(icp_i[~np.isnan(icp_i)])) if np.any(~np.isnan(icp_i)) else 0.0
            denom_ic = targets_ic + oov_ic_sum.astype(np.float32)
            icr_i = np.divide(correct_ic, denom_ic, out=np.zeros_like(correct_ic), where=(denom_ic > 0))
            icr = float(np.average(icr_i))

            dpp_i = correct_dp / cut_dp
            dpp = float(np.average(dpp_i[~np.isnan(dpp_i)])) if np.any(~np.isnan(dpp_i)) else 0.0
            denom_dp = targets_dp + oov_dp_sum.astype(np.float32)
            dpr_i = np.divide(correct_dp, denom_dp, out=np.zeros_like(correct_dp), where=(denom_dp > 0))
            dpr = float(np.average(dpr_i))

        precisions.append(0.0 if np.isnan(p) else float(p))
        recalls.append(float(r))
        icprecisions.append(0.0 if np.isnan(icp) else float(icp))
        icrecalls.append(float(icr))
        dpprecisions.append(0.0 if np.isnan(dpp) else float(dpp))
        dprecalls.append(float(dpr))

        f_score = 0.0 if (p + r) == 0 else float(2 * p * r / (p + r))
        s_score = float(np.sqrt(ru * ru + mi * mi))
        if (f_score > best_f) or (f_score == best_f and s_score < best_s):
            best_f = f_score
            best_s = s_score
            best_thr = float(cut)

    return {
        "Fmax": float(best_f),
        "Smin": float(best_s),
        "Aupr": float(trapz_aupr(precisions, recalls)),
        "ICAUPR": float(trapz_aupr(icprecisions, icrecalls)),
        "DPAUPR": float(trapz_aupr(dpprecisions, dprecalls)),
        "thr_fmax": float(best_thr),
    }


def metrics_from_probs(cache: Dict, probs: np.ndarray | None = None, goic_vector=None, godp_vector=None, fmax_steps: int = 101):
    scores = cache["probs"] if probs is None else probs
    return evalpy_curve_metrics_with_oov(
        y_true_in=cache["targets"].astype(np.int32),
        scores=scores.astype(np.float32),
        goic_vector=goic_vector,
        godp_vector=godp_vector,
        oov_cnt=cache["oov_cnt"],
        oov_ic_sum=cache["oov_ic"],
        oov_dp_sum=cache["oov_dp"],
        steps=fmax_steps,
    )
