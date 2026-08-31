#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import click as ck
import torch
from torch.utils.data import DataLoader
import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))

if src_path not in sys.path:
    sys.path.insert(0, src_path)
    
from go_prediction.alpha import DEFAULT_BEST_ALPHA
from go_prediction.data import (
    ProteinGoDataset,
    build_background_annots_from_items,
    build_go_vocab_from_train,
    build_items_from_pid_labels,
    collate_fn,
    compute_pos_weight_from_items,
    load_merged_embedding_dict,
    load_split_labels_with_oov,
    precompute_oov_stats,
)
from go_prediction.engine import (
    average_prediction_caches,
    collect_prediction_cache,
    load_checkpoint_to_model,
    run_final_diamond_fusion,
    save_fused_predictions_csv,
    save_json,
)
from go_prediction.metrics import build_child_to_ancestor_indices, prepare_ic_vectors
from go_prediction.model import ProteinGoTransformer
from go_prediction.ontology import Ontology, ROOT_GO_TERMS
from go_prediction.utils import get_device, set_seed


@ck.command()
@ck.option("--train-prop", type=ck.Path(exists=True), required=True)
@ck.option("--val-prop", type=ck.Path(exists=True), required=True)
@ck.option("--test-prop", type=ck.Path(exists=True), required=True)
@ck.option("--merged-emb-pt", type=ck.Path(exists=True), required=True)
@ck.option("--obo-file", type=ck.Path(exists=True), required=True)
@ck.option("--val-diamond-res", type=ck.Path(exists=True), required=True)
@ck.option("--test-diamond-res", type=ck.Path(exists=True), required=True)
@ck.option("--checkpoint-dir", type=ck.Path(exists=True), required=True, help="Directory containing epoch_098.pth, epoch_099.pth, epoch_100.pth or selected last3 checkpoints.")
@ck.option("--output-dir", type=ck.Path(), required=True, help="Final output directory for DIAMOND-fused test results.")
@ck.option("--namespace", type=ck.Choice(["bp", "cc", "mf"]), default="bp", show_default=True)
@ck.option("--epochs", type=int, default=100, show_default=True, help="Used to infer checkpoints [E-2,E-1,E].")
@ck.option("--batch-size", type=int, default=64, show_default=True)
@ck.option("--num-workers", type=int, default=0, show_default=True)
@ck.option("--input-dim", type=int, default=2560, show_default=True)
@ck.option("--model-dim", type=int, default=512, show_default=True)
@ck.option("--num-heads", type=int, default=8, show_default=True)
@ck.option("--num-layers", type=int, default=4, show_default=True)
@ck.option("--dropout", type=float, default=0.1, show_default=True)
@ck.option("--min-count", type=int, default=0, show_default=True)
@ck.option("--alpha-step", type=float, default=0.01, show_default=True)
@ck.option("--default-alpha", type=float, default=None, help="Optional fallback alpha. Otherwise bp=0.57, cc=0.91, mf=0.64.")
@ck.option("--fmax-steps", type=int, default=101, show_default=True)
@ck.option("--device", type=str, default="cuda:1", show_default=True)
@ck.option("--seed", type=int, default=0, show_default=True)
def main(
    train_prop,
    val_prop,
    test_prop,
    merged_emb_pt,
    obo_file,
    val_diamond_res,
    test_diamond_res,
    checkpoint_dir,
    output_dir,
    namespace,
    epochs,
    batch_size,
    num_workers,
    input_dim,
    model_dim,
    num_heads,
    num_layers,
    dropout,
    min_count,
    alpha_step,
    default_alpha,
    fmax_steps,
    device,
    seed,
):
    set_seed(seed)
    device = get_device(device)
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    used_epochs = list(range(max(1, epochs - 2), epochs + 1))
    config = {
        "task": "evaluate_downstream_go_prediction_with_diamond_fusion",
        "namespace": namespace,
        "epochs": epochs,
        "last3_epochs": used_epochs,
        "train_prop": train_prop,
        "val_prop": val_prop,
        "test_prop": test_prop,
        "merged_emb_pt": merged_emb_pt,
        "obo_file": obo_file,
        "val_diamond_res": val_diamond_res,
        "test_diamond_res": test_diamond_res,
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "default_alpha": default_alpha if default_alpha is not None else DEFAULT_BEST_ALPHA[namespace],
    }
    save_json(config, output_dir / "eval_config.json")

    emb_dict = load_merged_embedding_dict(merged_emb_pt)
    go2idx, idx2go = build_go_vocab_from_train(train_prop, namespace_filter=namespace, min_count=min_count)
    num_labels = len(idx2go)

    train_pid2in, _ = load_split_labels_with_oov(train_prop, go2idx, namespace_filter=namespace)
    val_pid2in, val_pid2oov = load_split_labels_with_oov(val_prop, go2idx, namespace_filter=namespace)
    test_pid2in, test_pid2oov = load_split_labels_with_oov(test_prop, go2idx, namespace_filter=namespace)

    zero_oov_stat = {pid: (0.0, 0.0, 0.0) for pid in train_pid2in}
    train_items = build_items_from_pid_labels(train_pid2in, zero_oov_stat, emb_dict)

    go = Ontology(obo_file, with_rels=True)
    go.calculate_ic(build_background_annots_from_items(train_items, idx2go, go))
    goic_vector, godp_vector = prepare_ic_vectors(go, idx2go)
    child_to_ancestors = build_child_to_ancestor_indices(idx2go, go, root_terms=ROOT_GO_TERMS)

    val_items = build_items_from_pid_labels(val_pid2in, precompute_oov_stats(val_pid2oov, go), emb_dict)
    test_items = build_items_from_pid_labels(test_pid2in, precompute_oov_stats(test_pid2oov, go), emb_dict)

    val_loader = DataLoader(ProteinGoDataset(val_items, num_labels, emb_dict), batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(ProteinGoDataset(test_items, num_labels, emb_dict), batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = ProteinGoTransformer(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers, num_labels=num_labels, dropout=dropout).to(device)
    pos_weight = compute_pos_weight_from_items(train_items, num_labels).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    val_caches, test_caches, actual_used_epochs = [], [], []
    for epoch in used_epochs:
        ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        load_checkpoint_to_model(ckpt_path, model, device)
        val_caches.append(collect_prediction_cache(model, val_loader, criterion, device, goic_vector, godp_vector, child_to_ancestors=child_to_ancestors, fmax_steps=fmax_steps))
        test_caches.append(collect_prediction_cache(model, test_loader, criterion, device, goic_vector, godp_vector, child_to_ancestors=child_to_ancestors, fmax_steps=fmax_steps))
        actual_used_epochs.append(epoch)

    avg_val_cache = average_prediction_caches(val_caches, goic_vector, godp_vector, fmax_steps=fmax_steps)
    avg_test_cache = average_prediction_caches(test_caches, goic_vector, godp_vector, fmax_steps=fmax_steps)

    fused_cache = run_final_diamond_fusion(
        avg_val_cache=avg_val_cache,
        avg_test_cache=avg_test_cache,
        idx2go=idx2go,
        train_prop=train_prop,
        val_diamond_res=val_diamond_res,
        test_diamond_res=test_diamond_res,
        namespace=namespace,
        child_to_ancestors=child_to_ancestors,
        goic_vector=goic_vector,
        godp_vector=godp_vector,
        output_dir=output_dir,
        used_epochs=actual_used_epochs,
        alpha_step=alpha_step,
        fmax_steps=fmax_steps,
        default_alpha=default_alpha,
    )
    save_fused_predictions_csv(fused_cache, idx2go, test_pid2oov, output_dir)
    print(f"[Final] saved DIAMOND-fused outputs only to: {output_dir}")


if __name__ == "__main__":
    main()
