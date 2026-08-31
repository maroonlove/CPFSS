"""Hierarchical label mapping utilities."""

from __future__ import annotations

import torch


STAGE1_DESCRIPTION = {
    0: "transporter",
    1: "ion_channel_group_1_6_7",
    2: "ion_channel_group_2",
    3: "ion_channel_group_3_8",
    4: "ion_channel_group_4_5",
}


def get_stage1_label(cls: int, label: int) -> int:
    """Map raw class/label to the stage-1 coarse label."""

    if cls == 0:
        return 0
    if label in [1, 6, 7]:
        return 1
    if label == 2:
        return 2
    if label in [3, 8]:
        return 3
    if label in [4, 5]:
        return 4
    raise ValueError(f"Invalid ion-channel label: {label}")


def get_stage2_label(label: int) -> int:
    """Map raw ion-channel label to the shared stage-2 fine label."""

    if label in [1, 6, 7]:
        return {1: 0, 6: 1, 7: 2}[label]
    if label == 2:
        return 0
    if label in [3, 8]:
        return {3: 0, 8: 1}[label]
    if label in [4, 5]:
        return {4: 0, 5: 1}[label]
    raise ValueError(f"Invalid label: {label}")


def make_labels(cls_list: torch.Tensor, labels: torch.Tensor, device: torch.device):
    """Create stage-1 labels, stage-2 labels, and stage-2 mask."""

    s1_labels = torch.tensor(
        [get_stage1_label(int(c), int(l)) for c, l in zip(cls_list, labels)],
        dtype=torch.long,
        device=device,
    )

    s2_mask = cls_list.to(device) == 1
    s2_labels = torch.full_like(s1_labels, -100)
    if s2_mask.any():
        s2_labels[s2_mask] = torch.tensor(
            [get_stage2_label(int(label)) for label in labels.to(device)[s2_mask]],
            dtype=torch.long,
            device=device,
        )
    return s1_labels, s2_labels, s2_mask


def combine_stage_predictions(cls_list: torch.Tensor, stage1_logits: torch.Tensor, stage2_logits: torch.Tensor) -> torch.Tensor:
    """Convert stage-1 and stage-2 outputs to final labels 0-8."""

    preds_stage1 = torch.argmax(stage1_logits, dim=1)
    preds_stage2 = torch.argmax(stage2_logits, dim=1)
    final_preds = preds_stage1.clone()

    ion_mask = cls_list.to(stage1_logits.device) == 1
    if ion_mask.any():
        ion_idx = torch.where(ion_mask)[0]
        for idx, fine_pred in zip(ion_idx, preds_stage2[ion_mask]):
            coarse_pred = int(preds_stage1[idx].item())
            fine_pred = int(fine_pred.item())
            if coarse_pred == 1:
                final_preds[idx] = 6 if fine_pred == 1 else 7 if fine_pred == 2 else 1
            elif coarse_pred == 2:
                final_preds[idx] = 2
            elif coarse_pred == 3:
                final_preds[idx] = 8 if fine_pred == 1 else 3
            elif coarse_pred == 4:
                final_preds[idx] = 5 if fine_pred == 1 else 4

    trans_mask = cls_list.to(stage1_logits.device) == 0
    final_preds[trans_mask] = 0
    return final_preds
