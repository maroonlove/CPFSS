#!/usr/bin/env python
"""Generate SaProt structure embeddings from PDB files."""

from __future__ import annotations

import sys
from pathlib import Path

import click as ck
import torch
from transformers import EsmTokenizer


@ck.command()
@ck.option("--input-dir", "-i", type=ck.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Directory containing input .pdb files.")
@ck.option("--output-dir", "-o", type=ck.Path(file_okay=False, path_type=Path), required=True, help="Directory where SaProt .pt embeddings will be written.")
@ck.option("--model-path", type=ck.Path(exists=True, file_okay=False, path_type=Path), required=True, help="SaProt model/config directory, for example SaProt_650M_AF2.")
@ck.option("--foldseek-path", type=ck.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Path to the foldseek executable.")
@ck.option("--saprot-code-dir", type=ck.Path(exists=True, file_okay=False, path_type=Path), default=None, help="Optional directory containing SaProt base.py and foldseek_util.py.")
@ck.option("--chain", multiple=True, default=("A",), show_default=True, help="PDB chain ID to use. Can be supplied multiple times.")
@ck.option("--device", type=str, default="cuda:0", show_default=True, help="Torch device, for example cpu, cuda, cuda:1.")
@ck.option("--overwrite/--skip-existing", default=False, show_default=True, help="Overwrite existing output .pt files.")
def main(
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    foldseek_path: Path,
    saprot_code_dir: Path | None,
    chain: tuple[str, ...],
    device: str,
    overwrite: bool,
) -> None:
    """Read PDB files and save one SaProt structure embedding per protein."""

    if saprot_code_dir is not None:
        sys.path.insert(0, str(saprot_code_dir))

    from base import SaprotBaseModel  # noqa: PLC0415
    from foldseek_util import get_struc_seq  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    chain_list = list(chain)

    config = {
        "task": "base",
        "config_path": str(model_path),
        "load_pretrained": True,
    }
    model = SaprotBaseModel(**config)
    tokenizer = EsmTokenizer.from_pretrained(config["config_path"])
    model.to(device_obj)

    processed = 0
    skipped = 0
    for pdb_path in sorted(input_dir.glob("*.pdb")):
        protein_id = pdb_path.stem
        output_path = output_dir / f"{protein_id}.pt"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            parsed_seqs = get_struc_seq(str(foldseek_path), str(pdb_path), chain_list, plddt_mask=False)[chain_list[0]]
            _, _, combined_seq = parsed_seqs
            inputs = tokenizer(combined_seq, return_tensors="pt")
            inputs = {key: value.to(device_obj) for key, value in inputs.items()}
            embeddings = model.get_hidden_states(inputs, reduction=None)
            embeddings_cpu = embeddings[0].unsqueeze(0).detach().cpu()
            torch.save(embeddings_cpu, output_path)
            processed += 1
            print(output_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {pdb_path}: {exc}")
        finally:
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()

    print(f"SaProt embeddings complete. processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
