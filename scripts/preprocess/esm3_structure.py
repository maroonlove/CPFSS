#!/usr/bin/env python
"""Generate ESM3-predicted PDB structures from FASTA files."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import click as ck
import torch
from Bio import SeqIO
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig
from huggingface_hub import login


def _login(token: str | None) -> None:
    token = token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)


@ck.command()
@ck.option("--input-dir", "-i", type=ck.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Directory containing input .fasta files.")
@ck.option("--output-dir", "-o", type=ck.Path(file_okay=False, path_type=Path), required=True, help="Directory where .pdb files will be written.")
@ck.option("--model-name", type=str, default="esm3-sm-open-v1", show_default=True, help="ESM3 model name passed to ESM3.from_pretrained.")
@ck.option("--device", type=str, default="cuda:0", show_default=True, help="Torch device, for example cpu, cuda, cuda:1.")
@ck.option("--num-steps", type=int, default=8, show_default=True, help="Number of ESM3 structure generation steps.")
@ck.option("--hf-token", type=str, default=None, help="Hugging Face token. If omitted, HF_TOKEN environment variable is used.")
@ck.option("--hf-endpoint", type=str, default=None, help="Optional Hugging Face endpoint mirror, for example https://hf-mirror.com.")
@ck.option("--overwrite/--skip-existing", default=False, show_default=True, help="Overwrite existing output .pdb files.")
def main(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    num_steps: int,
    hf_token: str | None,
    hf_endpoint: str | None,
    overwrite: bool,
) -> None:
    """Read FASTA files and save one ESM3-predicted PDB file per sequence."""

    warnings.filterwarnings("ignore", category=FutureWarning)
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    _login(hf_token)
    device_obj = torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ESM3.from_pretrained(model_name).to(device_obj)
    processed = 0
    skipped = 0

    for fasta_path in sorted(input_dir.glob("*.fasta")):
        protein_id = fasta_path.stem
        output_path = output_dir / f"{protein_id}.pdb"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            record = SeqIO.read(fasta_path, "fasta")
            protein = ESMProtein(sequence=str(record.seq))
            protein = model.generate(protein, GenerationConfig(track="structure", num_steps=num_steps))
            protein.to_pdb(output_path)
            processed += 1
            print(output_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {fasta_path}: {exc}")
        finally:
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()

    print(f"ESM3 structure prediction complete. processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
