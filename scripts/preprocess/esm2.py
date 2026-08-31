#!/usr/bin/env python
"""Generate ESM2 sequence embeddings from FASTA files."""

from __future__ import annotations

from pathlib import Path

import click as ck
import torch
from Bio import SeqIO
from esm import pretrained


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@ck.command()
@ck.option("--input-dir", "-i", type=ck.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Directory containing input .fasta files.")
@ck.option("--output-dir", "-o", type=ck.Path(file_okay=False, path_type=Path), required=True, help="Directory where .pt embeddings will be written.")
@ck.option("--model-path", type=ck.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Local ESM2 checkpoint path, for example esm2_t33_650M_UR50D.pt.")
@ck.option("--repr-layer", type=int, default=33, show_default=True, help="ESM2 representation layer to save.")
@ck.option("--device", type=str, default="auto", show_default=True, help="Torch device, for example auto, cpu, cuda, cuda:0.")
@ck.option("--overwrite/--skip-existing", default=False, show_default=True, help="Overwrite existing output .pt files.")
def main(input_dir: Path, output_dir: Path, model_path: Path, repr_layer: int, device: str, overwrite: bool) -> None:
    """Read FASTA files and save one ESM2 embedding tensor per sequence."""

    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = _resolve_device(device)

    model, alphabet = pretrained.load_model_and_alphabet_local(str(model_path))
    model.eval()
    model.to(device_obj)
    batch_converter = alphabet.get_batch_converter()

    processed = 0
    skipped = 0
    for fasta_path in sorted(input_dir.glob("*.fasta")):
        protein_id = fasta_path.stem
        output_path = output_dir / f"{protein_id}.pt"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            record = SeqIO.read(fasta_path, "fasta")
            sequence = str(record.seq)
            _, _, batch_tokens = batch_converter([(protein_id, sequence)])
            batch_tokens = batch_tokens.to(device_obj)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            embeddings = results["representations"][repr_layer].cpu()
            torch.save(embeddings, output_path)
            processed += 1
            print(output_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {fasta_path}: {exc}")
        finally:
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()

    print(f"ESM2 embeddings complete. processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
