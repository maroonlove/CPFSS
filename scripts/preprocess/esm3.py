#!/usr/bin/env python
"""Generate ESM3 sequence/function embeddings from FASTA and InterPro CSV files."""

from __future__ import annotations

import csv
import os
import re
import warnings
from pathlib import Path

import click as ck
import torch
from Bio import SeqIO
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein
from huggingface_hub import login


class FunctionAnnotation:
    """Small annotation object compatible with ESM3 function annotation fields."""

    def __init__(self, name: str, label: str, start: int, end: int) -> None:
        self.name = name
        self.label = label
        self.start = start
        self.end = end


def _login(token: str | None) -> None:
    token = token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)


def _load_supported_interpro_ids(path: Path) -> set[str]:
    interpro_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if row:
                interpro_ids.add(row[0])
    return interpro_ids


def _read_function_annotations(path: Path, protein_id: str, supported_interpro_ids: set[str]) -> list[FunctionAnnotation]:
    annotations = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            interpro_id = next((part for part in parts if re.fullmatch(r"IPR\d{6}", part)), None)
            if not interpro_id or interpro_id not in supported_interpro_ids:
                continue
            numbers = [part for part in parts if part.isdigit()]
            if len(numbers) < 3:
                continue
            annotations.append(FunctionAnnotation(protein_id, interpro_id, int(numbers[1]), int(numbers[2])))
    return annotations


@ck.command()
@ck.option("--input-dir", "-i", type=ck.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Directory containing input .fasta files.")
@ck.option("--interpro-dir", type=ck.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Directory containing per-protein InterPro .csv files.")
@ck.option("--output-dir", "-o", type=ck.Path(file_okay=False, path_type=Path), required=True, help="Directory where ESM3 .pt embeddings will be written.")
@ck.option("--interpro-map", type=ck.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Path to interpro_29026_to_keywords_58641.csv from the ESM installation.")
@ck.option("--model-name", type=str, default="esm3-sm-open-v1", show_default=True, help="ESM3 model name passed to ESM3.from_pretrained.")
@ck.option("--device", type=str, default="cuda:0", show_default=True, help="Torch device, for example cpu, cuda, cuda:1.")
@ck.option("--hf-token", type=str, default=None, help="Hugging Face token. If omitted, HF_TOKEN environment variable is used.")
@ck.option("--hf-endpoint", type=str, default=None, help="Optional Hugging Face endpoint mirror, for example https://hf-mirror.com.")
@ck.option("--overwrite/--skip-existing", default=False, show_default=True, help="Overwrite existing output .pt files.")
@ck.option("--skip-missing-interpro/--error-missing-interpro", default=True, show_default=True, help="Skip FASTA files without a matching InterPro CSV.")
def main(
    input_dir: Path,
    interpro_dir: Path,
    output_dir: Path,
    interpro_map: Path,
    model_name: str,
    device: str,
    hf_token: str | None,
    hf_endpoint: str | None,
    overwrite: bool,
    skip_missing_interpro: bool,
) -> None:
    """Read FASTA and InterPro annotations and save one ESM3 embedding per protein."""

    warnings.filterwarnings("ignore", category=FutureWarning)
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    _login(hf_token)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    supported_interpro_ids = _load_supported_interpro_ids(interpro_map)
    model = ESM3.from_pretrained(model_name).to(device_obj)

    processed = 0
    skipped = 0
    errors = []

    for fasta_path in sorted(input_dir.glob("*.fasta")):
        protein_id = fasta_path.stem
        output_path = output_dir / f"{protein_id}.pt"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        function_path = interpro_dir / f"{protein_id}.csv"
        if not function_path.exists():
            message = f"Missing InterPro CSV for {protein_id}: {function_path}"
            if skip_missing_interpro:
                print(message)
                skipped += 1
                continue
            raise ck.UsageError(message)

        try:
            record = SeqIO.read(fasta_path, "fasta")
            annotations = _read_function_annotations(function_path, protein_id, supported_interpro_ids)
            if not annotations:
                skipped += 1
                continue

            protein = ESMProtein(sequence=str(record.seq), function_annotations=annotations)
            encoded = model.encode(protein)
            inputs = {
                "sequence_tokens": encoded.sequence.unsqueeze(0).to(device_obj),
                "function_tokens": encoded.function.unsqueeze(0).to(device_obj),
            }
            model = model.to(torch.float32)
            with torch.no_grad():
                outputs = model(**inputs)
            torch.save(outputs.embeddings.cpu(), output_path)
            processed += 1
            print(output_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(fasta_path))
            print(f"Error processing {fasta_path}: {exc}")
        finally:
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()

    print(f"ESM3 embeddings complete. processed={processed}, skipped={skipped}, errors={len(errors)}")


if __name__ == "__main__":
    main()
