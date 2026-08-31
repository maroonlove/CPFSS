#!/usr/bin/env python
"""Run InterProScan and split TSV annotations into per-protein CSV files."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import click as ck
from Bio import SeqIO


def _merge_fastas(input_dir: Path, merged_fasta: Path, overwrite: bool) -> None:
    if merged_fasta.exists() and not overwrite:
        return

    merged_fasta.parent.mkdir(parents=True, exist_ok=True)
    with merged_fasta.open("w", encoding="utf-8") as handle:
        for fasta_path in sorted(input_dir.glob("*.fasta")):
            record = SeqIO.read(fasta_path, "fasta")
            protein_id = record.id or fasta_path.stem
            handle.write(f">{protein_id}\n{str(record.seq)}\n")


def _split_tsv(interproscan_output: Path, output_dir: Path, overwrite_csv: bool) -> None:
    opened_files = {}
    try:
        with interproscan_output.open("r", encoding="utf-8") as tsv_file:
            for line in tsv_file:
                if line.startswith("#") or not line.strip():
                    continue
                protein_id = line.split("\t", 1)[0]
                output_path = output_dir / f"{protein_id}.csv"
                mode = "w" if overwrite_csv and output_path not in opened_files else "a"
                if output_path not in opened_files:
                    opened_files[output_path] = output_path.open(mode, encoding="utf-8")
                opened_files[output_path].write(line.replace("\t", ","))
    finally:
        for handle in opened_files.values():
            handle.close()


@ck.command()
@ck.option("--input-dir", "-i", type=ck.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Directory containing input .fasta files.")
@ck.option("--output-dir", "-o", type=ck.Path(file_okay=False, path_type=Path), required=True, help="Directory for merged FASTA, InterProScan TSV, and per-protein CSV files.")
@ck.option("--interproscan-bin", type=ck.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Path to interproscan.sh.")
@ck.option("--cpu", type=int, default=32, show_default=True, help="CPU cores passed to InterProScan.")
@ck.option("--merged-fasta", type=ck.Path(dir_okay=False, path_type=Path), default=None, help="Optional merged FASTA path. Defaults to OUTPUT_DIR/all_sequences.fasta.")
@ck.option("--tsv-output", type=ck.Path(dir_okay=False, path_type=Path), default=None, help="Optional InterProScan TSV output path. Defaults to OUTPUT_DIR/interproscan.tsv.")
@ck.option("--run-interproscan/--split-only", default=True, show_default=True, help="Run InterProScan before splitting TSV output.")
@ck.option("--overwrite-merged/--reuse-merged", default=False, show_default=True, help="Overwrite merged FASTA if it already exists.")
@ck.option("--overwrite-csv/--append-csv", default=True, show_default=True, help="Overwrite per-protein CSV files instead of appending.")
def main(
    input_dir: Path,
    output_dir: Path,
    interproscan_bin: Path,
    cpu: int,
    merged_fasta: Path | None,
    tsv_output: Path | None,
    run_interproscan: bool,
    overwrite_merged: bool,
    overwrite_csv: bool,
) -> None:
    """Create InterPro function annotation CSV files for FASTA sequences."""

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_fasta = merged_fasta or output_dir / "all_sequences.fasta"
    tsv_output = tsv_output or output_dir / "interproscan.tsv"

    _merge_fastas(input_dir, merged_fasta, overwrite=overwrite_merged)
    if run_interproscan:
        command = [
            str(interproscan_bin),
            "-i",
            str(merged_fasta),
            "-o",
            str(tsv_output),
            "-f",
            "TSV",
            "-cpu",
            str(cpu),
        ]
        start_time = time.time()
        subprocess.run(command, check=True)
        print(f"InterProScan finished in {time.time() - start_time:.2f} seconds")

    if not tsv_output.exists():
        raise ck.UsageError(f"InterProScan TSV output does not exist: {tsv_output}")
    _split_tsv(tsv_output, output_dir, overwrite_csv=overwrite_csv)
    print(f"InterPro annotations saved to: {output_dir}")


if __name__ == "__main__":
    main()
