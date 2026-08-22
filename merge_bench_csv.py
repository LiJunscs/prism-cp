#!/usr/bin/env python3
"""
Merge benchmark CSV files for a specific CP size into a single consolidated CSV.

Reads all bench_*_cp<N>_*.csv files in the benchmarks/ directory (or a specified path),
strips comment lines, deduplicates, sorts by (model, phase, seq_length, cp_size),
and writes a single output CSV with all config metadata preserved as comments.

Usage:
    python benchmarks/merge_bench_csv.py --cp-size 8
    python benchmarks/merge_bench_csv.py --cp-size 4 --input-dir /path/to/csvs
    python benchmarks/merge_bench_csv.py --cp-size 8 --output benchmarks/results_cp8.csv
"""

import argparse
import glob
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Merge benchmark CSV files into one")
    p.add_argument("--cp-size", type=int, required=True,
                   help="Only merge CSV files with this CP size (matches filename pattern _cp<N>_)")
    p.add_argument("--input-dir", type=str, default=None,
                   help="Directory containing bench_*.csv files (default: same dir as this script)")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path (default: <input-dir>/bench_merged_cp<N>.csv)")
    return p.parse_args()


def extract_metadata(filepath):
    """Extract comment lines (metadata) from a CSV file."""
    metadata = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                # Parse "# Key: Value" format
                content = line.lstrip("# ").strip()
                if ":" in content:
                    key, _, value = content.partition(":")
                    metadata[key.strip()] = value.strip()
            else:
                break
    return metadata


def extract_data_rows(filepath):
    """Extract non-comment, non-header data rows from a CSV file."""
    rows = []
    header_seen = False
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Skip header row
            if not header_seen and line.startswith("model,"):
                header_seen = True
                continue
            rows.append(line)
    return rows


def main():
    args = parse_args()

    # Determine input directory
    if args.input_dir:
        input_dir = args.input_dir
    else:
        input_dir = os.path.dirname(os.path.abspath(__file__))

    # Find CSV files matching the specified CP size (filename contains _cp<N>_)
    pattern = os.path.join(input_dir, f"bench_*_cp{args.cp_size}_*.csv")
    csv_files = sorted(glob.glob(pattern))

    # Exclude the output file itself if it matches the pattern
    output_path = args.output or os.path.join(input_dir, f"bench_merged_cp{args.cp_size}.csv")
    output_path = os.path.abspath(output_path)
    csv_files = [f for f in csv_files if os.path.abspath(f) != output_path]

    if not csv_files:
        print(f"No files matching CP size={args.cp_size} found in '{input_dir}'.")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) with cp_size={args.cp_size}:")
    for f in csv_files:
        print(f"  {os.path.basename(f)}")

    # Collect all metadata (use the first file's config as reference,
    # note differences across files)
    all_metadata = []
    for f in csv_files:
        meta = extract_metadata(f)
        meta["_source_file"] = os.path.basename(f)
        all_metadata.append(meta)

    # Collect all data rows and deduplicate
    all_rows = []
    seen = set()
    for f in csv_files:
        rows = extract_data_rows(f)
        for row in rows:
            if row not in seen:
                seen.add(row)
                all_rows.append(row)

    # Detect CSV columns (check if peak_mem_gib column exists)
    has_mem = any("peak_mem_gib" in open(f).read() for f in csv_files)
    if has_mem:
        header = "model,phase,seq_length,cp_size,time_ms,tflops_per_gpu,mfu,peak_mem_gib"
    else:
        header = "model,phase,seq_length,cp_size,time_ms,tflops_per_gpu,mfu"

    # Sort rows: by model, phase order (fwd < bwd < fwd+bwd), seq_length, cp_size
    phase_order = {"fwd": 0, "bwd": 1, "fwd+bwd": 2}

    def sort_key(row):
        parts = row.split(",")
        model = parts[0]
        phase = parts[1]
        seq_len = int(parts[2])
        cp_size = int(parts[3])
        return (model, phase_order.get(phase, 9), seq_len, cp_size)

    all_rows.sort(key=sort_key)

    # Write merged output
    with open(output_path, "w") as f:
        # Write summary metadata
        f.write(f"# Merged benchmark results\n")
        f.write(f"# Source files: {len(csv_files)}\n")

        # Write common config from first file (if available)
        if all_metadata:
            ref = all_metadata[0]
            for key in ["GPUs", "CP size", "Hidden size", "Num attention heads",
                        "Num query groups (KV heads)", "GDN key heads", "GDN value heads",
                        "GDN key head dim", "GDN value head dim", "GDN conv kernel dim",
                        "Batch size", "Warmup iters", "Timed iters",
                        "GPU peak TFLOP/s", "dtype"]:
                if key in ref:
                    # Check if consistent across all files
                    values = set(m.get(key, "") for m in all_metadata)
                    if len(values) == 1:
                        f.write(f"# {key}: {ref[key]}\n")
                    else:
                        f.write(f"# {key}: {ref[key]} (varies across files)\n")

        f.write(f"#\n")
        f.write(f"# Source files:\n")
        for meta in all_metadata:
            src = meta.get("_source_file", "unknown")
            mod = meta.get("Module", "?")
            seq = meta.get("Seq length", "?")
            date = meta.get("Date", "?")
            f.write(f"#   {src} | module={mod} seq={seq} date={date}\n")
        f.write(f"#\n")

        # Write CSV
        f.write(header + "\n")
        for row in all_rows:
            f.write(row + "\n")

    print(f"\nMerged {len(all_rows)} data rows into: {output_path}")


if __name__ == "__main__":
    main()
