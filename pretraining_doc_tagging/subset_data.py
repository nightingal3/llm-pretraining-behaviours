import pyarrow as pa
import pyarrow.dataset as ds
import numpy as np
import json
from pathlib import Path
import argparse
import os


def create_random_subset(
    input_dir, output_file, subset_size, batch_size=10000, seed=42, compression="zstd"
):
    """Create a random subset from parquet/arrow/jsonl files and output as arrow."""
    np.random.seed(seed)

    input_path = Path(input_dir)
    arrow_files = list(input_path.glob("*.arrow"))
    jsonl_files = list(input_path.glob("*.jsonl")) + list(input_path.glob("*.json"))
    parquet_files = list(input_path.glob("*.parquet"))

    print(os.listdir(input_dir))
    print(len(parquet_files), len(arrow_files), len(jsonl_files))
    # Determine input format
    if parquet_files:
        dataset = ds.dataset(input_dir, format="parquet")
    elif arrow_files:
        dataset = ds.dataset(input_dir, format="arrow")
    elif jsonl_files:
        total_rows = sum(1 for f in jsonl_files for _ in open(f))
        sampling_prob = min(subset_size / total_rows, 1.0)
        rows_sampled = 0

        with open(output_file, "w") as out_f:
            for file_path in jsonl_files:
                with open(file_path, "r") as in_f:
                    for line in in_f:
                        if np.random.random() < sampling_prob:
                            out_f.write(line)
                            rows_sampled += 1
                            if rows_sampled % 1000 == 0:
                                print(f"Sampled {rows_sampled} rows", end="\r")
        print(f"\nFinished! Sampled {rows_sampled} rows out of {total_rows}")
        return rows_sampled
    else:
        raise ValueError(f"No supported files found in {input_dir}")

    # Process arrow/parquet datasets
    total_rows = 0
    schema = None
    for batch in dataset.to_batches(batch_size=batch_size):
        if schema is None:
            schema = batch.schema
        total_rows += len(batch)

    sampling_prob = min(subset_size / total_rows, 1.0)
    print(f"Total rows: {total_rows}, sampling prob: {sampling_prob}")

    # Write to arrow
    compress_options = (
        pa.ipc.IpcWriteOptions(compression=pa.Codec(compression))
        if compression
        else None
    )
    rows_sampled = 0

    with pa.ipc.RecordBatchFileWriter(
        output_file, schema, options=compress_options
    ) as writer:
        for batch in dataset.to_batches(batch_size=batch_size):
            mask = np.random.random(len(batch)) < sampling_prob
            subset_batch = batch.filter(mask)
            if len(subset_batch) > 0:
                writer.write_batch(subset_batch)
                rows_sampled += len(subset_batch)
                print(f"Sampled {rows_sampled} rows", end="\r")

    print(f"\nFinished! Sampled {rows_sampled} rows out of {total_rows}")
    return rows_sampled


def main():
    parser = argparse.ArgumentParser(description="Create a random subset of datasets")
    parser.add_argument("--input-dir", required=True, help="Directory containing files")
    parser.add_argument("--output-file", required=True, help="Path to output file")
    parser.add_argument(
        "--subset-size", type=int, required=True, help="Number of rows in subset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10000, help="Batch size for processing"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compression", choices=["zstd", "lz4", None], default="zstd")

    args = parser.parse_args()
    if not args.output_file.endswith(".arrow"):
        args.output_file = args.output_file + ".arrow"
    create_random_subset(**vars(args))


if __name__ == "__main__":
    main()
