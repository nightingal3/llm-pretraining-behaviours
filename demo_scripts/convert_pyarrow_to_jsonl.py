import pyarrow
import pyarrow.parquet as pq
import pandas
import argparse
import os
import gc
from tqdm import tqdm


def arrow_to_jsonl(arrow_file: str, jsonl_out: str) -> None:
    with open(arrow_file, "rb") as f:
        try:
            reader = pyarrow.ipc.RecordBatchStreamReader(f)
            table = reader.read_all()
            df = table.to_pandas()
        except:
            df = pandas.read_parquet(arrow_file)

        if "text" not in df.columns:
            df["text"] = df["content"]
        if not os.path.exists(os.path.dirname(jsonl_out)):
            os.makedirs(os.path.dirname(jsonl_out), exist_ok=True)
        df.to_json(jsonl_out, orient="records", lines=True)
        del df
        gc.collect()


def multi_arrows_to_jsonl_chunked(
    arrow_files: list, jsonl_out: str, chunk_size: int = 100000
) -> None:
    os.makedirs(os.path.dirname(jsonl_out), exist_ok=True)
    file_mode = "a" if os.path.exists(jsonl_out) else "w"

    for arrow_file in arrow_files:
        print("file:", arrow_file)

        dataset = pq.ParquetFile(arrow_file)
        num_rows = dataset.metadata.num_rows

        with open(jsonl_out, file_mode) as jsonl_file, tqdm(
            total=num_rows, desc=f"Progress: "
        ) as pbar:
            for batch in dataset.iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()
                if "text" not in df.columns:
                    df["text"] = df["content"]
                df.to_json(jsonl_file, orient="records", lines=True, force_ascii=False)
                pbar.update(len(df))
                del df
        file_mode = "a"  # Ensure that subsequent files append to the output file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.input.endswith(".arrow"):
        arrow_to_jsonl(args.input, args.output)
    else:
        if os.path.isdir(args.input):
            all_arrow_files = []
            for f in os.listdir(args.input):
                if f.endswith(".arrow"):
                    all_arrow_files.append(os.path.join(args.input, f))

            multi_arrows_to_jsonl_chunked(all_arrow_files, args.output)
