import pyarrow
import pyarrow.parquet as pq
import pandas
import argparse
import os


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


def multi_arrows_to_jsonl(arrow_files: list, jsonl_out: str) -> None:
    dfs = []
    for arrow_file in arrow_files:
        with open(arrow_file, "rb") as f:
            try:
                reader = pyarrow.ipc.RecordBatchStreamReader(f)
                table = reader.read_all()
                df = table.to_pandas()
            except:
                try:
                    df = pandas.read_parquet(arrow_file)
                except:
                    print(f"Error reading {arrow_file}")
                    continue

            df["text"] = df["content"]
            dfs.append(df)
    df = pandas.concat(dfs)
    df.to_json(jsonl_out, orient="records", lines=True)


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

            multi_arrows_to_jsonl(all_arrow_files, args.output)
