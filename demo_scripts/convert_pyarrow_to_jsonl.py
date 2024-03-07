import pyarrow
import pyarrow.parquet as pq
import pandas
import argparse
import os
from tqdm import tqdm
import gc


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
    is_first_file = True
    if os.path.exists(jsonl_out):
        os.remove(jsonl_out)

    with open(jsonl_out, "a", encoding="utf-8") as fout:
        for arrow_file in tqdm(arrow_files):
            print("Arrow file: ", arrow_file)
            with open(arrow_file, "rb") as f:
                try:
                    try:
                        reader = pyarrow.ipc.RecordBatchStreamReader(f)
                        table = reader.read_all()

                        df = table.to_pandas()
                    except:
                        df = pandas.read_parquet(arrow_file)

                    if "text" not in df.columns:
                        df["text"] = df["content"]
                except:
                    print("Error reading file: ", arrow_file)
                    continue

            json_str = df.to_json(
                orient="records",
                lines=True,
                force_ascii=False,
            )
            fout.write(json_str)
            if is_first_file:
                is_first_file = False
            else:
                fout.write("\n")

            fout.write(json_str)
            fout.write("\n")

            # I get an oom error after a while which shouldn't happen in theory...
            del df
            gc.collect()


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
            for root, dirs, files in os.walk(args.input):
                for file in files:
                    if file.endswith(".arrow"):
                        all_arrow_files.append(os.path.join(root, file))

            multi_arrows_to_jsonl(all_arrow_files, args.output)
