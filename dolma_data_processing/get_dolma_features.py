import argparse
import multiprocessing
import pyarrow
import pandas as pd

from calc_feature_utils import *

feature_registry = {
    "num_tokens": get_num_tokens,
    "char_len": get_num_chars,
    "lexical_diversity": get_lexical_diversity,
    "unique_tokens": get_num_unique_tokens,
    "seq_ind_tok": get_position_in_sequence,
    "num_times_token_appears": get_num_times_token_appears,
}
tokenize_sequence = {
    "num_tokens": True,
    "char_len": False,
    "lexical_diversity": True,
    "unique_tokens": True,
    "seq_ind_tok": True,
    "num_times_token_appears": True,   
}
def main(feature: str, input_filepath: str, output_filepath: str):
    feature_fn = feature_registry[feature]
    tokenize = tokenize_sequence[feature]

    with open(input_filepath, "rb") as f:
        try:
            reader = pyarrow.ipc.RecordBatchStreamReader(f)
            table = reader.read_all()
            df = table.to_pandas()
        except:
            df = pd.read_parquet(input_filepath)

    if "text" not in df.columns:
        df["text"] = df["content"]

    breakpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", choices=["entropy", "char_len", "num_tokens", "lexical_diversity", "unique_tokens", "seq_ind_tok"])
    parser.add_argument("--input", help="Input file (arrow)", type=str)
    parser.add_argument("--output", help="Output file (arrow)", type=str)

    args = parser.parse_args()
    main(args.feature, args.input, args.output)