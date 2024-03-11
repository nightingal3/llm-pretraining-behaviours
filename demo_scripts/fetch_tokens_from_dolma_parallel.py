import requests
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import random
import json
import gzip
from io import BytesIO
import pandas as pd
import os
import logging
from tqdm import tqdm
import sys
import gc
import multiprocessing

LLAMA_DIR = "/data/datasets/models/huggingface/meta-llama/Llama-2-70b-hf/"

TOKENS_TO_FETCH_10B = {
    "common-crawl": 5_186_000_000,
    "c4": 1_396_000_000,
    "peS2o": 796_000_000,
    "stack-code": 2_188_000_000,
    "gutenberg-books": 231_000_000,
    "wiki-en-simple": 200_000_000,
}

MAX_DUMP_SIZE = 500_000_000


def parse_num(val: str) -> int:
    if val.lower().endswith("t"):
        return int(val[:-1]) * 1_000_000_000_000
    if val.lower().endswith("b"):
        return int(val[:-1]) * 1_000_000_000
    elif val.lower().endswith("m"):
        return int(val[:-1]) * 1_000_000
    else:
        try:
            return int(float(val))
        except:
            raise ValueError(
                "You must pass either an integer, scientific notation, or xT/xB/xM for num tokens"
            )


def process_zipped_file(content: bytes, file_ind: int) -> list:
    with gzip.open(BytesIO(content), "rt", errors="ignore") as f:
        try:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            return lines
        except Exception as e:
            print(f"Error occured while reading gzip: {e}")
            print(f"Skipping file {file_ind}")
            return []


def fetch_tokens(
    process_ind: int,
    num_tokens: int,
    domain: str,
    output_dir: str or None,
    all_files_lst: list,
    seed: int = 42,
):
    current_tokens = 0
    output_dir = output_dir if output_dir else f"./dolma/{domain}_{num_tokens}"
    logging.info(f"Fetching {num_tokens} tokens from {domain} on process {process_ind}")

    # shuffle
    random.seed(seed)
    random.shuffle(all_files_lst)
    texts_to_dump = []

    # filter out non-gz files
    all_files_lst = [f for f in all_files_lst if f.endswith(".gz")]

    # filter by top level domain
    all_files_lst = [f for f in all_files_lst if domain in f]
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    file_ind = 0
    part_ind = 0
    with tqdm(total=num_tokens) as pbar:
        while current_tokens < num_tokens and file_ind < len(all_files_lst):
            response = requests.get(
                f"http://128.2.209.71:5000/{all_files_lst[file_ind]}"
            )
            logging.info(f"Fetched {all_files_lst[file_ind]} on process {process_ind}")

            if response.status_code != 200:
                logging.info(f"Error fetching {all_files_lst[file_ind]}")
                continue

            file_ind += 1

            docs = [
                json.loads(l) for l in process_zipped_file(response.content, file_ind)
            ]

            # just keep main data
            fields_to_keep = ["text", "id", "lang"]
            docs = [{k: v for k, v in d.items() if k in fields_to_keep} for d in docs]

            # tokenizing individually to avoid oom
            for _, doc in enumerate(docs):
                encoded_inputs = tokenizer(doc["text"], return_tensors="pt")
                num_non_padding_toks = (
                    (encoded_inputs["attention_mask"] == 1).sum(dim=1).tolist()
                )
                current_tokens += sum(num_non_padding_toks)
                pbar.update(sum(num_non_padding_toks))

                texts_to_dump.append(doc)
                # save the reduced dataset as a <= 500 MB arrow file
                ## for table of random strings each with length 2000,
                ## parquet file size is roughly 250 * size in memory
                if (
                    current_tokens >= num_tokens
                    or sys.getsizeof(texts_to_dump) * 250 >= MAX_DUMP_SIZE
                ):
                    part_ind += 1
                    output_file = f"{output_dir}/part_p{process_ind}_{part_ind}.arrow"

                    # mkdir -p
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    df = pd.DataFrame(texts_to_dump, columns=fields_to_keep)
                    df.to_parquet(output_file)

                    # if the output file is too large, recursively split it in half
                    # can't accurately predict parquet compression rate so this is necessary
                    def split_file(output_file, df):
                        if os.path.getsize(output_file) > MAX_DUMP_SIZE:
                            os.remove(output_file)
                            output_file_1 = f"{output_file[:-6]}_1.arrow"
                            output_file_2 = f"{output_file[:-6]}_2.arrow"
                            os.makedirs(os.path.dirname(output_file_1), exist_ok=True)
                            os.makedirs(os.path.dirname(output_file_2), exist_ok=True)
                            df1 = df[: df.shape[0] // 2]
                            df2 = df[df.shape[0] // 2 :]
                            df1.to_parquet(output_file_1)
                            df2.to_parquet(output_file_2)
                            split_file(output_file_1, df1)
                            split_file(output_file_2, df2)
                            del df1
                            del df2
                        del df
                        gc.collect()

                    split_file(output_file, df)
                    texts_to_dump = []

                    if current_tokens >= num_tokens:
                        break

    logging.info(f"Saved all output ({current_tokens} tokens)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_tokens",
        help="Number of tokens to fetch. You can also write xB/xM to fetch x billions/millions",
        type=str,
    )
    parser.add_argument(
        "--num_total_tokens",
        help="Total number of tokens to fetch. You can also write xB/xM to fetch x billions/millions",
        type=str,
    )
    parser.add_argument("--output", help="Output dir", type=str)
    parser.add_argument(
        "--domain",
        help="Domains to fetch",
        type=str,
        choices=[
            "peS2o",
            "common-crawl",
            "stack-code",
            "wiki-en-simple",
            "c4",
            "gutenberg-books",
        ],
    )
    parser.add_argument(
        "--file_lsts_dir",
        help="Path to directory containing file lists to download",
        type=str,
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not args.file_lsts_dir:
        raise ValueError("Please specify path to file lists directory")

    if args.num_tokens and args.num_total_tokens:
        raise ValueError(
            "Please specify only one of --num_tokens or --num_total_tokens (num_tokens is per domain, while num_total will calculate the appropriate number for a domain based on the total)"
        )

    if not args.domain:
        raise ValueError("Please specify domain")

    if args.num_tokens:
        logging.info(f"Fetching {args.num_tokens} tokens")
        num_tokens = parse_num(args.num_tokens)
    # Calculate num_tokens from domain and num_total_tokens
    elif args.num_total_tokens:
        logging.info("Total domain tokens not specified, using 10B ratio mix")
        num_tokens = (
            int(
                (
                    (parse_num(args.num_total_tokens) / 10_000_000_000)
                    * TOKENS_TO_FETCH_10B[args.domain]
                )
            )
            // 1_000_000
        ) * 1_000_000
    # Calculate num_tokens from domain and num_total_tokens=10B
    else:
        logging.info("Total tokens/domain tokens not specified, using 10B mix")
        num_tokens = TOKENS_TO_FETCH_10B[args.domain]

    # walk through the files in file_lists_dir/domain
    file_lsts_lst = []
    for root, _, filenames in os.walk(os.path.join(args.file_lsts_dir, args.domain)):
        for filename in filenames:
            with open(os.path.join(root, filename), "r") as file:
                file_lst = file.read().split("\n")
                file_lsts_lst.append(file_lst)

    def fetch_file_lst(process_ind, file_lst, split_num):
        fetch_tokens(
            process_ind=process_ind,
            num_tokens=num_tokens // split_num,
            domain=args.domain,
            output_dir=args.output,
            all_files_lst=file_lst,
            seed=args.seed,
        )

    pool_args = [
        (i, file_lsts_lst[i], len(file_lsts_lst)) for i in range(len(file_lsts_lst))
    ]
    pool = multiprocessing.pool.Pool(len(pool_args))
    pool.starmap(fetch_file_lst, pool_args)
    pool.join()
