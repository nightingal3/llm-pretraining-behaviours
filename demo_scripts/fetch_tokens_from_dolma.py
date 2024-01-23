import requests
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import random
import json
import gzip
from io import BytesIO
import pyarrow
import pandas as pd
import os
import logging
from tqdm import tqdm

LLAMA_DIR = "/data/datasets/models/huggingface/meta-llama/Llama-2-70b-hf/"

TOKENS_TO_FETCH_10B = {
    "common-crawl": 5_186_000_000,
    "c4": 1_396_000_000,
    "peS2o": 796_000_000,
    "stack-code": 2_188_000_000,
    "gutenberg-books": 231_000_000,
    "wiki-en-simple": 200_000_000,
}

DUMP_FREQUENCY = 1_000_000


def parse_num(val: str) -> int:
    if val.lower().endswith("b"):
        return int(val[:-1]) * 1_000_000_000
    elif val.lower().endswith("m"):
        return int(val[:-1]) * 1_000_000
    else:
        try:
            return int(float(val))
        except:
            raise ValueError(
                "You must pass either an integer, scientific notation, or xB/xM for num tokens"
            )


def process_zipped_file(content: bytes, file_ind: int) -> list:
    if (file_ind % 50 == 0):
        print(f"Processing file {file_ind}")
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
    num_tokens: int, domain: str, output_dir: str or None, all_files_lst: list
):
    current_tokens = 0
    output_dir = output_dir if output_dir else f"./dolma/{domain}_{num_tokens}"
    logging.info(f"Fetching {num_tokens} tokens from {domain}")

    # shuffle
    random.seed(42)
    random.shuffle(all_files_lst)
    all_texts = []

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

            if response.status_code != 200:
                logging.info(f"Error fetching {all_files_lst[file_ind]}")
                continue

            file_ind += 1

            docs = [json.loads(l) for l in process_zipped_file(response.content, file_ind)]
            texts = [d["text"] for d in docs]

            # tokenizing individually to avoid oom
            for i, text in enumerate(texts):
                all_texts.append(docs[i])
                encoded_inputs = tokenizer(
                    text, truncation=True, padding=True, return_tensors="pt"
                )
                num_non_padding_toks = (
                    encoded_inputs["attention_mask"].sum(dim=1).tolist()
                )
                current_tokens += sum(num_non_padding_toks)
                pbar.update(sum(num_non_padding_toks))

                # save the reduced dataset as an arrow file, dump every 1M lines
                if current_tokens >= num_tokens or len(all_texts) >= DUMP_FREQUENCY:
                    part_ind += 1
                    output_file = f"{output_dir}/part_{part_ind}.arrow"
                    logging.info("Output file is ", output_file)
                    # mkdir -p
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)

                    # just keep main data
                    fields_to_keep = ["text", "id", "lang"]
                    all_texts = [
                        {k: v for k, v in line.items() if k in fields_to_keep}
                        for line in all_texts
                    ]
                    df = pd.DataFrame(all_texts)
                    df.to_parquet(output_file)
                    logging.info(
                        f"Wrote dataset of size {current_tokens} to {output_file}"
                    )

                    del df
                    all_texts = []

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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.num_tokens and args.num_total_tokens:
        raise ValueError(
            "Please specify only one of --num_tokens or --num_total_tokens (num_tokens is per domain, while num_total will calculate the appropriate number for a domain based on the total)"
        )

    if args.num_tokens:
        logging.info(f"Fetching {args.num_tokens} tokens")
        num_tokens = parse_num(args.num_tokens)
    # Calculate num_tokens from domain and num_total_tokens
    elif args.num_total_tokens and args.domain:
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
    elif args.domain:
        logging.info("Total tokens/domain tokens not specified, using 10B mix")
        num_tokens = TOKENS_TO_FETCH_10B[args.domain]

    # the flask server has to be up on clio
    all_files_lst = requests.get("http://128.2.209.71:5000/list-all").json()
    if args.domain:
        fetch_tokens(
            num_tokens=num_tokens,
            domain=args.domain,
            output_dir=args.output,
            all_files_lst=all_files_lst,
        )
    else:
        logging.info("Fetching from all domains following the 10B ratio mix")
        for domain in TOKENS_TO_FETCH_10B.keys():
            logging.info(f"Fetching {domain}")
            if args.num_total_tokens:
                logging.info("Calculating num_tokens from given args.num_total_tokens")
                num_tokens = (
                    int(
                        (
                            (parse_num(args.num_total_tokens) / 10_000_000_000)
                            * TOKENS_TO_FETCH_10B[domain]
                        )
                    )
                    // 1_000_000
                ) * 1_000_000
            else:
                logging.info("Calculating num_tokens from args.num_total_tokens = 10B")
                num_tokens = TOKENS_TO_FETCH_10B[domain]
            fetch_tokens(
                num_tokens=num_tokens,
                domain=domain,
                # Slightly jank - append domain dir here instead of in get_tokens.sh if running on all domains
                output_dir=args.output + f"/{domain}",
                all_files_lst=all_files_lst,
            )
