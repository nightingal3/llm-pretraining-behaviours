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

LLAMA_DIR = "/data/models/huggingface/meta-llama/Llama-2-70b-hf/"

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
    if val.lower() == "all":
        return float("inf")
    elif val.lower().endswith("t"):
        return int(val[:-1]) * 1_000_000_000_000
    elif val.lower().endswith("b"):
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


def tokenize_texts(texts: list, tokenizer: AutoTokenizer) -> list:
    return tokenizer(texts, return_tensors="pt")


def save_state(file_index: int, curr_tokens: int, output_dir: str) -> None:
    state = {
        "last_processed_file_ind": file_index,
        "num_current_tokens": curr_tokens,
    }

    with open(f"{output_dir}/state.json", "w") as f:
        json.dump(state, f)


def load_state(output_dir: str) -> dict:
    try:
        with open(f"{output_dir}/state.json", "r") as f:
            return json.load(f)
    except:
        return {"last_processed_file_ind": 0, "num_current_tokens": 0}


def process_files(files: list, tokenizer: AutoTokenizer) -> list:
    try:
        texts = [json.loads(l)["text"] for l in files]
        return tokenize_texts(texts, tokenizer)
    except Exception as e:
        print(f"Error occured while reading gzip: {e}")
        return []


def process_zipped_file(content: bytes, file_ind: int) -> list:
    if file_ind % 50 == 0:
        print(f"Processing file {file_ind}")
    try:
        with gzip.open(BytesIO(content), "rt", errors="ignore") as f:
            for line in f:
                processed_line = line.strip()
                yield processed_line
    except Exception as e:
        print(f"Error occured while reading gzip: {e}")
        print(f"Skipping file {file_ind}")
        return []


def _fetch_tokens(
    num_tokens: int,
    domain: str,
    output_dir: str or None,
    all_files_lst: list,
    seed: int = 42,
):
    current_tokens = 0
    output_dir = output_dir if output_dir else f"./dolma/{domain}_{num_tokens}"
    logging.info(f"Fetching {num_tokens} tokens from {domain}")

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
                ## parquet file size is roughly 500 * size in memory
                if (
                    current_tokens >= num_tokens
                    or sys.getsizeof(texts_to_dump) * 500 >= MAX_DUMP_SIZE
                ):
                    part_ind += 1
                    output_file = f"{output_dir}/part_{part_ind}.arrow"
                    logging.info(f"Output file is: {output_file}")

                    # mkdir -p
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    df = pd.DataFrame(texts_to_dump, columns=fields_to_keep)
                    df.to_parquet(output_file)

                    # save state in case we crash due to OOM in the next iteration
                    save_state(file_ind, current_tokens, f"{output_dir}/state.json")

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

                    if current_tokens >= num_tokens:
                        break

    logging.info(f"Saved all output ({current_tokens} tokens)")


def fetch_and_process(file_info, progress_queue):
    file_url, file_index = file_info
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    response = requests.get(file_url)
    if response.status_code == 200:
        docs = [
            json.loads(line)
            for line in process_zipped_file(response.content, file_index)
        ]

        tokenized_data = []
        num_tokens = 0
        for doc in docs:
            encoded_inputs = tokenizer(doc["text"], return_tensors="pt")
            tokens_count = (encoded_inputs["attention_mask"] == 1).sum().item()
            num_tokens += tokens_count
            tokenized_data.append((doc, tokens_count))
        progress_queue.put(num_tokens)
        return tokenized_data
    else:
        logging.error(f"Failed to fetch data from {file_url}")
        return []


def fetch_tokens(num_tokens, domain, output_dir, all_files_lst, seed=42):
    current_tokens = 0
    output_dir = output_dir if output_dir else f"./dolma/{domain}_{num_tokens}"
    logging.info(f"Fetching {num_tokens} tokens from {domain}")

    # Shuffle the file list
    random.seed(seed)
    random.shuffle(all_files_lst)

    # Setup Manager and queue for tracking progress
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    # Prepare the list of URLs and indices
    file_info_list = [
        (f"http://128.2.209.71:5000/{file}", i)
        for i, file in enumerate(all_files_lst)
        if file.endswith(".gz")
    ]

    # Create a pool of workers to process files
    with multiprocessing.Pool(
        processes=(
            multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1
        )
    ) as pool:  # Adjust based on your CPU
        result_objects = [
            pool.apply_async(fetch_and_process, args=(info, progress_queue))
            for info in file_info_list
        ]
        pool.close()

        # Setup tqdm progress bar
        pbar = tqdm(total=num_tokens)
        while any(res.ready() == False for res in result_objects):
            while not progress_queue.empty():
                tokens_processed = progress_queue.get()
                pbar.update(tokens_processed)

        pool.join()
        pbar.close()

    # Aggregate results
    results = [res.get() for res in result_objects]
    texts_to_dump = []
    for result in results:
        for doc, _ in result:
            texts_to_dump.append(doc)

    # Saving the processed data
    save_to_disk(texts_to_dump, output_dir)


def save_to_disk(data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "processed_data.parquet")
    df = pd.DataFrame(data)
    df.to_parquet(output_file)

    logging.info(f"Saved all output to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_tokens",
        help="Number of tokens to fetch. You can also write xT/xB/xM to fetch x billions/millions. Write 'all' to fetch all tokens (may take a lot of space)",
        type=str,
    )
    parser.add_argument(
        "--num_total_tokens",
        help="Total number of tokens to fetch. You can also write xT/xB/xM to fetch x billions/millions.",
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
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
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
        _fetch_tokens(
            num_tokens=num_tokens,
            domain=args.domain,
            output_dir=args.output,
            all_files_lst=all_files_lst,
            seed=args.seed,
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
