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

def process_zipped_file(content: bytes) -> list:
    with gzip.open(BytesIO(content), "rt") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines

def fetch_tokens(num_tokens: int, domain: str, output_dir: str or None, all_files_lst: list):
    current_tokens = 0
    output_dir = output_dir if output_dir else f"./dolma/{domain}_{num_tokens}"
    print(f"Fetching {num_tokens} tokens from {domain}")

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

    i = 0
    part_ind = 0
    with tqdm(total=num_tokens) as pbar:
        while current_tokens < num_tokens:
            response = requests.get(f"http://128.2.209.71:5000/{all_files_lst[i]}")

            if response.status_code != 200:
                print(f"Error fetching {all_files_lst[i]}")
                continue
            
            i += 1

            docs = [json.loads(l) for l in process_zipped_file(response.content)]
            texts = [d["text"] for d in docs]
            #all_texts.extend(lines)

            # tokenizing individually to avoid oom
            for i, text in enumerate(texts):
                all_texts.append(docs[i])
                encoded_inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
                num_non_padding_toks = encoded_inputs["attention_mask"].sum(dim=1).tolist()
                current_tokens += sum(num_non_padding_toks)
                pbar.update(sum(num_non_padding_toks))

                #if current_tokens >= num_tokens:
                    #break

                # save the reduced dataset as an arrow file, dump every 1M lines
                if current_tokens >= num_tokens or len(all_texts) >= DUMP_FREQUENCY:
                    part_ind += 1
                    output_file = f"{output_dir}/part_{part_ind}.arrow"
                    print("Output file is ", output_file)
                    # mkdir -p
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)

                    # just keep main data
                    fields_to_keep = ["text", "id", "lang"]
                    all_texts = [{k: v for k, v in line.items() if k in fields_to_keep} for line in all_texts]
                    df = pd.DataFrame(all_texts)
                    df.to_parquet(output_file)
                    print(f"Wrote dataset of size {current_tokens} to {output_file}")

                    del df
                    all_texts = []
            

    print(f"Saved all output ({current_tokens} tokens)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", help="Number of tokens to fetch", type=int)
    parser.add_argument("--output", help="Output dir", type=str)
    parser.add_argument("--domain", help="Domains to fetch", type=str, choices=["peS2o", "common-crawl", "stack-code", "wiki-en-simple", "c4", "gutenberg-books"])
    args = parser.parse_args()

    # the flask server has to be up on clio
    all_files_lst = requests.get("http://128.2.209.71:5000/list-all").json()
    if args.domain:
        num_tokens = args.num_tokens if args.num_tokens else TOKENS_TO_FETCH_10B[args.domain]
        fetch_tokens(num_tokens=num_tokens, domain=args.domain, 
                     output_dir=args.output, all_files_lst=all_files_lst)
    else:
        for domain in TOKENS_TO_FETCH_10B.keys():
            num_tokens = args.num_tokens if args.num_tokens else TOKENS_TO_FETCH_10B[domain]
            fetch_tokens(num_tokens=num_tokens, domain=domain, 
                         output_dir=args.output, all_files_lst=all_files_lst)
