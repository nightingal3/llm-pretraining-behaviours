import argparse
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Callable, Iterable, Dict
from functools import partial
import math
import datasets
import pickle
import pandas as pd
import multiprocessing
import torch
import tiktoken
import string
from tqdm import tqdm
import seaborn as sns
from itertools import islice
import matplotlib.pyplot as plt
import sys
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging
from pathlib import Path
import json

# if tiktoken says "permission denied", do this:
# export TIKTOKEN_CACHE_DIR = "desired_path_to_tmp"
def preproc(orig_text: str, tokenize: bool = True, remove_punct: bool = False) -> str:
    text = orig_text
    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))  # Fixed: remove after assignment
    if tokenize:
        encoding = tiktoken.get_encoding("p50k_base")
        encoded_text = encoding.encode(text, disallowed_special=(encoding.special_tokens_set - {"<|endoftext|>"}))
        decoded_text, offsets = encoding.decode_with_offsets(encoded_text)
        tokenized_words = []
        for i in range(len(offsets)):
            start_idx = offsets[i]
            end_idx = offsets[i + 1] if i + 1 < len(offsets) else len(decoded_text)
            tokenized_words.append(decoded_text[start_idx:end_idx])
        text = tokenized_words
    return text

def preproc_parallel(
    docs_list: List[str], tokenize: bool = True, remove_punct: bool = False
) -> List[str]:
    preproc_fn = partial(preproc, tokenize=tokenize, remove_punct=remove_punct)
    chunk_fn = partial(preproc_chunk, preproc_fn=preproc_fn)

    doc_chunks = list(
        chunk_data(docs_list, len(docs_list) // multiprocessing.cpu_count())
    )
    print(f"Created {len(doc_chunks)} chunks")

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(chunk_fn, doc_chunks),
                total=len(doc_chunks),
                desc="Preprocessing"
            )
        )

    pool.close()
    pool.join()
    flattened_docs = [item for sublist in results for item in sublist]
    return flattened_docs

def get_doc_entropy_stats(doc_tokens: List[int], entropy_by_ngram: Dict, n: int) -> Optional[Dict]:
    """Get entropy stats for a single document using corpus-level distributions"""
    if not doc_tokens or len(doc_tokens) <= n:
        return None
        
    doc_entropies = []
    for i in range(len(doc_tokens) - n):
        curr_ngram = tuple(doc_tokens[i:i + n])
        if curr_ngram in entropy_by_ngram:
            doc_entropies.append(entropy_by_ngram[curr_ngram])
    
    if not doc_entropies:
        return None

    return {
        'mean_doc_entropy': float(np.mean(doc_entropies)),
        'max_doc_entropy': float(max(doc_entropies)),
        'min_doc_entropy': float(min(doc_entropies)),
        'entropy_percentiles': {
            'p25': float(np.percentile(doc_entropies, 25)),
            'p75': float(np.percentile(doc_entropies, 75))
        }
    }

def preproc_chunk(chunk: List[List], preproc_fn: Callable):
    print(f"Processing chunk of {len(chunk)} documents")
    return [preproc_fn(doc) for doc in chunk]


def compute_entropy_counts(
    doc_tokens: List[List], n: int = 1
) -> Tuple[dict, dict, float]:
    """
    Compute entropy over next-word distribution given counts in the corpus
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    next_token_counts = defaultdict(Counter)
    context_counts = defaultdict(int)
    num_skipped = 0
    print("Getting counts...")
    for tokens in tqdm(doc_tokens):
        if not tokens or len(tokens) <= n:
            num_skipped += 1
            continue
        for i in range(len(tokens) - n):
            curr_ngram = tuple(tokens[i : i + n])
            next_token = tokens[i + n]
            next_token_counts[curr_ngram][next_token] += 1
            context_counts[curr_ngram] += 1

    entropy_vals = {}
    weighted_entropy_sum = 0
    print("Calculating entropy over next tokens...")
    entropy_vals, weighted_entropy_sum = compute_entropy_parallel_with_chunking(
        next_token_counts, context_counts
    )
    print("Num seqs skipped: ", num_skipped)

    return entropy_vals, next_token_counts, weighted_entropy_sum


def init_model_hf(
    model_name: str, device: str = "cpu"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, AutoConfig]:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    model.eval()
    return model, tokenizer, config


def compute_entropy_lm(
    args: Tuple[List[List], Optional[int], str, str]
) -> Tuple[dict, dict, float]:
    # NOTE: not using this
    doc_tokens, max_window_size, model_name, device = args
    device_num = int(device.split(":")[1])
    torch.cuda.set_device(device_num)

    model, tokenizer, config = init_model_hf(model_name, device)
    print(f"Loaded model on GPU #{device}")
    if max_window_size is None:
        max_window_size = config.max_position_embeddings

    context_counts = defaultdict(int)
    entropy_vals = defaultdict(list)

    weighted_entropy_sum = 0
    max_window_size = config.max_position_embeddings

    for i, doc in tqdm(enumerate(doc_tokens)):
        # Note: there's no preproc when method="lm" is true, so we can pass this directly
        input_ids = tokenizer(doc, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            if input_ids.shape[1] < max_window_size:
                logits = model(input_ids).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            else:
                logits = []
                start_ind = 0

                while start_ind < input_ids.shape[1]:
                    end_ind = start_ind + max_window_size
                    chunk = input_ids[:, start_ind:end_ind]
                    chunk_logits = model(chunk).logits
                    logits.append(chunk_logits[:, :-1, :])
                    start_ind = end_ind - 1
                logits = torch.cat(logits, dim=1).to(device)
                probs = torch.nn.functional.softmax(logits, dim=-1)

                del chunk, chunk_logits
                torch.cuda.empty_cache()

            for i in range(input_ids.shape[1] - 1):
                # don't save context identity here.
                context = i
                next_token_dist = probs[0, i]
                entropy = -torch.sum(
                    next_token_dist * torch.log2(next_token_dist)
                ).item()
                context_counts[context] = 1
                entropy_vals[context] = entropy

            # for some reason OOM happens after a few iterations
            del input_ids, logits, probs
            torch.cuda.empty_cache()

    return (
        entropy_vals,
        context_counts,
        -1,
    )  # TODO: need to fix the weighted entropy sum


def compute_entropy_lm_parallel(
    doc_tokens: List[List],
    max_window_size: Optional[int] = None,
    model_name: str = "EleutherAI/gpt-neo-1.3B",
) -> Tuple[dict, dict, float]:
    # Split documents into chunks for each GPU
    # num_gpus = torch.cuda.device_count()
    # assert num_gpus > 1, "You need >1 GPU to parallelize this"
    num_gpus = 1
    print(f"Working on {num_gpus} GPUs")

    chunk_size = len(doc_tokens) // num_gpus
    print(chunk_size)
    # doc_chunks = [doc_tokens[i:i+chunk_size] for i in range(0, len(doc_tokens), chunk_size)]
    doc_chunks = [doc_tokens]
    # compute_entropy_lm([doc_chunks[0], None, model_name, "cuda:0"])

    with multiprocessing.Pool(num_gpus) as pool:
        results = list(
            tqdm(
                pool.imap(
                    compute_entropy_lm,
                    [
                        (chunk, max_window_size, model_name, f"cuda:{i}")
                        for i, chunk in enumerate(doc_chunks)
                    ],
                ),
                position=0,
                file=sys.stdout,
            )
        )

    pool.close()
    pool.join()
    # Aggregate the results
    all_entropy_vals = {}
    all_context_counts = defaultdict(int)
    total_weighted_entropy_sum = 0

    for entropy_vals, context_counts, weighted_entropy_sum in results:
        all_entropy_vals.update(entropy_vals)
        for k, v in context_counts.items():
            all_context_counts[k] += v
        total_weighted_entropy_sum += weighted_entropy_sum

    return all_entropy_vals, all_context_counts, total_weighted_entropy_sum


# Helper function to chunk the data
def chunk_data(data, chunk_size):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

# Modified worker function to handle chunks
def compute_entropy_for_chunk(chunk):
    chunk_results = []
    for ngram, next_token_counter, context_counts, total_context_counts in chunk:
        total = sum(next_token_counter.values())
        entropy = -sum(
            (count / total) * math.log2(count / total)
            for count in next_token_counter.values()
        )
        weighted_entropy = (context_counts[ngram] / total_context_counts) * entropy
        chunk_results.append((ngram, entropy, weighted_entropy))
    return chunk_results


# Parallelized entropy computation with chunking
def compute_entropy_parallel_with_chunking(next_token_counts, context_counts, chunk_size=100000):
    entropy_vals = {}
    weighted_entropy_sum = 0
    total_context_counts = sum(context_counts.values())

    # Prepare all data first
    data = [
        (ngram, next_token_counter, context_counts, total_context_counts)
        for ngram, next_token_counter in next_token_counts.items()
    ]
    
    # Create chunks more efficiently
    num_chunks = (len(data) + chunk_size - 1) // chunk_size
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    
    num_workers = min(multiprocessing.cpu_count(), num_chunks)  # Don't create more workers than chunks
    print(f"Processing {num_chunks} chunks using {num_workers} workers...")
    
    with multiprocessing.Pool(num_workers) as pool:
        chunked_results = list(
            tqdm(
                pool.imap(compute_entropy_for_chunk, chunks),
                total=len(chunks),
                desc="Computing entropy"
            )
        )

    # Process results in batches to save memory
    for chunk_result in chunked_results:
        for ngram, entropy, weighted_entropy in chunk_result:
            entropy_vals[ngram] = entropy
            weighted_entropy_sum += weighted_entropy
        
    return entropy_vals, weighted_entropy_sum
def plot_histogram_entropy_vals(
    entropy_df: pd.DataFrame,
    out_filename: str = "entropy_dist",
    show_top_ngrams: bool = False,
) -> None:
    entropy_df = entropy_df.sort_values(by="entropy", ascending=False)
    entropy_df["log_entropy"] = np.log1p(entropy_df["entropy"])
    # entropy_df = entropy_df[entropy_df["entropy"] > 0] # get rid of ngrams with only one continuation for better viz
    fig, main_ax = plt.subplots(figsize=(12, 6))

    if "language" in entropy_df.columns:
        sns.histplot(entropy_df, x="log_entropy", kde=True, hue="language")
    else:
        sns.histplot(entropy_df, x="log_entropy", kde=True)

    if show_top_ngrams:
        # annotate with highest/lowest vals for illustration
        top_ngrams = entropy_df.head(10)
        bottom_ngrams = entropy_df.head(10)

        # Create an inset axis to list top ngrams, positioned to the right of the main figure
        axins = main_ax.inset_axes([1.05, 0.6, 0.4, 0.3], transform=main_ax.transAxes)
        axins.axis("off")  # Turn off axis
        title_fontsize = 12
        ngram_fontsize = 10
        line_spacing = 0.18

        # TODO: Draw a box around the inset

        # Add top ngrams to inset
        axins.text(
            0.3,
            1,
            "Top Ngrams by Entropy:",
            transform=axins.transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            fontsize=ngram_fontsize,
            fontweight="bold",
        )
        for idx, row in top_ngrams.iterrows():
            axins.text(
                0,
                1 - (idx + 1) * 0.2,
                f"{row['ngram']} : {row['entropy']:.2f}",
                transform=axins.transAxes,
                verticalalignment="top",
                fontsize=ngram_fontsize,
            )

    # Save plot
    plt.tight_layout()
    plt.xlabel("Log entropy over next-token distribution")
    plt.savefig(f"{out_filename}.png")
    plt.savefig(f"{out_filename}.pdf")


def analyze_entropy(
    iterable_collection: Iterable[Iterable],
    lang_option: Optional[str] = None,
    method_type: str = "counts",
) -> pd.DataFrame:
    if lang_option is not None:
        assert len(lang_option) == len(
            iterable_collection
        ), "language list must match number of document subsets"

    all_words, langs = [], []
    all_dfs = []
    # TODO: let's run for 100k or 1M documents to test out multiproc
    for i, iterable in enumerate(iterable_collection):
        print("Preprocessing...")
        all_words = preproc_parallel(
            iterable, tokenize=method_type == "counts", remove_punct=args.remove_punct
        )
        if method_type == "counts":
            (
                entropy_by_ngram,
                next_token_counts,
                weighted_avg_entropy,
            ) = compute_entropy_counts(all_words, n=args.ngram)
        else:
            (
                entropy_by_ngram,
                next_token_counts,
                weighted_avg_entropy,
            ) = compute_entropy_lm_parallel(all_words)

        entropy_by_ngram = sorted(
            entropy_by_ngram.items(), key=lambda x: x[1], reverse=True
        )

        print(entropy_by_ngram[:10])
        print(entropy_by_ngram[-10:])
        # print("AVG ENTROPY: ", weighted_avg_entropy)

        df = pd.DataFrame(entropy_by_ngram, columns=["ngram", "entropy"])
        if lang_option:
            df["language"] = lang_option[i]
        all_dfs.append(df)

    return pd.concat(all_dfs, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                       help="Input file (parquet/jsonl/arrow)")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Column containing text to analyze")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ngram", type=int, default=1,
                       help="Context size for entropy calculation")
    parser.add_argument("--num_docs", type=int, default=None,
                       help="Limit number of documents to process")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    elif args.input.endswith('.jsonl'):
        df = pd.read_json(args.input, lines=True)
    else:
        import datasets
        df = datasets.load_from_disk(args.input)
        df = pd.DataFrame(df)

    if args.num_docs:
        df = df.head(args.num_docs)

    # Preprocess texts
    texts = df[args.text_column].tolist()
    tokenized_texts = preproc_parallel(texts)

    # Compute corpus-level entropy
    entropy_by_ngram, context_counts, total_contexts = compute_entropy_counts(
        tokenized_texts, 
        n=args.ngram
    )

    # Compute document-level stats
    doc_features = []
    for doc_id, tokens in enumerate(tqdm(tokenized_texts, desc="Computing document stats")):
        doc_stats = get_doc_entropy_stats(tokens, entropy_by_ngram, args.ngram)
        if doc_stats:
            doc_features.append({
                'id': doc_id,
                'feature': doc_stats
            })

    # Save document-level features
    doc_df = pd.DataFrame(doc_features)
    doc_df.to_parquet(output_dir / f"doc_entropy_n{args.ngram}.parquet")

    # Save corpus-level stats
    entropy_vals = list(entropy_by_ngram.values())
    corpus_stats = {
        'mean_corpus_entropy': float(np.mean(entropy_vals)),
        'median_corpus_entropy': float(np.median(entropy_vals)),
        'distribution_percentiles': {
            'p10': float(np.percentile(entropy_vals, 10)),
            'p25': float(np.percentile(entropy_vals, 25)),
            'p75': float(np.percentile(entropy_vals, 75)),
            'p90': float(np.percentile(entropy_vals, 90))
        },
        'num_unique_contexts': len(entropy_by_ngram),
        'total_contexts': total_contexts
    }
    
    with open(output_dir / f"corpus_entropy_n{args.ngram}.json", 'w') as f:
        json.dump(corpus_stats, f, indent=2)

    logging.info("Completed entropy analysis")
    logging.info(corpus_stats)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()