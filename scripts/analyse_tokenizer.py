import os
from tokenizers.implementations import ByteLevelBPETokenizer
import argparse
from tokenizers import pre_tokenizers, Tokenizer
from tokenizers.pre_tokenizers import Whitespace, Digits
import nltk
import transformers
import json
from collections import defaultdict
import jieba

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer_dir")
parser.add_argument("--eval_sets", required=True, nargs="+")
parser.add_argument(
    "--baseline_tokenizers",
    nargs="+",
    default=[
        "togethercomputer/LLaMA-2-7B-32K",
        "mistralai/Mistral-7B-v0.1",
        "bigscience/tokenizer",
    ],
)
args = parser.parse_args()

tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_dir)
baseline_tokenizers = [
    transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    for tokenizer_name in args.baseline_tokenizers
]
data = dict()
tokenized_data = dict()
baselines_tokenized_data = [dict() for _ in baseline_tokenizers]

for eval_set in args.eval_sets:
    non_tokenized = []
    tokenized = []
    tokenized_v2 = []
    baseline_tokenized = [[] for _ in baseline_tokenizers]

    with open(eval_set, "r") as f:
        original = f.readlines()

    for line in original:
        line.strip()
        tokens_ = tokenizer.convert_ids_to_tokens(tokenizer_v2.encode(line))
        tokens = [item for item in tokens_ if item != "<s>"]

        baseline_tokens = [
            tokenizer.convert_ids_to_tokens(tokenizer.encode(line))
            for tokenizer in baseline_tokenizers
        ]
        tokenized.append(tokens)
        for i in range(len(baseline_tokenizers)):
            baseline_tokenized[i].append(baseline_tokens[i])

        if "zh" in eval_set:
            non_tokenized.append(jieba.lcut(line))
        else:
            non_tokenized.append(nltk.word_tokenize(line))

    set_name = os.path.basename(eval_set)
    data[set_name] = non_tokenized
    tokenized_data[set_name] = tokenized
    for i in range(len(baseline_tokenizers)):
        baselines_tokenized_data[i][set_name] = baseline_tokenized[i]


def compute_stats(text):
    # compute these excluding whitespace
    nchars = 0
    nbytes = 0
    ntoks = 0
    nseqs = 0

    for toks in text:
        ntoks += len(toks)
        nchars += sum(len(tok) for tok in toks)
        nbytes += sum(len(bytes(tok, "utf-8")) for tok in toks)
        nseqs += 1

    return {"toks": ntoks, "chars": nchars, "bytes": nbytes, "nseqs": nseqs}


results_json = defaultdict(dict)
for set_name in data:
    raw = data[set_name]
    seg = tokenized_data[set_name]
    baseline_segs = baselines_tokenized_data

    raw_counts = compute_stats(raw)
    seg_counts = compute_stats(seg)

    baseline_counts = [
        compute_stats(baseline_seg[set_name]) for baseline_seg in baseline_segs
    ]
    print("baseline", baseline_counts)
    pieces = seg_counts["toks"]

    words = raw_counts["toks"]
    sents = raw_counts["nseqs"]
    nc = raw_counts["chars"]
    nb = raw_counts["bytes"]

    sents_per_context = 2048 * sents / pieces

    results_json["ours"][set_name] = {
        "pieces/word": pieces / words,
        "pieces/sentence": pieces / sents,
        "chars/piece": nc / pieces,
        "bytes/piece": nb / pieces,
        "sentences/context": sents_per_context,
    }

    for i in range(len(baseline_tokenizers)):
        baseline_pieces = baseline_counts[i]["toks"]

        baseline_sents_per_context = 2048 * sents / baseline_pieces
        results_json[args.baseline_tokenizers[i]][set_name] = {
            "pieces/word": baseline_pieces / words,
            "pieces/sentence": baseline_pieces / sents,
            "chars/piece": nc / baseline_pieces,
            "bytes/piece": nb / baseline_pieces,
            "sentences/context": baseline_sents_per_context,
        }

with open("./analysis_results", "w") as f:
    json.dump(results_json, f, indent=2)
