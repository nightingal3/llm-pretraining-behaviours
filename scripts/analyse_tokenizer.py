import os
from tokenizers.implementations import ByteLevelBPETokenizer
import argparse
from tokenizers import pre_tokenizers, Tokenizer
from tokenizers.pre_tokenizers import Whitespace, Digits
import nltk
import transformers
import json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer_dir')
parser.add_argument('--eval_sets', required=True, nargs='+')
parser.add_argument('--baseline_tokenizers', nargs='+', default=[
    'togethercomputer/LLaMA-2-7B-32K', 'mistralai/Mistral-7B-v0.1'
])
args = parser.parse_args()

tokenizer = Tokenizer.from_file(args.tokenizer_dir+'/tokenizer.json')

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
    baseline_tokenized = [[] for _ in baseline_tokenizers]
    
    with open(eval_set, 'r') as f:
        original = f.readlines()
    
    for line in original:
        line.strip()
        tokens = tokenizer.encode(line).tokens
        baseline_tokens = [
            tokenizer.convert_ids_to_tokens(tokenizer.encode(line)) for tokenizer in baseline_tokenizers
        ]
        tokenized.append(tokens)
        for i in range(len(baseline_tokenizers)):
            baseline_tokenized[i].append(baseline_tokens[i])

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
    baseline_counts = [compute_stats(baseline_seg) for baseline_seg in baseline_segs]
    pieces = seg_counts["toks"]

    words = raw_counts["toks"]
    sents = raw_counts["nseqs"]
    nc = raw_counts["chars"]
    nb = raw_counts["bytes"]

    sents_per_context = 2048 * sents / pieces

    results_json["ours"][eval_set] = {
        "pieces/word": pieces / words,
        "pieces/sentence": pieces / sents,
        "chars/piece": nc / pieces,
        "bytes/piece": nb / pieces,
        "sentences/context": sents_per_context,
    }
    for i in range(len(baseline_tokenizers)):
        baseline_pieces = baseline_counts[i]["toks"]
        baseline_words = baseline_counts[i]["toks"]
        baseline_sents = baseline_counts[i]["nseqs"]
        baseline_nc = baseline_counts[i]["chars"]
        baseline_nb = baseline_counts[i]["bytes"]

        baseline_sents_per_context = 2048 * baseline_sents / baseline_pieces
        results_json[args.baseline_tokenizers[i]][eval_set] = {
            "pieces/word": baseline_pieces / baseline_words,
            "pieces/sentence": baseline_pieces / baseline_sents,
            "chars/piece": baseline_nc / baseline_pieces,
            "bytes/piece": baseline_nb / baseline_pieces,
            "sentences/context": baseline_sents_per_context,
        }

with open('analysis_results', 'w') as f:
    json.dump(results_json, f, indent=2)