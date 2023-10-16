import argparse
import json
from typing import Iterable, TextIO
import gzip
import datasets

print("Running prepare_tokenizer_corpus.py", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--data_paths', type=str, nargs='+')
parser.add_argument('--words_per_source', type=int, nargs='+')
parser.add_argument('--data_type', choices=['jsonl', 'hf', 'gzip'], default='jsonl')
parser.add_argument('--output_dir')
args = parser.parse_args()

if len(args.words_per_source) > 1 and len(args.words_per_source) != len(args.data_paths):
    raise ValueError("`words_per_source` must be either a single value or a list of the same length as `data_paths`")

if len(args.words_per_source) == 1:
    args.words_per_source = args.words_per_source[0] * len(args.data_paths)

def _close_when_exhausted(file: TextIO) -> Iterable[str]:
    with file:
        for line in file:
            yield json.loads(line)

def open_read_cleaned(filename, is_gzip=False) -> Iterable[str]:
    file: TextIO = gzip.open(filename, "rt")  if is_gzip else open(filename, "r")
    return _close_when_exhausted(file)

for i, (data_path, max_words) in enumerate(zip(args.data_paths, args.words_per_source), 1):
    print(f"Processing {data_path}...", flush=True)
    # open the json file 
    if args.data_type == 'jsonl' or args.data_type == 'gzip':
        corpus = open_read_cleaned(data_path, is_gzip=args.data_type == 'gzip')
    elif args.data_type == 'hf':
        corpus = datasets.load_from_disk(data_path)
        
    n_words=0
    text=[]
    with open(f"{args.output_dir}.{i}.txt", 'w') as f:
        for j, doc in enumerate(corpus, 1):
            if j % 10000 == 0:
                print(f"Prepared {j} documents", flush=True)
            print(doc['text'], file=f)

        n_words += len(doc['text'].split(' '))
        if n_words>=max_words:
            break

