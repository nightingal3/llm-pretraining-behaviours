import json
from datasets import load_dataset
import numpy as np
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)
import gzip

def _close_when_exhausted(file: TextIO) -> Iterable[str]:
    with file:
        for line in file:
            yield json.loads(line)

def open_read_cleaned(filename) -> Iterable[str]:
    file: TextIO = gzip.open(filename, "rt")  # type: ignore
    return _close_when_exhausted(file)


TO_REMOVE = [
    'meta', 'perplexity_score', 'text_length', 'url', 'domain', 'dup_ratio', 'pairs', 'repetitions', 'included_in_dedup', 'cluster', 'id'
]

def dump_hf_dataset(dataset_name, output_file, filtering, max_tokens, dataset_path, split='train'):
    # to load clean data file coming from data team
    #
    # corpus = open_read_cleaned(dataset_paths[idx])
    # for doc in corpus:
    #     perplexity=doc['perplexity']


    #dataset = load_dataset(dataset_name, split=split)
    dataset = datasets.load_from_disk(dataset_path)
    
    print('50 percentile filtering')
    p_50 = np.percentile(dataset["perplexity_score"], 50)
    dataset = dataset.filter(lambda x: x["perplexity_score"] < p_50, num_proc=128)

    n_words=0
    for idx in range(len(dataset)):
        n_words += len(dataset[int(idx)]['text'].split(' '))

        if n_words>=max_tokens:
            break
    
    print('n words', n_words)
    print('n docs', idx+1)
        dataset = dataset.select(range(idx+1))

    # Remove columns if they exist
    for column in TO_REMOVE:
        if column in dataset.column_names:
            dataset = dataset.remove_columns(column)
    
    dataset.to_json(output_file, lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--filter', type=bool, required=False, default=False)
    parser.add_argument('--n_tokens', type=int, required=False, default=None)
    args = parser.parse_args()
    dump_hf_dataset(args.dataset_name, args.output, False, args.n_tokens, args.dataset_path)
