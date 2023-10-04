import os
import json
import datasets
import numpy as np
from typing import (
    List,
    Optional,
)
import gzip

def get_hf_dataset(
    dataset_name: str, 
    path: str=None,
    dirs: List[str]=None, 
    split: str=None,
    stream: bool=False
):
    if path is not None:
        assert splits is None, "Cannot specify both dataset_path and splits"
        dataset = datasets.load_from_disk(dataset_path)
    else:
        # HACK: need this to support partial load of stream datasets
        data_files = [os.path.join(d, "**") for d in dirs]
        # TODO: fix this hard-coded train (sub)split
        dataset = datasets.load_dataset(
            dataset_name, 
            data_files=data_files,
            streaming=stream)[split]
        
    return dataset


def filter_hf_dataset(
    dataset: datasets.Dataset,
    max_tokens: Optional[int]=None, 
    percentile: Optional[int]=50
):
    assert max_tokens is not None or percentile is not None, "Must specify either max_tokens or percentile"

    if percentile is not None:
        ppl_perc = np.percentile(dataset["perplexity_score"], percentile)
        dataset = dataset.filter(lambda x: x["perplexity_score"] < ppl_perc, num_proc=128)

    if max_tokens is not None:
        n_words=0
        for idx in range(len(dataset)):
            n_words += len(dataset[int(idx)]['text'].split(' '))

            if n_words>=max_tokens:
                break
    
        print('n words', n_words)
        print('n docs', idx+1)
        dataset = dataset.select(range(idx+1))
    
    return dataset

COLUMNS_TO_REMOVE = [
    'meta', 
    'perplexity_score', 
    'text_length', 
    'url', 
    'domain', 
    'dup_ratio', 
    'pairs', 
    'repetitions', 
    'included_in_dedup', 
    'cluster', 
    'id'
]

def dump_hf_dataset(
    dataset: datasets.Dataset,
    output_file: str,
    text_only: bool=False
):
    # Remove columns if they exist
    existing_columns = dataset.column_names
    if existing_columns is not None:
        for column in COLUMNS_TO_REMOVE:
            if column in existing_columns:
                dataset = dataset.remove_columns(column)
    
    # dataset.to_json(output_file, lines=True)
    # due to stream, print each line to file
    with open(output_file, 'w') as f:
        for i, example in enumerate(dataset):
            if i % 1000 == 0:
                print("Saved {} examples".format(i))
            if text_only:
                print(example['text'], file=f)
            else:
                print(json.dumps(example), file=f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=False, default=None)
    parser.add_argument('--dataset_dirs', type=str, required=False, nargs='+', default=None)
    parser.add_argument('--dataset_split', type=str, required=False, default="train")
    parser.add_argument('--filter', default=False, action='store_true')
    parser.add_argument('--percentile', type=int, required=False, default=50)
    parser.add_argument('--n_tokens', type=int, required=False, default=None)
    parser.add_argument('--stream', default=False, action='store_true')
    parser.add_argument('--text-only', default=False, action='store_true')
    args = parser.parse_args()
    
    dataset = get_hf_dataset(
        dataset_name=args.dataset_name, 
        path=args.dataset_path, 
        dirs=args.dataset_dirs,
        split=args.dataset_split,
        stream=args.stream
    )
    if args.filter:
        datset = filter_hf_dataset(
            dataset, 
            percentile=args.percentile,
            max_tokens=args.n_tokens
        )

    dump_hf_dataset(dataset, args.output, text_only=args.text_only)

