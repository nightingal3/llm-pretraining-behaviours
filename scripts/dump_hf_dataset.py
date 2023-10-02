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
    dataset_path: str=None, 
    splits: List[str]=None
):
    if dataset_path is not None:
        assert splits is None, "Cannot specify both dataset_path and splits"
        dataset = datasets.load_from_disk(dataset_path)
    else:
        split = "+".join(splits) if splits is not None else None
        dataset = datasets.load_dataset(dataset_name, split, streaming=True)['train']
        
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
):
    # Remove columns if they exist
    existing_columns = dataset.column_names
    for column in COLUMNS_TO_REMOVE:
        if column in existing_columns:
            dataset = dataset.remove_columns(column)
    
    # dataset.to_json(output_file, lines=True)
    # due to stream, print each line to file
    with open(args.output_file, 'w') as f:
        for i, example in enumerate(dataset):
            if i % 1000 == 0:
                print("Saved {} examples".format(i))
            print(json.dumps(example), file=f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=False, default=None)
    parser.add_argument('--splits', type=str, required=False, nargs='+', default=None)
    parser.add_argument('--filter', default=False, action='store_true')
    parser.add_argument('--percentile', type=int, required=False, default=50)
    parser.add_argument('--n_tokens', type=int, required=False, default=None)
    args = parser.parse_args()
    
    dataset = get_hf_dataset(
        dataset_name=args.dataset_name, 
        dataset_path=args.dataset_path, 
        splits=args.splits
    )
    if args.filter:
        datset = filter_hf_dataset(
            dataset, 
            percentile=args.percentile,
            max_tokens=args.n_tokens
        )

