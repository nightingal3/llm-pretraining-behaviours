import os
import json
import datasets
import numpy as np
import pandas as pd
from typing import (
    List,
    Optional,
    TextIO,
    Iterable
)
import gzip
import random
from itertools import chain
import argparse


def _close_when_exhausted(file: TextIO) -> Iterable[str]:
        with file:
            for line in file:
                yield json.loads(line)

def open_read_cleaned(filename) -> Iterable[str]:
    file: TextIO = gzip.open(filename, "rt")  # type: ignore
    return _close_when_exhausted(file)

def get_hf_dataset(
    dataset_name: str, 
    path: str=None,
    dirs: List[str]=None, 
    split: str='train',
    stream: bool=False,
    shuffle: bool=False,
    shards: int=1000,
    code: bool=False
):
    if path is not None:
        assert os.path.exists(path), f"Path does not exist, {path}"
        assert dirs is None, "Cannot specify both path and dirs"
        dataset = datasets.load_from_disk(path)
    else:
        # HACK: need this to support partial load of stream datasets
        if dirs is not None:
            data_files = [os.path.join(d, "**") for d in dirs]
        else:
            data_files = None
            
        dataset = datasets.load_dataset(
            dataset_name, 
            data_files=data_files,
            streaming=stream)
    if not code:
        dataset = dataset[split]
    if shuffle:
        print("Shuffling dataset")
        # we shard the dataset to speed up shuffling
        shards = [
            dataset.shard(shards, i, contiguous=True).shuffle()
            for i in range(shards)
        ]
        # shuffle the shards
        random.shuffle(shards)
        # concatenate the shards
        dataset = datasets.concatenate_datasets(shards)

    return dataset

def get_cleaned_dataset(
    directory: str=None,
):
    i=0
    for folder in os.listdir(directory):
        if "json" in folder:
            # not a folder
            path=directory+folder
        elif "perplexity" in folder:
            path=directory+folder+"/0000.json.gz"
        else:
            path=directory+folder+"/perplexity/0000.json.gz"
        assert os.path.exists(path), f"Path does not exist {path}"
        data = open_read_cleaned(path)
        if i==0:
            i+=1
            dataset=data
        else:
            dataset = chain(dataset,data)
        
    return dataset

def get_bilingual_dataset(
    directory: str=None,
    max_tokens: Optional[int]=None,
    max_tokens_test: Optional[int]=None,
):
    if max_tokens is not None:
        max_tokens = max_tokens/11
    if max_tokens_test is not None:
        max_tokens_test = max_tokens_test/11

    data = []
    data_test = []
    total_words=0
    for folder in os.listdir(directory):
        source_lang = folder.split('-')[0]
        target_lang = folder.split('-')[1]
        
        source_path=directory+folder+"/cometkiwi/threshold_0.85/cometiwi_data."+source_lang+'-'+target_lang+'.'+source_lang
        target_path=directory+folder+"/cometkiwi/threshold_0.85/cometiwi_data."+target_lang+'-'+source_lang+'.'+target_lang
        
        assert os.path.exists(source_path), "Source path does not exist"
        assert os.path.exists(target_path), "Target path does not exist"

        source_data = open_read_cleaned(source_path)
        target_data = open_read_cleaned(target_path)

        n_words_test=0
        n_docs_test=0
        if max_tokens_test is not None:
            for source_doc, target_doc in zip(source_data, target_data):
                n_docs_test+=1
                n_words += len(source_doc['text'].split(' ')) + len(target_doc['text'].split(' '))
                data_test.append(source_doc['text'] + "</s>" + "<s>" + target_doc['text'])
                if n_words>=max_tokens_test:
                    break
        
        n_words=0
        n_docs=0
        for source_doc, target_doc in zip(source_data, target_data):
            if n_docs<=n_docs_test:
                n_docs+=1
            else:
                n_words += len(source_doc['text'].split(' ')) + len(target_doc['text'].split(' '))
                data.append({'text': source_doc['text'] + "</s>" + "<s>" + target_doc['text']})
                if max_tokens is not None:
                    if n_words>=max_tokens:
                        break
        print('n words', n_words)
        total_words += n_words
    
    print('total words', total_words)
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data))

    with open('test_data','w') as f:
        f.write('\n'.join(data_test))
    
    return dataset

def filter_hf_dataset(
    dataset: datasets.Dataset,
    max_tokens: Optional[int]=None,
    max_tokens_test: Optional[int]=None,
    max_samples: Optional[int]=None,
    percentile: Optional[int]=50
):
    assert (max_tokens is not None or 
            max_samples is not None or
            percentile is not None
    ), "Must specify either max_tokens or percentile"

    if percentile is not None:
        ppl_perc = np.percentile(dataset["perplexity_score"], percentile)
        dataset = dataset.filter(lambda x: x["perplexity_score"] < ppl_perc, num_proc=128)

    if max_tokens_test is not None:
        n_words_test=0
        n_docs_test=0
        data_test=[]
        for idx in range(len(dataset)):
            n_words_test += len(dataset[int(idx)]['text'].split(' '))
            data_test.append(dataset[int(idx)]['text'])
            if n_words_test>=max_tokens_test:
                n_docs_test=idx
                break
        with open('test_data','w') as f:
            f.write('\n'.join(data_test))

    if max_tokens is not None:
        n_words=0
        for idx in range(n_docs_test+1, len(dataset)):
            n_words += len(dataset[int(idx)]['text'].split(' '))
            if n_words>=max_tokens:
                break
    
        print('n words', n_words)
        print('n docs', idx+1)
        dataset = dataset.select(range(n_docs_test+1, idx+1))

    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    return dataset

def filter_code_dataset(
    dataset: datasets.Dataset,
    max_tokens: Optional[int]=None,
    max_tokens_test: Optional[int]=None,
    max_samples: Optional[int]=None,
    percentile: Optional[int]=50
):
    assert (max_tokens is not None or 
            max_samples is not None or
            percentile is not None
    ), "Must specify either max_tokens or percentile"

    if percentile is not None:
        stars=[]
        for doc in dataset:
            if doc['source'] not in ['git-commits', 'git-issues']:
                stars.append(doc['max_stars_count'])
            if len(stars)>1000000:
                break
        threshold = np.median(np.array(stars))
        print('threshold', threshold)

    if max_tokens_test is not None:
        n_words_test=0
        n_docs_test=0
        data_test=[]
        for idx in range(len(dataset)):
            if dataset[idx]['source'] in ['git-commits', 'git-issues'] or dataset[idx]['max_stars_count']>=threshold:
                n_words_test += len(dataset[int(idx)]['content'].split(' '))
                data_test.append(dataset[int(idx)]['content'])
                if n_words_test>=max_tokens_test:
                    n_docs_test=idx
                    break
        with open('test_data','w') as f:
            f.write('\n'.join(data_test))

    if max_tokens is not None:
        n_words=0
        dataset_idxs=[]
        for idx in range(n_docs_test+1, len(dataset)):
            if dataset[idx]['source'] in ['git-commits', 'git-issues'] or dataset[idx]['max_stars_count']>=threshold:
                dataset_idxs.append(idx)
                n_words += len(dataset[int(idx)]['content'].split(' '))
                if n_words>=max_tokens:
                    break
    
        print('n words', n_words)
        print('n docs', len(dataset_idxs))
        dataset = dataset.select(dataset_idxs)

    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    return dataset

def filter_cleaned_dataset(
    dataset: datasets.Dataset,
    max_tokens: Optional[int]=None,
    max_tokens_test: Optional[int]=None,
    percentile: Optional[int]=50
):
    assert max_tokens is not None or percentile is not None, "Must specify either max_tokens or percentile"

    if percentile is not None:
        perplexities=[]
        for doc in dataset:
            perplexities.append(doc['perplexity'])
            if len(perplexities)>1000000:
                break
                
        threshold = np.percentile(np.array(perplexities), percentile)
        print('threshold', threshold)

        if max_tokens_test is not None:
            data_test = []
            n_docs_test=0
            n_words_test=0
            for doc in dataset:
                n_docs_test+=1
                if doc['perplexity'] < threshold:
                    n_words_test += len(doc['text'].split(' '))
                    data_test.append(doc['text'])
                    if n_words_test>=max_tokens_test:
                        break
            with open('test_data','w') as f:
                f.write('\n'.join(data_test))

        data = []
        n_words=0
        n_docs=0
        for doc in dataset:
            if n_docs<=n_docs_test:
                n_docs+=1
            else:
                if doc['perplexity'] < threshold:
                    n_words += len(doc['text'].split(' '))
                    data.append({'text': doc['text']})
                    if max_tokens is not None:
                        if n_words>=max_tokens:
                            break
    
        print('n words', n_words)

        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    
    else:

        if max_tokens_test is not None:
            n_words_test=0
            n_docs_test=0
            data_test = []
            for doc in dataset:
                n_docs_test+=1
                n_words_test += len(doc['text'].split(' '))
                data_test.append(doc['text'])
                if n_words_test>=max_tokens_test:
                    break
            with open('test_data','w') as f:
                f.write('\n'.join(data_test))

        if max_tokens is not None:
            n_words=0
            data = []
            n_docs=0
            for doc in dataset:
                if n_docs<=n_docs_test:
                    n_docs+=1
                else:
                    n_words += len(doc['text'].split(' '))
                    data.append({'text': doc['text']})
                    if n_words>=max_tokens:
                        break
        
            print('n words', n_words)
            dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data))

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
        for i, example in enumerate(dataset, 1):
            if i % 1000 == 0:
                print("Saved {} examples".format(i))
            if text_only:
                print(example['text'], file=f)
            else:
                print(json.dumps(example), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=False, default=None)
    parser.add_argument('--dataset_dirs', type=str, required=False, nargs='+', default=None)
    parser.add_argument('--dataset_split', type=str, required=False, default="train")
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--filter', default=False, action='store_true')
    parser.add_argument('--percentile', type=int, required=False, default=None)
    parser.add_argument('--n_tokens', type=int, required=False, default=None)
    parser.add_argument('--n_tokens_test', type=int, required=False, default=None)
    parser.add_argument('--n_docs', type=int, required=False, default=None)
    parser.add_argument('--stream', default=False, action='store_true')
    parser.add_argument('--text-only', default=False, action='store_true')
    parser.add_argument('--hf_dataset', default=False, action='store_true')
    parser.add_argument('--bilingual', default=False, action='store_true')
    parser.add_argument('--code', default=False, action='store_true')
    args = parser.parse_args()
    

    if args.hf_dataset:
        dataset = get_hf_dataset(
            dataset_name=args.dataset_name, 
            path=args.dataset_path, 
            dirs=args.dataset_dirs,
            split=args.dataset_split,
            stream=args.stream,
            shuffle=args.shuffle,
            code=args.code
            )
        if args.filter:
            if args.code:
                dataset = filter_code_dataset(
                    dataset, 
                    percentile=args.percentile,
                    max_tokens=args.n_tokens,
                    max_tokens_test=args.n_tokens_test,
                    max_samples=args.n_docs
                )
            else:
                dataset = filter_hf_dataset(
                    dataset, 
                    percentile=args.percentile,
                    max_tokens=args.n_tokens,
                    max_tokens_test=args.n_tokens_test,
                    max_samples=args.n_docs
                )

    else:
        if args.bilingual:
            dataset = get_bilingual_dataset(
                directory=args.dataset_path,
                max_tokens=args.max_tokens,
                max_tokens_test=args.max_tokens_test
            )
        else:
            dataset = get_cleaned_dataset(
                directory=args.dataset_path,
            )

            if args.filter:
                dataset = filter_cleaned_dataset(
                    dataset, 
                    percentile=args.percentile,
                    max_tokens=args.n_tokens,
                    max_tokens_test=args.n_tokens_test
                )

    dump_hf_dataset(dataset, args.output, text_only=args.text_only)