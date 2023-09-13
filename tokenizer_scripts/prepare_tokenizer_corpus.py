import argparse
import json
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

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
parser.add_argument('--max_words', type=int)
parser.add_argument('--languages')
parser.add_argument('--output_dir')
args = parser.parse_args()

def _close_when_exhausted(file: TextIO) -> Iterable[str]:
    with file:
        for line in file:
            yield json.loads(line)

def open_read_cleaned(filename) -> Iterable[str]:
    file: TextIO = gzip.open(filename, "rt")  # type: ignore
    return _close_when_exhausted(file)

languages = args.languages.split(',')

for language in languages:
    corpus = open_read_cleaned(args.data_path+'/'+language+'/0000.json.gz')
    
    n_words=0
    text=[]
    for doc in corpus:
        text.append(doc['text'])
        n_words += len(doc['text'].split(' '))

        if n_words>=args.max_words:
            with open(args.output_dir+'/'+language+'.txt', 'w') as f:
                f.write('\n'.join(text))
            break

