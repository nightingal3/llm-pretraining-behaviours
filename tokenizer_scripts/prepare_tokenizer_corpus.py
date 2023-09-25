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
import datasets


parser = argparse.ArgumentParser()
parser.add_argument('--data_path_de')
parser.add_argument('--data_path_en')
parser.add_argument('--data_path_es')
parser.add_argument('--data_path_fr')
parser.add_argument('--data_path_it')
parser.add_argument('--data_path_ko')
parser.add_argument('--data_path_nl')
parser.add_argument('--data_path_pl')
parser.add_argument('--data_path_pt')
parser.add_argument('--data_path_ru')
parser.add_argument('--data_path_sv')
parser.add_argument('--data_path_zh')
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

for data_path in [args.data_path_de, args.data_path_en, args.data_path_es, args.data_path_fr, args.data_path_it, args.data_path_ko,
                    args.data_path_nl, args.data_path_pl, args.data_path_pt, args.data_path_ru, args.data_path_sv, args.data_path_zh]:

    language = data_path.split('/')[5]
    
    if language in languages:
        if language=='en':
            corpus = datasets.load_from_disk(data_path)
        else:
            corpus = open_read_cleaned(data_path)
        
        n_words=0
        text=[]
        for doc in corpus:
            text.append(doc['text'])
            n_words += len(doc['text'].split(' '))

            if n_words>=args.max_words:
                with open(args.output_dir+'/'+language+'.txt', 'w') as f:
                    f.write('\n'.join(text))
                break

