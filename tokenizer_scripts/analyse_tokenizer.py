from tokenizers.implementations import ByteLevelBPETokenizer
import argparse
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Digits

parser = argparse.ArgumentParser()
parser.add_argument('--vocab')
parser.add_argument('--merges')
parser.add_argument('--datasets', nargs='+')
parser.add_argument('--languages', nargs='+')
args = parser.parse_args()

tokenizer = ByteLevelBPETokenizer(args.vocab, args.merges)
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])

data = dict()
tokenized_data = dict()

for dataset_idx in range(len(args.datasets)):
    non_tokenized = []
    tokenized = []
    
    with open(args.datasets[dataset_idx], 'r') as f:
        original = f.readlines()
    
    for line in original:
        line.strip()
        line = pre_tokenizer.pre_tokenize_str(line)
        line = '10'
        tokens = tokenizer.encode(line).tokens
        tokenized.append(tokens)
        non_tokenized.append(line.split())
    
    data[args.languages[dataset_idx]] = non_tokenized
    tokenized_data[args.languages[dataset_idx]] = tokenized


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


for lang in args.languages:
    raw = data[lang]
    seg = tokenized_data[lang]

    raw_counts = compute_stats(raw)
    seg_counts = compute_stats(seg)
    pieces = seg_counts["toks"]
    words = raw_counts["toks"]
    sents = raw_counts["nseqs"]
    nc = raw_counts["chars"]
    nb = raw_counts["bytes"]

    sents_per_context = 2048 * sents / pieces

    print("lang: {}\tpieces/word: {:.2f}\tpieces/sentence: {:.2f}\tchars/piece: {:.2f}\tbytes/piece: {:.2f}\tsentences/context: {:.2f}\t\tntokens: {}\tnpieces: {}".format(lang, pieces / words, pieces / sents, nc / pieces, nb / pieces, sents_per_context, words, pieces))



