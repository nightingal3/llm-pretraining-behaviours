from tokenizers.implementations import ByteLevelBPETokenizer
import argparse
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Digits

parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer_dir')
parser.add_argument('--dataset')
parser.add_argument('--languages')
args = parser.parse_args()

languages = args.languages.split(',')

tokenizer = ByteLevelBPETokenizer(args.tokenizer_dir+'/vocab.json', args.tokenizer_dir+'/merges.txt')
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])

data = dict()
tokenized_data = dict()

languages.append('en')
languages.append('fr')
for language in languages:
    non_tokenized = []
    tokenized = []
    
    with open(args.dataset+'/'+language, 'r') as f:
        original = f.readlines()
    
    for line in original:
        line.strip()
        tokens = tokenizer.encode(line).tokens
        tokenized.append(tokens)
        non_tokenized.append(line.split())
    
    data[language] = non_tokenized
    tokenized_data[language] = tokenized


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


results=[]
for lang in languages:
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

    result = "lang: {}\tpieces/word: {:.2f}\tpieces/sentence: {:.2f}\tchars/piece: {:.2f}\tbytes/piece: {:.2f}\tsentences/context: {:.2f}\t\tntokens: {}\tnpieces: {}".format(lang, pieces / words, pieces / sents, nc / pieces, nb / pieces, sents_per_context, words, pieces)
    print(result)
    results.append(result)

with open('analysis_results', 'w') as f:
    f.write('\n'.join(results))


