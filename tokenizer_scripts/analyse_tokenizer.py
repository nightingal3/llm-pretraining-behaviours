from tokenizers.implementations import ByteLevelBPETokenizer
import argparse
from tokenizers import pre_tokenizers, Tokenizer
from tokenizers.pre_tokenizers import Whitespace, Digits
import nltk
import transformers

parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer_dir')
parser.add_argument('--dataset')
parser.add_argument('--languages')
args = parser.parse_args()

languages = args.languages.split(',')

#tokenizer = Tokenizer.from_file(args.tokenizer_dir+'/tokenizer.json')
#tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])

tokenizer_llama = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path = args.tokenizer_dir+'/tokenizer.json',
        use_fast=False,
        )

tokenizer_llama = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        use_fast=False,
        )

tokenizer_bloom = transformers.AutoTokenizer.from_pretrained(
        "bigscience/tokenizer",
        use_fast=False,
        )

data = dict()
tokenized_data = dict()
tokenized_data_llama = dict()
tokenized_data_bloom = dict()

for language in languages:
    non_tokenized = []
    tokenized = []
    tokenized_llama = []
    tokenized_bloom = []
    
    with open(args.dataset+'/'+language, 'r') as f:
        original = f.readlines()
    
    for line in original:
        line.strip()
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(line))
        tokens_llama = tokenizer_llama.convert_ids_to_tokens(tokenizer_llama.encode(line))
        tokens_bloom = tokenizer_bloom.convert_ids_to_tokens(tokenizer_bloom.encode(line))
        tokenized.append(tokens)
        #non_tokenized.append(line.split())
        non_tokenized.append(nltk.word_tokenize(line))

        tokenized_llama.append(tokens_llama)
        tokenized_bloom.append(tokens_bloom)

    
    data[language] = non_tokenized
    tokenized_data[language] = tokenized
    tokenized_data_llama[language] = tokenized_llama
    tokenized_data_bloom[language] = tokenized_bloom


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
results_llama=[]
results_bloom=[]
for lang in languages:
    raw = data[lang]
    seg = tokenized_data[lang]
    seg_llama = tokenized_data_llama[lang]
    seg_bloom = tokenized_data_bloom[lang]

    raw_counts = compute_stats(raw)
    seg_counts = compute_stats(seg)
    seg_counts_llama = compute_stats(seg_llama)
    seg_counts_bloom = compute_stats(seg_bloom)
    pieces = seg_counts["toks"]
    pieces_llama = seg_counts_llama["toks"]
    pieces_bloom = seg_counts_bloom["toks"]
    words = raw_counts["toks"]
    sents = raw_counts["nseqs"]
    nc = raw_counts["chars"]
    nb = raw_counts["bytes"]

    sents_per_context = 2048 * sents / pieces

    result = "model: ours \t lang: {}\tpieces/word: {:.2f}\tpieces/sentence: {:.2f}\tchars/piece: {:.2f}\tbytes/piece: {:.2f}\tsentences/context: {:.2f}\t\tntokens: {}\tnpieces: {}".format(lang, pieces / words, pieces / sents, nc / pieces, nb / pieces, sents_per_context, words, pieces)
    result_llama = "model: llama \t lang: {}\tpieces/word: {:.2f}\tpieces/sentence: {:.2f}\tchars/piece: {:.2f}\tbytes/piece: {:.2f}\tsentences/context: {:.2f}\t\tntokens: {}\tnpieces: {}".format(lang, pieces_llama / words, pieces_llama / sents, nc / pieces_llama, nb / pieces_llama, sents_per_context, words, pieces_llama)
    result_bloom = "model: bloom \t lang: {}\tpieces/word: {:.2f}\tpieces/sentence: {:.2f}\tchars/piece: {:.2f}\tbytes/piece: {:.2f}\tsentences/context: {:.2f}\t\tntokens: {}\tnpieces: {}".format(lang, pieces_bloom / words, pieces_bloom / sents, nc / pieces_bloom, nb / pieces_bloom, sents_per_context, words, pieces_bloom)

    print(result)
    print(result_llama)
    print(result_bloom)
    results.append(result)
    results_llama.append(result_llama)
    results_bloom.append(result_bloom)

with open('analysis_results', 'w') as f:
    f.write('\n'.join(results))

with open('analysis_results_llama', 'w') as f:
    f.write('\n'.join(results_llama))

with open('analysis_results_bloom', 'w') as f:
    f.write('\n'.join(results_bloom))
