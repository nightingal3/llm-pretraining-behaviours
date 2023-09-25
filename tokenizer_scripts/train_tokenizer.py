from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import argparse
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Digits

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
parser.add_argument('--vocab_size', type=int)
parser.add_argument('--min_frequency', type=int, default=2)
parser.add_argument('--output_dir')
parser.add_argument('--extra_tokens', type=int)
args = parser.parse_args()


paths = [str(x) for x in Path(args.data_path).glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])

special_tokens = ["<s>","<pad>","</s>","<unk>","<mask>",]
for i in range(args.extra_tokens):
    special_tokens.append('<extra_token_'+str(i)+'>')

# Customize training
tokenizer.train(files=paths, vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=special_tokens)

# Save files to disk
tokenizer.save(args.output_dir+'/tokenizer.json')
