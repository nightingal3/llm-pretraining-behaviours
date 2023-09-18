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
args = parser.parse_args()


paths = [str(x) for x in Path(args.data_path).glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
#tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])

# Customize training
tokenizer.train(files=paths, vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_pretrained(args.output_dir+'/tokenizer.json')
