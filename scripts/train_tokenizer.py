from typing import Optional
import datasets
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import argparse
from pathlib import Path


def build_tokenizer(
    replacement: str = "▁",
    add_prefix_space: bool = True,
    dropout: Optional[float] = None,
    fuse_unk: Optional[bool] = True,
):
    """
    Build a tokenizer.
    :return: The tokenizer.
    """
    tokenizer = Tokenizer(
        models.BPE(
            dropout=dropout, unk_token=None, fuse_unk=fuse_unk, byte_fallback=True
        )
    )
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Metaspace(
                replacement=replacement, add_prefix_space=add_prefix_space
            ),
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.Punctuation(),
            # to deal with "......" or "-----" tokens (which might be interesting to have actually)
        ]
    )

    tokenizer.decoder = decoders.Sequence(
        [
            decoders.ByteFallback(),
            decoders.Metaspace(
                replacement=replacement, add_prefix_space=add_prefix_space
            ),
            decoders.Fuse(),
            decoders.Strip(content=" ", left=1, right=0),
        ]
    )
    return tokenizer


def fit_tokenizer(
    tokenizer, paths, extra_tokens=None, vocab_size=None, min_frequency=None
):
    """
    Fit a tokenizer on a dataset.
    :param tokenizer: The tokenizer to fit.
    :param dataset: The dataset to fit the tokenizer on.
    :return: The fitted tokenizer.
    """

    special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]
    special_tokens += [f"<extra_id_{i}>" for i in range(extra_tokens)]
    bpe_trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=special_tokens,
        limit_alphabet=1000,
        initial_alphabet=[],
    )

    tokenizer.train(files=paths, trainer=bpe_trainer)
    return tokenizer


def refit_tokenizer(
    tokenizer, paths, extra_tokens=None, vocab_size=None, min_frequency=None
):
    """
    Fit a tokenizer on a dataset.
    :param tokenizer: The tokenizer to fit.
    :param dataset: The dataset to fit the tokenizer on.
    :return: The fitted tokenizer.
    """

    def batch_iterator(batch_size=50):
        for data_path in paths:
            with open(data_path, "r") as f:
                lines = f.readlines()
                for i in range(0, len(lines), batch_size):
                    yield lines[i : i + batch_size]

    new_special_tokens = ["<pad>"]
    new_special_tokens += [f"<extra_id_{i}>" for i in range(extra_tokens)]
    it = batch_iterator()

    tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.Punctuation(),
        ]
    )

    tokenizer = tokenizer.train_new_from_iterator(
        it,
        vocab_size=vocab_size,
        new_special_tokens=new_special_tokens,
    )

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--output_dir")
    parser.add_argument("--extra_tokens", type=int)
    args = parser.parse_args()

    paths = [str(x) for x in Path(args.data_path).glob("**/*.txt")]
    print(paths)

    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tok = refit_tokenizer(
        tok,
        paths,
        extra_tokens=args.extra_tokens,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    tok.save_pretrained(args.output_dir)

    example_sentence = "This is a test sentence. On va voir comment elle est gérée .... 123 + 56 = 2567. Let's go!"
    encoded = tok.encode(example_sentence)
    print(tok.tokenize(example_sentence))
    decoded = tok.decode(encoded)
    print(decoded)

    # tok = build_tokenizer()
    # tok.save_pretrained(args.output_dir+'/tower_llm_tokenizer')
    # tok.save(args.output_dir+"/tokenizer.json")
