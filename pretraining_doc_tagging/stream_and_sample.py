import argparse, random, json, time
from datasets import load_dataset
from pathlib import Path
import os

def reservoir_sample(stream, k, seed, log_every, max_docs=100_000_000):
    random.seed(seed)
    reservoir = []
    for i, rec in enumerate(stream, start=1):
        if max_docs and i > max_docs:
            print(f"→ reached max_docs={max_docs:,}")
            break
        if i % log_every == 0:
            print(f"[{time.strftime('%H:%M:%S')}] seen={i:,} reservoir={len(reservoir):,}")
        if i <= k:
            reservoir.append(rec)
        else:
            j = random.randint(1, i)
            if j <= k:
                reservoir[j-1] = rec
    return reservoir

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      required=True,
                   help="HF dataset ID, e.g. EleutherAI/pile")
    p.add_argument("--split",        default="train")
    p.add_argument("--subset_size",  type=int, required=True)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output",       required=True,
                   help="Where to write JSONL subset")
    p.add_argument("--log-every",    type=int,   default=100_000,
                   help="How often to print progress")
    p.add_argument("--max_docs", type=int)
    args = p.parse_args()

    print(f"→ streaming {args.dataset}:{args.split}, target={args.subset_size:,}")
    if "c4" in args.dataset:
        # C4 is a special case, it has a streaming split
        ds_stream = load_dataset(args.dataset, "en",
                                 split=args.split,
                                 streaming=True,
                                 trust_remote_code=True)
    elif "RedPajama" in args.dataset:
        ds_stream = load_dataset(args.dataset, "default",
                                 split=args.split,
                                 streaming=True,
                                 trust_remote_code=True)
    else:
        ds_stream = load_dataset(args.dataset,
                                split=args.split,
                                streaming=True, trust_remote_code=True)
    sampled = reservoir_sample(ds_stream,
                               args.subset_size,
                               args.seed,
                               args.log_every,
                               max_docs=args.max_docs)

    print(f"→ dumping {len(sampled):,} docs to {args.output}")

    # mkdir -p for args.output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for rec in sampled:
            f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()