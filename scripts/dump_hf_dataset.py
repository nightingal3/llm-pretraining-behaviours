import json
from datasets import load_dataset
import numpy as np

TO_REMOVE = [
    'meta', 'perplexity_score', 'text_length', 'url', 'domain', 'dup_ratio', 'pairs', 'repetitions', 'included_in_dedup', 'cluster', 'id'
]

def dump_hf_dataset(dataset_name, output_file, filtering, max_tokens, split='train'):
    dataset = load_dataset(dataset_name, split=split)

    if filtering:
        print('filtering')
        sorted_perp = np.argsort(dataset["perplexity_score"], axis=0)
        
        n_tokens=0
        filtered_idxs=[]
        for idx in sorted_perp:
            filtered_idxs.append(int(idx))
            n_tokens += dataset[int(idx)]['text_length']

            if n_tokens>=max_tokens:
                break

        dataset = dataset.select(filtered_idxs)

    # Remove columns if they exist
    for column in TO_REMOVE:
        if column in dataset.column_names:
            dataset = dataset.remove_columns(column)
    
    dataset.to_json(output_file, lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--filter', type=bool, required=False, default=False)
    parser.add_argument('--n_tokens', type=int, required=False, default=None)
    args = parser.parse_args()
    dump_hf_dataset(args.dataset_name, args.output, args.filter, args.n_tokens)
