import json
from datasets import load_dataset

TO_REMOVE = [
    'meta', 'perplexity_score', 'text_length', 'url', 'domain', 'dup_ratio', 'pairs', 'repetitions', 'included_in_dedup', 'cluster', 'id'
]

def dump_hf_dataset(dataset_name, output_file, split='train'):
    dataset = load_dataset(dataset_name, split=split)
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
    args = parser.parse_args()
    dump_hf_dataset(args.dataset_name, args.output)
