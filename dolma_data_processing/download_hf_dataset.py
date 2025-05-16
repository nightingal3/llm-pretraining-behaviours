import os
import argparse
from datasets import load_dataset
import json
from tqdm import tqdm

def download_dataset(dataset_name, output_dir, subset=None, batch_size=1000):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset in streaming mode
    if subset:
        dataset = load_dataset(dataset_name, subset, split="train", streaming=True)
    else:
        try:
            dataset = load_dataset(dataset_name, split="train", streaming=True)
        except:
            dataset = load_dataset(dataset_name, streaming=True)

    # Initialize counters and batch
    total_examples = 0
    batch = []
    file_index = 0

    # Create a progress bar
    pbar = tqdm(desc=f"Downloading {dataset_name}", unit=" examples")
    print_interval = 100
    steps_until_print = 100

    for example in dataset:
        batch.append(example)
        steps_until_print -= 1
        if steps_until_print == 0:
            pbar.update(print_interval)
            steps_until_print = print_interval
        if len(batch) >= batch_size:
            # Write the batch to a file
            print(f"Writing part {file_index:05d}")
            file_path = os.path.join(output_dir, f"part_{file_index:05d}.jsonl")
            with open(file_path, 'w') as f:
                for item in batch:
                    json.dump(item, f)
                    f.write('\n')
            
            # Update counters and progress bar
            total_examples += len(batch)
            
            # Clear the batch and increment file index
            batch = []
            file_index += 1

    # Write any remaining examples
    if batch:
        file_path = os.path.join(output_dir, f"part_{file_index:05d}.jsonl")
        with open(file_path, 'w') as f:
            for item in batch:
                json.dump(item, f)
                f.write('\n')
        total_examples += len(batch)
        pbar.update(len(batch))

    pbar.close()
    print(f"Downloaded {total_examples} examples to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset to disk")
    parser.add_argument("dataset_name", help="Name of the Hugging Face dataset to download")
    parser.add_argument("output_dir", help="Directory to save the downloaded dataset")
    parser.add_argument("--subset", help="subset of the dataset (not train/test/etc, but a subset of data if available)")
    parser.add_argument("--batch_size", type=int, default=1_000_000, help="Number of examples to write per file")
    args = parser.parse_args()

    download_dataset(args.dataset_name, args.output_dir, args.subset, args.batch_size)