import os
import argparse
import logging

source_dir = "/data/tir/projects/tir7/user_data/mchen5/dolma_100B"
target_dir = "/data/tir/projects/tir7/user_data/mchen5/dolma_100B_pruned_names"
domains = [
    "c4",
    "common-crawl",
    "peS2o",
    "stack-code",
    "gutenberg-books",
    "wiki-en-simple"
]
MIN_BYTES = 1_000_000

parts = {
    "c4" : 4213499,
    "common-crawl" : 510983,
    "gutenberg-books" : 1178,
    "peS2o" : 20803,
    "stack-code" : 103818,
    "wiki-en-simple" : 2785999,
}

def prune_domain(domain: str):
    logging.info(f"Pruning {domain}")

    total_bytes = 0
    bytes_copied = 0

    source_path = f"{source_dir}/{domain}"
    with open(f"{target_dir}/{domain}", "a") as file:
        for part in range(1, parts[domain] + 1):
            filename = f"{source_path}/part_{part}.arrow"

            num_bytes = os.path.getsize(filename)
            if num_bytes >= MIN_BYTES:
                file.write(f"{filename}\n")
                bytes_copied += num_bytes
            total_bytes += num_bytes

            if part % 5000 == 0:
                logging.info(f"Pruned up to part {part}; copied {bytes_copied:.4e} / {total_bytes:.4e} bytes")
    logging.info(f"Finished pruning; copied {bytes_copied:.4e} / {total_bytes:.4e} bytes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        type=str,
        choices=[
            "peS2o",
            "common-crawl",
            "stack-code",
            "wiki-en-simple",
            "c4",
            "gutenberg-books",
        ],
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prune_domain(args.domain)
