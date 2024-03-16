import pyarrow.parquet as pq
from pathlib import Path
import pandas as pd
import sys
import os
import multiprocessing
import logging
import argparse


def decontaminate(df: pd.DataFrame, janitor) -> (pd.DataFrame, int):
    contamination_indices = 0

    def dedup(text: str) -> (str, int, bool):
        nonlocal contamination_indices

        cleaned_text = text
        thrown = False
        try:
            cleaned, num_contaminated = janitor.clean_python(text)
            if num_contaminated != 0:
                contamination_indices += num_contaminated
                cleaned_text = "".join(cleaned)
        except Exception as e:
            logging.error(f"{e}")
            thrown = True
        return (cleaned_text, num_contaminated, thrown)

    dedup_lambda = lambda text: dedup(text)
    df["text"], df["num_contaminated"], df["thrown"] = zip(
        *df["text"].map(dedup_lambda)
    )
    return (df, contamination_indices)


# Deduplicates the file at this path and saves the output to dolma_100B_deduped
def process_file(args):
    file_path, directory_name, file_name = args
    df: pd.DataFrame = pq.read_table(file_path).to_pandas()
    (df, contamination_indices) = decontaminate(df, janitor=janitor)
    df.to_parquet(f"{output_dir}/{directory_name}/{file_name}")
    return contamination_indices


def main():
    global output_dir
    global janitor
    sys.path.append(
        "/data/tir/projects/tir7/user_data/mchen5/llm-pretraining-behaviours/lm-evaluation-harness"
    )
    from lm_eval.decontamination.janitor import Janitor

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contaminant_path",
        help="Path to contaminant .txt file",
        type=str,
    )
    parser.add_argument(
        "--base_dir",
        help="Path to dataset base directory (should contain subdirectories for each domain)",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="Path to output base dir (should contain subdirectories for each domain)",
        type=str,
    )
    parser.add_argument(
        "--domain",
        help="Which domain to deduplicate (if not all)",
        type=str,
    )
    parser.add_argument(
        "--num_processes", help="Number of processes to run in parallel", type=int
    )
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    if not args.contaminant_path:
        raise ValueError("Please specify path to contaminant file")
    if not args.base_dir:
        raise ValueError("Please specify dataset base directory")
    if not args.output_dir:
        raise ValueError("Please specify output base directory")
    if args.domain and args.domain not in [
        "c4",
        "common-crawl",
        "gutenberg-books",
        "peS2o",
        "stack-code",
        "wiki-en-simple",
    ]:
        raise ValueError("Invalid domain")

    num_processes = args.num_processes if args.num_processes else 64
    logging.info(f"num_processes set to {num_processes}")

    # Make janitor, register contaminant
    with open(
        args.contaminant_path,
        "r",
    ) as file:
        contaminant: str = file.read()
    janitor = Janitor()
    janitor.register_contaminant(contaminant)
    logging.info("Created janitor, registered contaminant")
    base_dir = args.base_dir
    output_dir = args.output_dir
    process_inputs = []
    if not args.domain:
        for directory_name in os.listdir(base_dir):
            directory_path = os.path.join(base_dir, directory_name)
            if os.path.isdir(directory_path):
                for root, _, files in os.walk(directory_path):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        process_inputs.append((file_path, directory_name, file_name))
    else:
        directory_path = os.path.join(base_dir, domain)
        if os.path.isdir(directory_path):
            for root, _, files in os.walk(os.path.join(base_dir, domain)):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    process_inputs.append((file_path, directory_name, file_name))
    pool = multiprocessing.Pool(num_processes)
    contamination_indices_list = pool.map(process_file, process_inputs)
    logging.info("Finished decontamination")
    logging.info(f"{sum(contamination_indices_list)} total contamination indices.")


if __name__ == "__main__":
    main()
