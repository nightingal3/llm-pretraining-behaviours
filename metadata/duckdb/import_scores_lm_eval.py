#!/usr/bin/env python3

import os
import glob
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, List

from model_metadata_db import AnalysisStore

our_model_paths = {
    "llama2_460M_nl_code_shuf-hf": "/data/tir/projects/tir5/users/mengyan3/dolma_checkpts/llama2_460M_nl_code_shuf-hf",
    "llama2_220M_nl_only_shuf-hf": "/data/tir/projects/tir5/users/mengyan3/dolma_checkpts/llama2_220M_nl_only_shuf-hf/",
    "llama2_220M_nl_code_shuf-hf": "/data/tir/projects/tir5/users/mengyan3/dolma_checkpts/llama2_220M_nl_code_shuf-hf/",
    "llama2_220M_nl_40_code_60": "/data/tir/projects/tir5/users/mengyan3/dolma_checkpts_hf_final/llama2_220M_nl_40_code_60/",
    "llama2_220M_nl_20_code_80": "/data/tir/projects/tir5/users/mengyan3/dolma_checkpts_hf_final/llama2_220M_nl_20_code_80/",
    "llama2_220M_nl_0_code_100": "/data/tir/projects/tir5/users/mengyan3/dolma_checkpts_hf_final/llama2_220M_nl_0_code_100/",
}
# reverse the dictionary
our_model_paths = {v: k for k, v in our_model_paths.items()}


def setup_logging():
    """Configure logging with timestamp and formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_latest_duckdb(db_dir: str) -> Optional[str]:
    """Find the most recent duckdb file in the specified directory"""
    duckdb_files = glob.glob(os.path.join(db_dir, "*.duckdb"))
    if not duckdb_files:
        return None

    return max(duckdb_files, key=os.path.getctime)


def import_lmeval_scores(
    store: AnalysisStore, base_dir: str, excludes: List[str] = []
) -> int:
    """
    Import scores from lm-eval output files

    Args:
        store: AnalysisStore instance
        base_dir: Directory containing lm-eval output files
        excludes: List of benchmark names to exclude

    Returns:
        Number of successfully imported files
    """
    json_files = glob.glob(os.path.join(base_dir, "**/*.json"), recursive=True)
    logging.info(f"Found {len(json_files)} JSON files to process")

    total_imported = 0

    for json_path in json_files:
        # Skip temp/backup files
        if "temp" in json_path or "backup" in json_path:
            continue

        try:
            logging.info(f"Processing: {json_path}")
            store.import_scores_from_lm_eval_json(
                json_path, excludes=excludes, alternate_name_map=our_model_paths
            )
            total_imported += 1
            logging.info(f"Successfully imported scores from {json_path}")
        except Exception as e:
            logging.error(f"Error importing {json_path}: {e}")

    return total_imported


def main():
    setup_logging()

    # Configuration
    benchmarks = [
        "arc_challenge",
        "arithmetic",
        "gsm8k",
        "hellaswag",
        "lambada",
        "mmlu",
        "truthfulqa",
        "winogrande",
        "humaneval",
        "xnli",  # actually contains all brier scores
    ]
    DB_DIR = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb"
    EXCLUDES = []  # Add any benchmark names to exclude

    excludes_per_benchmark = {
        "arc_challenge": [],
        "arithmetic": [],
        "gsm8k": [],
        "hellaswag": [],
        "lambada": [],
        "mmlu": [
            "mmlu_other",
            "mmlu_social_sciences",
            "mmlu_stem",
            "mmlu_humanities",
            "mmlu",
        ],
        "truthfulqa": [],
        "winogrande": [],
        "xnli": ["xnli", "reasoning-mcq"],
    }

    for benchmark in benchmarks:
        logging.info(f"Processing benchmark: {benchmark}")
        LMEVAL_DIR = (
            f"/data/tir/projects/tir5/users/mengyan3/lm_eval_outputs_3_1/{benchmark}"
        )

        # Find latest database
        latest_db = get_latest_duckdb(DB_DIR)
        if not latest_db:
            logging.error(f"No DuckDB files found in {DB_DIR}")
            return

        logging.info(f"Using latest database: {latest_db}")

        try:
            # Load the store
            store = AnalysisStore.from_existing(latest_db)
            logging.info("Successfully loaded analysis store")

            # Import scores
            total_imported = import_lmeval_scores(
                store, LMEVAL_DIR, excludes=excludes_per_benchmark.get(benchmark, [])
            )
            logging.info(f"Successfully imported scores from {total_imported} files")

            # Save with today's date
            date_str = datetime.now().strftime("%Y_%m_%d")
            output_path = os.path.join(DB_DIR, f"{date_str}.duckdb")

            logging.info(f"Saving updated database to: {output_path}")
            store.save_database(output_path)
            logging.info("Database saved successfully")

        except Exception as e:
            logging.error(f"Fatal error: {e}")
            raise


if __name__ == "__main__":
    main()
