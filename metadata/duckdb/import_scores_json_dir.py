#!/usr/bin/env python3

import os
import glob
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional

from model_metadata_db import AnalysisStore

BENCHMARK_DEFAULTS = {
    "arc_challenge": ["25-shot"],
    "hellaswag": ["10-shot"],
    "mmlu": ["0-shot", "5-shot"],
    "truthfulqa": ["0-shot"],
    "winogrande": ["5-shot"],
    "lambada": ["0-shot"],
    "gsm8k": ["5-shot"],
    "arithmetic": ["5-shot"],
    "minerva": ["5-shot"],
    #'mathqa': ['5-shot'],
    #'xnli': ['0-shot'],
    #'anli': '0-shot',
    #'logiqa2': '0-shot',
    #'fld': '0-shot',
    #'asdiv': '5-shot'
}


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


def main():
    setup_logging()

    # Configuration
    DB_DIR = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb"
    SCORES_DIR = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/remaining_evals"

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

        # Get all JSON files in the directory
        json_files = glob.glob(os.path.join(SCORES_DIR, "*.json"))
        logging.info(f"Found {len(json_files)} JSON files to process")

        # Skip temp/backup files
        json_files = [f for f in json_files if "temp" not in f and "backup" not in f]

        # Import scores
        try:
            store.import_scores_from_json_dir(SCORES_DIR)
            logging.info(f"Successfully processed {len(json_files)} files")
        except Exception as e:
            logging.error(f"Error processing JSON files: {e}")
            raise

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
