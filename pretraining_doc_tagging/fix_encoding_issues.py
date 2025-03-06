import os
import re
import glob
import json
import ftfy
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import jsonlines

# Configuration
BASE_DIR = "/data/tir/projects/tir5/users/mengyan3/freegens_all"
CORRECTED_DIR = "/data/tir/projects/tir5/users/mengyan3/freegens_all_corrected"
ALLOWED_MODELS_CSV = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/pretraining_doc_tagging/ground_truth_dataset_info.csv"

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_allowed_models():
    """Load model IDs from CSV with validation"""
    try:
        df = pd.read_csv(ALLOWED_MODELS_CSV)
        return set(df['id'].dropna().values)
    except Exception as e:
        logger.error(f"Error loading allowed models: {str(e)}")
        return set()

def get_latest_jsonl(model_path, model_org, model_name):
    """Robust latest JSONL finder with error handling"""
    try:
        model_subdir = f"{model_org}__{model_name}"
        filtered_dir = Path(model_path) / model_subdir / "filtered"
        
        if not filtered_dir.exists():
            return None

        jsonl_files = list(filtered_dir.glob("filtered_*.[jJ][sS][oO][nN][lL]*"))
        jsonl_files = [f for f in jsonl_files if "results" not in f.name]
        
        if not jsonl_files:
            return None

        # Extract timestamp safely
        def get_ts(f):
            match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+', f.name)
            return match.group() if match else ""
            
        return max(jsonl_files, key=lambda x: get_ts(x))
    
    except Exception as e:
        logger.error(f"Error finding JSONL in {model_path}: {str(e)}")
        return None

def fix_text_fields(data):
    """Recursively fix all text fields in JSON data using ftfy"""
    if isinstance(data, dict):
        return {k: fix_text_fields(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [fix_text_fields(item) for item in data]
    elif isinstance(data, str):
        return ftfy.fix_text(data, normalization='NFKC').replace('\x00', '')
    return data

def process_jsonl(input_path, output_path):
    """Process a single JSONL file with comprehensive fixes"""
    fixed_entries = 0
    entries_with_errors = 0
    try:
        with jsonlines.open(input_path) as reader, \
             jsonlines.open(output_path, mode='w') as writer:

            for entry in tqdm(reader, desc=f"Processing {Path(input_path).name}"):
                try:
                    fixed_entry = fix_text_fields(entry)
                    writer.write(fixed_entry)
                    fixed_entries += 1
                except Exception as e:
                    logger.error(f"Failed to process entry: {str(e)}")
                    writer.write(entry)  # Preserve original on failure
                    entries_with_errors += 1
            print(f"Fixed entries: {fixed_entries}, Entries with errors: {entries_with_errors}")
            return True
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {str(e)}")
        return False

def process_models():
    """Main processing loop with enhanced error handling"""
    allowed_models = load_allowed_models()
    logger.info(f"Loaded {len(allowed_models)} allowed models")
    
    processed_count = 0
    error_count = 0

    # Walk through base directory
    for org_dir in os.listdir(BASE_DIR):
        org_path = Path(BASE_DIR) / org_dir
        if not org_path.is_dir():
            continue

        logger.info(f"Processing organization: {org_dir}")
        
        for model_dir in org_path.iterdir():
            if not model_dir.is_dir():
                continue

            model_id = f"{org_dir}/{model_dir.name}"
            
            logger.info(f"Processing model: {model_id}")
            
            # Get input file
            input_file = get_latest_jsonl(model_dir, org_dir, model_dir.name)
            if not input_file or not input_file.exists():
                logger.warning(f"No JSONL found for {model_id}")
                continue

            # Create output path
            rel_path = model_dir.relative_to(BASE_DIR)
            output_dir = Path(CORRECTED_DIR) / rel_path / "filtered"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / input_file.name

            # Process file
            success = process_jsonl(input_file, output_file)
            
            if success:
                processed_count += 1
                logger.info(f"Successfully processed {model_id}")
            else:
                error_count += 1
                logger.error(f"Failed to process {model_id}")

    logger.info(f"\nProcessing complete\nTotal processed: {processed_count}\nErrors: {error_count}")

if __name__ == "__main__":
    process_models()