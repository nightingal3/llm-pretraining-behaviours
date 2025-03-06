import json
from collections import defaultdict
from typing import List, Dict, Tuple
import re
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import os
import glob
from pathlib import Path
import csv

class RepetitionFilter:
    # [Previous RepetitionFilter class implementation remains the same]
    def __init__(
        self,
        duplicate_line_threshold: float = 0.8,
        duplicate_paragraph_threshold: float = 0.7,
        ngram_threshold: float = 0.6,
        ngram_sizes: List[int] = [2, 3],
        min_length: int = 200,
        max_consecutive_repeats: int = 50
    ):
        self.duplicate_line_threshold = duplicate_line_threshold
        self.duplicate_paragraph_threshold = duplicate_paragraph_threshold
        self.ngram_threshold = ngram_threshold
        self.ngram_sizes = ngram_sizes
        self.min_length = min_length
        self.max_consecutive_repeats = max_consecutive_repeats

    # [Rest of the RepetitionFilter methods remain the same]
    def has_pathological_repetition(self, text: str) -> Tuple[bool, str]:
        words = text.lower().split()
        consecutive_count = 1
        prev_word = None
        
        for word in words:
            if word == prev_word:
                consecutive_count += 1
                if consecutive_count > self.max_consecutive_repeats:
                    return True, f"Word '{word}' repeated {consecutive_count} times"
            else:
                consecutive_count = 1
            prev_word = word
            
        return False, ""

    def normalize_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    
    def get_paragraphs(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        return paragraphs
    
    def get_lines(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return lines

    def calculate_duplicate_fraction(self, items: List[str]) -> float:
        if not items:
            return 0.0
        unique_items = set(items)
        return 1.0 - (len(unique_items) / len(items))

    def calculate_ngram_repetition(self, text: str, n: int) -> float:
        words = word_tokenize(text.lower())
        if len(words) < n:
            return 0.0
            
        text_ngrams = list(ngrams(words, n))
        if not text_ngrams:
            return 0.0
            
        ngram_counts = defaultdict(int)
        for gram in text_ngrams:
            ngram_counts[gram] += 1
            
        repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)
        total_possible_repetitions = len(text_ngrams) - len(ngram_counts)
        
        if total_possible_repetitions == 0:
            return 0.0
            
        return repeated_ngrams / len(text_ngrams)

    def check_text(self, text: str) -> Tuple[bool, Dict]:
        text_length = len(text)
        if text_length < self.min_length:
            return False, {
                "reason": "too_short",
                "details": {
                    "length": text_length,
                    "min_required": self.min_length
                }
            }

        has_pathological, details = self.has_pathological_repetition(text)
        if has_pathological:
            return False, {
                "reason": "pathological_repetition",
                "details": {"pattern": details}
            }

        lines = self.get_lines(text)
        line_dup_fraction = self.calculate_duplicate_fraction(lines)
        if line_dup_fraction > self.duplicate_line_threshold:
            return False, {
                "reason": "extreme_line_repetition",
                "details": {"duplicate_line_fraction": line_dup_fraction}
            }

        paragraphs = self.get_paragraphs(text)
        para_dup_fraction = self.calculate_duplicate_fraction(paragraphs)
        if para_dup_fraction > self.duplicate_paragraph_threshold:
            return False, {
                "reason": "extreme_paragraph_repetition",
                "details": {"duplicate_paragraph_fraction": para_dup_fraction}
            }

        for n in self.ngram_sizes:
            ngram_rep_fraction = self.calculate_ngram_repetition(text, n)
            if ngram_rep_fraction > self.ngram_threshold:
                return False, {
                    "reason": "degenerate_repetition",
                    "details": {
                        "ngram_size": n,
                        "repetition_fraction": ngram_rep_fraction
                    }
                }

        return True, {"reason": "passed", "details": {
            "line_repetition": line_dup_fraction,
            "paragraph_repetition": para_dup_fraction
        }}

def process_directory(base_dir: str, filter_params: Dict = None):
    """
    Recursively process all samples_*.jsonl files in the directory structure
    and save filtered results to a /filtered subdirectory at each level.
    """
    # Initialize filter with default or custom parameters
    rep_filter = RepetitionFilter(**(filter_params or {}))
    
    # Find all samples_*.jsonl files
    for root, _, files in os.walk(base_dir):
        sample_files = [f for f in files if f.startswith('samples_') and f.endswith('.jsonl')]
        
        if not sample_files:
            continue
            
        # Create filtered directory at this level
        filtered_dir = os.path.join(root, 'filtered')
        os.makedirs(filtered_dir, exist_ok=True)
        
        for sample_file in sample_files:
            input_path = os.path.join(root, sample_file)
            if "llama2" not in input_path:
                continue
            
            # Create output paths in the filtered directory
            output_filtered = os.path.join(filtered_dir, f"filtered_{sample_file}")
            output_removed = os.path.join(filtered_dir, f"removed_{sample_file}")
            
            filtered_generations = []
            removed_generations = []
            
            print(f"\nProcessing: {input_path}")
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    for line in tqdm(f):
                        data = json.loads(line)
                        gen_text = data["resps"][0][0]
                        
                        # Apply filtering
                        passed, details = rep_filter.check_text(gen_text)
                        
                        if passed:
                            filtered_generations.append(data)
                        else:
                            removed_generations.append({
                                "reason": details["reason"],
                                "data": data,
                                "filter_details": details
                            })
                
                # Save results
                with open(output_filtered, "w", encoding="utf-8") as f:
                    for gen in filtered_generations:
                        f.write(json.dumps(gen) + "\n")
                        
                with open(output_removed, "w", encoding="utf-8") as f:
                    for gen in removed_generations:
                        f.write(json.dumps(gen) + "\n")
                
                print(f"Processed {sample_file}:")
                print(f"- Kept {len(filtered_generations)} generations out of {len(filtered_generations) + len(removed_generations)}")
                print(f"- Filtered generations saved to: {output_filtered}")
                print(f"- Removed generations saved to: {output_removed}")
                
            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")


def collect_random_samples(base_dir: str, n_samples: int = 250, seed: int = 42, output_file: str = None):
    """
    Collect random samples from filtered generations across all models.
    
    Args:
        base_dir: Base directory containing the filtered files
        n_samples: Target number of samples to collect
        seed: Random seed for reproducibility
    """
    import random
    random.seed(seed)
    
    # First, find all filtered files
    filtered_files = []
    for root, _, files in os.walk(base_dir):
        if 'filtered' in root:
            filtered_files.extend([
                os.path.join(root, f) for f in files 
                if f.startswith('filtered_') and f.endswith('.jsonl')
            ])
    
    if not filtered_files:
        print("No filtered files found!")
        return
    
    # Get model names from paths
    model_files = defaultdict(list)
    for file_path in filtered_files:
        # Extract model name from path
        parts = Path(file_path).parts
        try:
            org_idx = next(i for i, part in enumerate(parts) if 'filtered' in part) - 2
            model_name = f"{parts[org_idx]}/{parts[org_idx+1]}"
            model_files[model_name].append(file_path)
        except Exception as e:
            print(f"Skipping {file_path}: {str(e)}")
            continue
    
    # Calculate samples per model
    n_models = len(model_files)
    base_samples_per_model = n_samples // n_models
    extra_samples = n_samples % n_models
    
    print(f"\nCollecting approximately {n_samples} samples across {n_models} models")
    
    # Collect samples
    all_samples = []
    for i, (model_name, files) in enumerate(model_files.items()):
        samples_this_model = base_samples_per_model + (1 if i < extra_samples else 0)
        model_generations = []
        
        # Read all filtered generations for this model
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    model_generations.extend([json.loads(line) for line in f])
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                continue
        
        # Randomly sample from this model's generations
        if model_generations:
            n_available = len(model_generations)
            n_to_sample = min(samples_this_model, n_available)
            sampled = random.sample(model_generations, n_to_sample)
            
            # Add model information to each sample
            for sample in sampled:
                sample['source_model'] = model_name
                sample['source_file'] = file_path
            
            all_samples.extend(sampled)
            print(f"Collected {n_to_sample} samples from {model_name} (had {n_available} available)")
        else:
            print(f"No valid generations found for {model_name}")
    
    # Save collected samples to CSV
    output_file = os.path.join(base_dir, 'annotation_samples.csv') if output_file is None else output_file
    
    # Define CSV columns
    fieldnames = [
        'id',
        'text',
        'source_model',
        'domain',  # To be annotated
        'notes',   # Optional annotator notes
        'source_file'
    ]
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, sample in enumerate(all_samples, 1):
            writer.writerow({
                'id': f'sample_{i:04d}',
                'text': sample["resps"][0][0],
                'source_model': sample['source_model'],
                'domain': '',  # Empty column for annotation
                'notes': '',   # Empty column for annotator notes
                'source_file': sample['source_file']
            })
    
    print(f"\nCollected {len(all_samples)} total samples")
    print(f"Saved to: {output_file}")
    return all_samples

if __name__ == "__main__":
    base_directory = "/data/tir/projects/tir5/users/mengyan3/freegens_all/"
    
    # Optional: customize filter parameters
    filter_params = {
        "duplicate_line_threshold": 0.6,
        "duplicate_paragraph_threshold": 0.6,
        "ngram_threshold": 0.5,
        "ngram_sizes": [2, 3],
        "min_length": 50
    }
    
    process_directory(base_directory, filter_params)
    #collect_random_samples(base_directory, n_samples=300, output_file="./generation_domain_examples.csv")