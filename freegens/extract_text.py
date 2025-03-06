import json
import os
from pathlib import Path


def process_jsonl_file(input_path, output_path):
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            try:
                data = json.loads(line.strip())
                if "resps" in data:
                    new_data = {"text": data["resps"][0][0]}
                    outfile.write(json.dumps(new_data) + "\n")
            except json.JSONDecodeError:
                continue


def process_directory(dir_path):
    base_path = Path(dir_path)
    for filepath in base_path.glob("**/*.jsonl"):
        output_path = filepath.parent / (filepath.stem + "_processed.jsonl")
        process_jsonl_file(filepath, output_path)


# Run the processor
process_directory(
    "/data/tir/projects/tir6/general/mengyan3/tower-llm-training/freegens/generations"
)
