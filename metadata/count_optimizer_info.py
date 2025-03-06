import os
import json

# Directory containing your JSON files
directory = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/model_metadata"

# Counter for files meeting the criteria
count = 0

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r") as file:
                data = json.load(file)
                # Check if 'training_stages' exists and is a list
                if "training_stages" in data and isinstance(
                    data["training_stages"], list
                ):
                    # Iterate over each stage
                    for stage in data["training_stages"]:
                        # Check for 'pretraining' stage with 'optimizer' key
                        if stage.get("name") == "pretraining" and "optimizer" in stage:
                            count += 1
                            break  # No need to check other stages in this file
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error processing {filepath}: {e}")

print(f"Number of files with 'optimizer' in 'pretraining' stage: {count}")
