import argparse
import os
import json
import pandas as pd
from collections import defaultdict
from typing import Any

def search_for_file(filename: str, data_type: str = "model") -> str:
    data_dir = f"./metadata/{data_type}_metadata"
    filename = filename.replace("/", "_") + ".json"
    full_path = os.path.join(data_dir, filename)
    return full_path if os.path.exists(full_path) else ""

def get_dataset_info(filename: str) -> dict:
    with open(filename, "r") as file:
        return json.load(file)

def extract_features_from_json(json_data: dict, features: list) -> dict:
    extracted_features = {}
    for feature in features:
        feature_value = json_data
        for path in feature.split(":"):
            feature_value = feature_value.get(path, None)
            if feature_value is None:
                break
        extracted_features[feature] = feature_value
    return extracted_features

def extract_data_from_training_stages(model_data: dict, features: list) -> dict:
    extracted_data = defaultdict(int)
    for stage in model_data.get("training_stages", []):
        if stage["name"] == "pretraining" and 'total_tokens_billions' in stage:
            extracted_data['total_tokens_billions'] = stage['total_tokens_billions']
        else:
            dataset_file = search_for_file(stage["data"], "dataset")
            if dataset_file:
                dataset_info = get_dataset_info(dataset_file)
                extracted_features = extract_features_from_json(dataset_info, features)
                for feature, value in extracted_features.items():
                    if value is not None:
                        extracted_data[feature] += value
    return dict(extracted_data)

def combine_stage_info(stage_info: dict, property: str = "total_tokens_billions") -> dict:
    return {'total_tokens_billions': stage_info.get(property, 0)}

def extract_features_from_json_dataset(json_model_data: dict, features: list) -> dict:
    all_stages_info = defaultdict(int)
    all_stages_info["id"] = json_model_data["id"]
    if "base_model" in json_model_data:
        base_model_file = search_for_file(json_model_data["base_model"], "model")
        if base_model_file:
            base_model_data = get_dataset_info(base_model_file)
            base_data = extract_data_from_training_stages(base_model_data, features)
            all_stages_info.update(base_data)
    curr_data = extract_data_from_training_stages(json_model_data, features)
    all_stages_info.update(curr_data)
    return combine_stage_info(all_stages_info)

def main(input_dir: str, output_dir: str, config_file: str, type_selection: str):
    with open(config_file, "r") as file:
        config = json.load(file)
    features_to_extract = config[type_selection]

    extracted_features_all = []
    for input_file in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, input_file)
        if os.path.isfile(input_file_path):
            try:
                with open(input_file_path, "r") as file:
                    json_data = json.load(file)
                extracted_features = extract_features_from_json_dataset(json_data, features_to_extract)
                extracted_features_all.append(extracted_features)
            except Exception as e:
                print(f"Error processing {input_file_path}: {e}")

    df = pd.DataFrame(extracted_features_all)
    output_file = os.path.join(output_dir, f"training_{type_selection}_final.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Features extracted and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing JSON files to process.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory where the results CSV will be saved.")
    parser.add_argument("--config_file", type=str, required=True, help="JSON configuration file specifying which features to extract.")
    parser.add_argument("--type", type=str, choices=["model", "dataset"], default="model", help="Type of features to extract (model or dataset).")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.config_file, args.type)
