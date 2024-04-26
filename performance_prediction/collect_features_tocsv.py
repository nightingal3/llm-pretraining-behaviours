import argparse
import os
import json
import pandas as pd
from collections import defaultdict
from typing import Any


def search_for_file(filename: str, data_type: str = "model") -> str:
    """
    Search for a file in the metadata directory. Returns the proper path of the file if found.

    Args:
        filename (str): The name of the file to search for.
        data_type (str): The type of data to search for. Either "model" or "dataset".
    Returns:
        str: The path of the file if found. Else ""
    """
    if data_type == "model":
        data_dir = "./metadata/model_metadata"
    elif data_type == "dataset":
        data_dir = "./metadata/dataset_metadata"
    else:
        raise ValueError("data_type must be either 'model' or 'dataset'")

    full_path = os.path.join(data_dir, filename)
    if not full_path.endswith(".json"):
        full_path += ".json"

    if os.path.exists(full_path) and os.path.isfile(full_path):
        return full_path

    for file in os.listdir(data_dir):
        curr_path = os.path.join(data_dir, file)
        if f"{filename}.json" == file:
            return curr_path
    return ""


def get_dataset_info(filename: str) -> dict:
    """Returns json data from a dataset file."""
    with open(filename, "r") as file:
        return json.load(file)


def combine_stage_info(
    stage_info: list, property: str = "summary:total_size_tokens_billions"
) -> Any:
    """
    Combine the properties (currently just total tokens) from each stage of the training process.
    Returns stage_info with the additional properties total_<property> for each property.
    Args:
        stage_info (list): The list of stages to combine.
        property (str): The property to combine.

    Returns:
        Any: The combined property.
    """
    total = 0
    for key, value in stage_info.items():
        if property in key:
            total += value

    stage_info[f"total_{property}"] = total
    return stage_info


def extract_features_from_json(json_data: dict, features: list) -> dict:
    """
    Extract features from JSON. Features should be specified like this in the config:
    top_level:level1:level2:feature where : represents the hierarchy of the JSON.

    Args:
        json_data (dict): The JSON data to extract features from.
        features (list): The list of features to extract.
    """
    extracted_features = {}
    for feature in features:
        feature_path = feature.split(":")
        feature_value = json_data
        for path in feature_path:
            if isinstance(feature_value, dict):
                feature_value = feature_value.get(path, None)
            else:
                feature_value = None
                break  # Stop iteration if feature_value is not a dictionary
        extracted_features[feature] = feature_value

    return extracted_features


def extract_data_from_training_stages(model_data: dict, features: list):
    extracted_data = defaultdict(int)

    for stage in model_data.get("training_stages", []):
        dataset_file = search_for_file(stage["data"], "dataset")
        if dataset_file != "":
            dataset_info = get_dataset_info(dataset_file)
            extracted_features = extract_features_from_json(dataset_info, features)
            for feature, value in extracted_features.items():
                extracted_data[f"{stage['name']}_{feature}"] += value

    return extracted_data


def extract_features_from_json_dataset(json_model_data: dict, features: list) -> dict:
    """
    Extract features from JSON. Follows pointers from a model's training data to corresponding files in
    metadata/dataset_metadata. (TODO: this may be kind of awkward, but we currently have the dataset information kind of stored in both places.)
    However, the config should list features as they're formatted in the dataset configs.

    Args:
        json_data (dict): The JSON data to extract features from.
        features (list): The list of features to extract.
    """

    all_stages_info = defaultdict(int)

    # add data from a base model if specified
    if "base_model" in json_model_data:
        base_model_file = search_for_file(json_model_data["base_model"], "model")
        if base_model_file != "":
            base_model_data = get_dataset_info(base_model_file)
            base_data = extract_data_from_training_stages(base_model_data, features)
            all_stages_info.update(base_data)

    # add data from the current model
    curr_data = extract_data_from_training_stages(json_model_data, features)
    all_stages_info.update(curr_data)

    all_stages_info = combine_stage_info(
        all_stages_info,
    )

    return all_stages_info


def normalize_features(row: pd.Series):
    """
    Puts features from scores into a standardized format, since they are dicts inside the dataset's column.
    Ignores columns that are not dicts.

    Args:
        row (pd.Series): The row to normalize.

    Returns:
        dict: The normalized features.

    Example:
        Given a row with the column "drop_3_shot", which has a dict {"acc": 0.5, "acc_stderr": 0.1},
        the output will be {"drop_3_shot_acc": 0.5, "drop_3_shot_acc_stderr": 0.1}.
    """
    normalized_features = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                normalized_features[f"{key}_{subkey}"] = subvalue
        else:
            normalized_features[key] = value

    return normalized_features


def main(
    input_dir: str,
    output_dir: str,
    config_path: str,
    type_selection: str,
    dataset: str = "all",
    n_shots: int = -1,
):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    features_to_extract = config[type_selection]

    extracted_features_all = []
    for input_file in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, input_file)

        with open(input_file_path, "r") as file:
            json_data = json.load(file)

        # the processing is pretty different between scores and model/dataset features, so separating them
        if type_selection == "score":
            if dataset == "all":
                for _, ds_value in json_data.get("results", {}).items():
                    extracted_features = process_dataset_scores(
                        input_file, ds_value, n_shots
                    )
                    if extracted_features:
                        extracted_features["model_name"] = json_data["model_name"]
                        extracted_features_all.append(extracted_features)
            else:
                ds_value = json_data.get("results", {}).get(dataset, {})
                extracted_features = process_dataset_scores(
                    input_file, ds_value, n_shots
                )
                if extracted_features:
                    extracted_features["model_name"] = json_data["model_name"]
                    extracted_features_all.append(extracted_features)
        elif type_selection == "model":
            extracted_features = extract_features_from_json(
                json_data, features_to_extract
            )
            extracted_features_all.append(extracted_features)
        else:
            extracted_features = extract_features_from_json_dataset(
                json_data, features_to_extract
            )
            if len(extracted_features) > 0:
                extracted_features

            extracted_features_all.append(extracted_features)

    df = pd.DataFrame(extracted_features_all)

    if type_selection == "score":
        features_df = df.drop(columns=["model_name"]).apply(
            normalize_features, axis=1, result_type="expand"
        )
        final_df = pd.concat([df[["model_name"]], features_df], axis=1)
    else:
        final_df = df

    output_file = os.path.join(output_dir, f"training_{type_selection}_final.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)
    print(f"Features extracted and saved to {output_file}")


def process_dataset_scores(input_file: str, ds_value: dict, n_shots: int = -1):
    """
    Processes dataset performance results, handling n-shot scenarios.

    Args:
        input_file (str): The name of the input file.
        ds_value (dict): The model's performance results across datasets.
        n_shots (int): The number of shots to use as the model's final result. -1 will attempt to grab all shots available.

    Returns:
        dict: The extracted features.

    """
    extracted_features = {}

    for dataset in ds_value:
        try:
            if n_shots == -1:
                avail_shots = list(ds_value[dataset].keys())
                for n_s in avail_shots:
                    n_shots_tmp = int(n_s.split("-")[0])
                    new_key = f"{dataset}_{n_shots_tmp}-shot"
                    extracted_features[new_key] = ds_value[dataset][
                        f"{n_shots_tmp}-shot"
                    ]
            else:
                new_key = f"{dataset}_{n_shots}-shot"
                extracted_features[new_key] = ds_value[dataset][f"{n_shots}-shot"]
        except KeyError:
            print(f"{input_file}: {n_shots}-shot results not found in the dataset")
            continue
        except:
            print(f"{input_file}: {dataset} not found in the dataset")
            continue

    return extracted_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="input directory for the feature to be collected. For 'dataset' features, please also pass the same directory as for model features.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./config.json",
        help="Path to the configuration file.",
    ),
    parser.add_argument(
        "--type",
        type=str,
        choices=["score", "model", "dataset"],
        default="score",
        help="collect model or score or dataset features",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="dataset to collect results for. Pass 'all' to collect all datasets. Only relevant for the 'score' type.",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        default=-1,
        help="number of shots to collect results for. -1 will attempt to grab all shots available.",
    )
    args = parser.parse_args()
    main(
        args.input_dir,
        args.output_dir,
        args.config_file,
        args.type,
        args.dataset,
        args.n_shots,
    )
