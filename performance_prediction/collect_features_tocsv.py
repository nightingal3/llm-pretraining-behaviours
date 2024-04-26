import argparse
import os
import json
import pandas as pd
from collections import defaultdict


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


def extract_features_from_json_dataset(json_model_data: dict, features: list) -> dict:
    """
    Extract features from JSON. Follows pointers from a model's training data to corresponding files in
    metadata/dataset_metadata. (TODO: this may be kind of awkward, but we currently have the dataset information kind of stored in both places.)
    However, the config should list features as they're formatted in the dataset configs.

    Args:
        json_data (dict): The JSON data to extract features from.
        features (list): The list of features to extract.
    """
    if "training_stages" not in json_model_data:
        # Dataset not documented
        return {"id": json_model_data["id"]}

    dataset_files = {
        stage["name"]: stage["data"] for stage in json_model_data["training_stages"]
    }
    all_stages_info = defaultdict(int)

    # other features not in config - may revisit
    all_stages_info["id"] = json_model_data["id"]
    all_stages_info["is_instruction_tuned"] = "instruction" in dataset_files

    for stage_name, dataset_file in dataset_files.items():
        if not (os.path.exists(dataset_file) and os.path.isfile(dataset_file)):
            # search for the dataset within the metadata/dataset_metadata directory
            dataset_file = os.path.join(
                "./metadata/dataset_metadata",
                f"{dataset_file}.json",
            )
            if not os.path.exists(dataset_file):
                # Dataset not found
                return {}

        with open(dataset_file, "r") as file:
            dataset_json = json.load(file)

        extracted_data_features = extract_features_from_json(dataset_json, features)

        # TODO: we need some handling of the inherited/nested features. This format will only really work for total tokens/similar
        for feature, value in extracted_data_features.items():
            all_stages_info[f"total_{feature}"] += value

        for feature, value in extracted_data_features.items():
            modified_key = f"{stage_name}_{feature}"
            all_stages_info[modified_key] = value

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
