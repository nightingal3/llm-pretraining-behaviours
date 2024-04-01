import argparse
import os
import json
import pandas as pd


def extract_features_from_json(json_data, features):
    """
    Extracts specified features from a nested JSON data structure.

    :param json_data: The JSON data as a dictionary.
    :param features: List of strings representing the keys of the features to extract.
    :return: Dictionary with the extracted features and their values.
    """
    extracted_features = {}

    def extract(data, keys):
        """
        Recursive helper function to search and extract values for specified keys.

        :param data: Current level of the JSON data.
        :param keys: Remaining keys to search for.
        """
        if not keys or not isinstance(data, dict):
            return
        for key, value in data.items():
            if key in keys:
                extracted_features[key] = value
            extract(value, keys)

    extract(json_data, features)
    for feature in features:
        if feature not in extracted_features:
            print(feature, " was not found so adding default value of -1!")
            extracted_features[feature] = (
                -1
            )  # using -1 as default val since a model could score 0 for real

    extracted_features = {
        key: extracted_features[key] for key in sorted(extracted_features)
    }

    return extracted_features


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
    breakpoint()
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
        else:
            extracted_features = extract_features_from_json(
                json_data, features_to_extract
            )
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
        choices=["score", "model"],
        default="score",
        help="collect model or score or dataset features",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="dataset to collect results for. Pass 'all' to collect all datasets",
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
