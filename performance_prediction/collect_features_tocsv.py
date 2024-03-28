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


def normalize_features(row):
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
        input_file = os.path.join(input_dir, input_file)
        output_file = os.path.join(output_dir, f"training_{type_selection}_final.csv")
        # Load the JSON file into a dictionary
        with open(input_file, "r") as file:
            json_data = json.load(file)

        extracted_features = extract_features_from_json(json_data, features_to_extract)
        # Display the extracted features
        if type_selection == "score":
            if extracted_features[dataset] or dataset == "all":
                try:
                    n_shots_selection = list(extracted_features[dataset].keys())
                    if n_shots == -1:
                        # take any number of shots
                        extracted_features[dataset] = extracted_features[dataset][
                            n_shots_selection[0]
                        ]
                    else:
                        try:
                            extracted_features[dataset] = extracted_features[dataset][
                                f"{n_shots}-shot"
                            ]
                        except KeyError:
                            print(
                                f"{input_file}: {n_shots}-shot results not found in the dataset"
                            )
                            continue
                except:
                    print(f"{input_file}: {dataset} not found in the dataset")
                    continue

        extracted_features_all.append(extracted_features)

    df = pd.DataFrame(extracted_features_all)
    if type_selection == "score":
        features_df = df.drop(columns=["model_name"]).apply(
            normalize_features, axis=1, result_type="expand"
        )
        final_df = pd.concat([df[["model_name"]], features_df], axis=1)
    else:
        final_df = df

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
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
    )  # TODO - all is not yet working

    args = parser.parse_args()
    main(args.input_json, args.output_dir, args.config_file, args.type, args.dataset)
