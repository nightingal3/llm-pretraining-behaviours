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
            print(feature, " was not found so adding default value of 0!")
            extracted_features[feature] = 0

    extracted_features = {key: extracted_features[key] for key in sorted(extracted_features)}


    return extracted_features


def main(input_dir, output_dir):
    # List of features to extract
    features_to_extract = [
    "model_name", "hellaswag"
    ]
    for input_file in os.listdir(input_dir):

        input_file = os.path.join(input_dir, input_file)
        output_file = os.path.join(output_dir, "training_scores.csv")
        # Load the JSON file into a dictionary
        with open(input_file, 'r') as file:
            json_data = json.load(file)

        extracted_features = extract_features_from_json(json_data, features_to_extract)
        # Display the extracted features
        if extracted_features['hellaswag']:
            extracted_features['hellaswag'] = extracted_features['hellaswag']['10-shot']['acc']


        df = pd.DataFrame([extracted_features])
        file_exists = os.path.exists(output_file)

        df.to_csv(output_file, mode='a', index=False, header=not file_exists)
    



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



    args = parser.parse_args()
    main(args.input_json, args.output_dir)