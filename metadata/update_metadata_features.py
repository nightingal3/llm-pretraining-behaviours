import argparse
import json
import os
import pandas as pd


def main(feature: str, domain: str, feature_dir: str, metadata_file: str):
    metadata = json.load(metadata_file)
    for filename in os.listdir(feature_dir):
        f = os.path.join(feature_dir, filename)
        if f.endswith(".parquet"):
            # for simple features that we want to aggregate, pandas should work fine
            feature_df = pd.read_parquet(f)
            feature_mean = feature_df[feature].mean()
            feature_std = feature_df[feature].std()
            metadata["features"][f"{feature}_mean"] = feature_mean
            metadata["features"][f"{feature}_std"] = feature_std
        else:
            continue
    updated_metadata = json.dumps(metadata, indent=4)
    with open(metadata_file, "w") as file:
        file.write(updated_metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", help="Name of feature", type=str, required=True)
    parser.add_argument(
        "--domain", help="Name of dataset domain", type=str, required=True
    )
    parser.add_argument(
        "--feature_dir",
        help="Directory where dataset domain features are saved",
        type=str,
        required=True,
    )
    parser.add_argument("--metadata_file", help="Dataset metadata file", type=str)

    args = parser.parse_args()

    main(args.feature_dir, args.metadata_file)
