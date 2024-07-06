import pandas as pd
import json
import argparse

from collect_features_tocsv import search_for_file

model_cols = [
    "dimension",
    "num_heads",
    "num_layers",
    "mlp_ratio",
    "positional_embedding",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update json files with gathered data")
    parser.add_argument("--gathered_data", type=str, help="Path to the annotated data")
    args = parser.parse_args()

    gathered_data = pd.read_csv(args.gathered_data)

    models_to_update = gathered_data[model_cols].tolist()

    for model in models_to_update:
        model_str = "_".join([str(x) for x in model])
        model_file = search_for_file(model_str)
        with open(model_file, "r") as f:
            model_data = json.load(f)

        model_data["performance"] = gathered_data[
            gathered_data[model_cols].eq(model).all(axis=1)
        ]["performance"].values[0]

        with open(model_file, "w") as f:
            json.dump(model_data, f, indent=4)
