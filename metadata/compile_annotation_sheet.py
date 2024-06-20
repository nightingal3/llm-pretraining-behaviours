import pandas as pd
import os
import argparse
import json

from performance_prediction.collect_features_tocsv import extract_features_from_json


def collect_relevant_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df[["image_path", "label"]]


def json_to_row(json_file: str) -> pd.DataFrame:
    with open(json_file, "r") as f:
        data = json.load(f)

    fields = [
        "id",
        "base model",
        "merged",
        "dimension",
        "num_heads",
        "num_layers",
        "mlp_ratio",
        "intermediate_size",
        "layer_norm_type",
        "positional_embeddings",
        "attention_variant",
        "biases",
        "block_type",
        "activation",
        "sequence_length",
        "batch_instances",
        "weight_tying",
        "total_params",
        "vocab_size",
        "link to pretraining data",
        "pretraining data",
        "percentage english data",
    ]
    fields = {field: data.get(field, None) for field in fields}

    if data.get("training_stages"):
        # search for "pretraining" stages
        pretraining_stages = [
            stage
            for stage in data["training_stages"]
            if stage["name"] == "pretraining" and "data" in stage
        ]
        if len(pretraining_stages) > 0:
            features = [
                "summary:total_size_tokens_billions",
                "summary:percentage_web",
                "summary:percentage_code",
                "summary:percentage_books",
                "summary:percentage_reference",
                "summary:percentage_academic",
                "summary:percentage_english",
            ]
            data_feats = extract_features_from_json(
                pretraining_stages[0]["data"], features
            )
            fields.update(data_feats)

    return fields


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_metadata_dir", type=str, default="metadata/model_metadata"
    )
    parser.add_argument(
        "--output_file", type=str, default="metadata/annotation_sheet.csv"
    )
    parser.add_argument(
        "--already_done",
        type=str,
        default="metadata/performance_prediction/gathered_data/model_features_annot_4_22.csv",
        help="Path to already annotated data (should have a column id identifying the model)",
    )
    args = parser.parse_args()
    done_ids = pd.read_csv(args.already_done)["id"].tolist()

    rows = []
    for root, _, files in os.walk(args.model_metadata_dir):
        for file in files:
            if file.endswith(".json"):
                row = json_to_row(os.path.join(root, file))
                if row["id"] not in done_ids:
                    rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("id")
    df.to_csv(args.output_file, index=False)
