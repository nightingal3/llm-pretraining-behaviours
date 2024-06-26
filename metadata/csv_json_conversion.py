import csv
import json
import pandas as pd
import os

from collect_model_metadata import get_model_metadata

with open("/data/tir/projects/tir6/general/mengyan3/tower-llm-training/metadata/model_metadata_schema.json", "r") as f:
    MODEL_METADATA_SCHEMA = json.load(f)

def process_entry(row: dict, has_base_model: bool = False, json_model_dir="/data/tir/projects/tir6/general/mengyan3/tower-llm-training/metadata/model_metadata", json_data_dir="/data/tir/projects/tir6/general/mengyan3/tower-llm-training/metadata/dataset_metadata"):
    model_id = row["id"]
    model_id_filename = model_id.replace("/", "_")
    model_json_path = os.path.join(json_model_dir, "processed_6_9", f"{model_id_filename}.json")
    print(f"Processing model {model_json_path}")

    if not row["total_params"] or not row["summary:total_tokens_billions"]:
        print("Not enough data for this model, skipping...")
        return

    if not os.path.exists(model_json_path):
        # create the metadata file first
        new_model_metadata = get_model_metadata(model_id)
        with open(model_json_path, "w") as json_file:
            json.dump(new_model_metadata, json_file, indent=4)

    if os.path.exists(model_json_path):
        with open(model_json_path, 'r') as json_file:
            model_data = json.load(json_file)

    if has_base_model:
        base_model_id = row["base_model"]
        base_json_path = os.path.join(json_model_dir, f"{base_model_id}.json")
        if os.path.exists(base_json_path):
            with open(base_json_path, "r") as base_json_file:
                base_model_data = json.load(base_json_file)
            if "training_stages" in base_model_data:
                model_data["training_stages"] = base_model_data["training_stages"]
    
    model_features = {k: row[k] for k in row if not k.startswith("summary") and not k.startswith("pretraining") and not "sft" in k} # exclude sft for now
    model_features = {k: v for k, v in model_features.items() if pd.notna(v)}
    model_data.update(model_features)

    # TODO: needs to be rethought maybe, but start by just using the summary? We don't have info on instruction tuning datasets right now
    summary_feats = {k: row[k] for k in row if k.startswith("summary") and pd.notna(row[k])}
    summary_feats = {k.replace("summary:", ""): v for k, v in summary_feats.items()}


    if summary_feats and pd.notna(row["pretraining name"]):
        pretraining_data_filename = row["pretraining name"].replace("/", "_")
        pretraining_json_path = os.path.join(json_data_dir, f"{pretraining_data_filename}.json")

        if os.path.exists(pretraining_json_path):
            with open(pretraining_json_path, "r") as pretraining_json_file:
                pretraining_data = json.load(pretraining_json_file)

            # update only if it's new info - the model may sometimes be trained on more/less data, so there should be two separate values in "total" for the dataset and for the model. But for other features they presumably can just be added here
            for key, val in summary_feats.items():
                if key not in pretraining_data["summary"]:
                    pretraining_data["summary"][key] = val
            
        else:
            # very hard to annotate domains in csv format, just fill out summary
            pretraining_data = {"domains": [], "summary": summary_feats}

        # save the updated pretraining data
        with open(pretraining_json_path, "w") as pretraining_json_file:
            json.dump(pretraining_data, pretraining_json_file, indent=4)

        summary_feats_non_percentage = {k: v for k, v in summary_feats.items() if "percentage" not in k}
        model_data["training_stages"] = model_data.get("training_stages", [])
        model_data["training_stages"].append({"name": "pretraining", "data": pretraining_data_filename, **summary_feats_non_percentage})

    # write json
    with open(model_json_path, "w") as json_file:
        print(f"Writing to {model_json_path}")
        json.dump(model_data, json_file, indent=4)
    print(f"Model {model_id} processed")



def csv_to_json(csv_path, json_model_dir, json_data_dir):
    df = pd.read_csv(csv_path)
    df["total_params"] = df["total_params"].astype(float)

    # Process all base models first since derived models may be in the same sheet
    base_model_entries = df[df["base_model"].isna()]
    non_base_model_entries = df[~df["base_model"].isna()]

    for entry in base_model_entries.to_dict(orient="records"):
        process_entry(entry, json_model_dir=json_model_dir, json_data_dir=json_data_dir)

    for entry in non_base_model_entries.to_dict(orient="records"):
        process_entry(entry, has_base_model=True, json_model_dir=json_model_dir, json_data_dir=json_data_dir)


# Example usage
csv_to_json(
    "/data/tir/projects/tir6/general/mengyan3/tower-llm-training/metadata/performance_prediction/gathered_data/merged_5_29.csv",
    "/data/tir/projects/tir6/general/mengyan3/tower-llm-training/metadata/model_metadata",
    "/data/tir/projects/tir6/general/mengyan3/tower-llm-training/metadata/dataset_metadata",
)
