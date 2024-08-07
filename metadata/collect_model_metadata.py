from huggingface_hub import model_info, hf_hub_download
from typing import Any
import json
import argparse
import os
import dataclasses
import datetime


def robust_asdict(obj):
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif dataclasses.is_dataclass(obj):
        return {k: robust_asdict(v) for k, v in dataclasses.asdict(obj).items()}
    elif isinstance(obj, list):
        return [robust_asdict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: robust_asdict(v) for k, v in obj.items()}
    else:
        return obj


def config_data_to_features(config_data: dict) -> dict:
    """
    Reformat fields from config.json to match our features.
    """
    # TODO: None - has to be filled in manually/I'm not sure what field it corresponds to
    config_to_feature_name = {
        "base_model": None,
        "merged": None,
        "dimension": config_data.get("hidden_size"),
        "num_heads": config_data.get("num_attention_heads"),
        "num_layers": config_data.get("num_hidden_layers"),
        "mlp_ratio": (
            config_data.get("intermediate_size") / config_data.get("hidden_size")
            if config_data.get("intermediate_size") and config_data.get("hidden_size")
            else None
        ),
        "intermediate_size": config_data.get("intermediate_size"),
        "layer_norm_type": None,
        "positional_embedding_type": None,
        "attention_variant": None,
        "biases": None,
        "block_type": None,
        "activation": config_data.get("hidden_act"),
        "sequence_length": config_data.get("max_position_embeddings"),
        "batch_instances": None,
        "batch_tokens": None,
        "weight_tying": True if config_data.get("tie_word_embeddings") else False,
        "total_params": None,
        "vocab_size": config_data.get("vocab_size"),
    }

    # remove Nones
    config_to_feature_name = {
        k: v for k, v in config_to_feature_name.items() if v is not None
    }

    return config_to_feature_name


def get_model_metadata(repo_id: str, token: str = None) -> dict[str, Any]:
    """Get metadata for a model from the Hugging Face Hub.
    Also collect data from huggingface config json.

    Args:
        repo_id: The name of the model to get metadata for.
        token: The Hugging Face authentication token. Defaults to None.

    Raises:
        ValueError: If the model is not found.

    Returns:
        The metadata for the model.
    """
    # Fetch model information
    info = model_info(repo_id=repo_id, token=token)

    # Convert the model info to a dictionary
    data_dict = robust_asdict(info)

    # Exclude a few noisy fields related to Hugging Face in particular
    exclude_fields = {
        "siblings",
        "spaces",
        "widget_data",
        "widgetData",
        "likes",
        "downloads",
    }
    data_dict = {k: v for k, v in data_dict.items() if k not in exclude_fields}

    # Get data from the model config
    try:
        config_path = hf_hub_download(
            repo_id=repo_id, filename="config.json", use_auth_token=token
        )

        with open(config_path, "r") as file:
            config_data = json.load(file)

        relevant_fields = config_data_to_features(config_data)
        # Merge the config data with the existing metadata
        data_dict.update(relevant_fields)
    except Exception as e:
        print(f"Failed to download or parse model_config.json: {e}")

    return data_dict


if __name__ == "__main__":
    # Define the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "repo_id", type=str, help="The name of the model to get metadata for."
    )
    parser.add_argument(
        "--token",
        type=str,
        help="The Hugging Face authentication token (only necessary for private "
        "datasets).",
        default=os.getenv("HF_TOKEN"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory to output to.",
        default="model_metadata",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the metadata file if it already exists.",
    )
    args = parser.parse_args()

    # Define the JSON file name
    file_name = os.path.join(args.output_dir, f"{args.repo_id.replace('/', '_')}.json")

    # Get the model metadata
    model_metadata = get_model_metadata(args.repo_id, args.token)

    # Write the metadata to a JSON file
    if os.path.exists(file_name) and not args.overwrite:
        raise FileExistsError(f"The file '{file_name}' already exists.")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(file_name, "w") as json_file:
            json.dump(model_metadata, json_file, indent=4)

        print(f"Metadata for '{args.repo_id}' saved to '{file_name}'")
