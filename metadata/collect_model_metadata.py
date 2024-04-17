from huggingface_hub import model_info
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


def get_model_metadata(repo_id: str, token: str = None) -> dict[str, Any]:
    """Get metadata for a model from the Hugging Face Hub.

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
    exclude_fields = {"siblings", "spaces"}
    data_dict = {k: v for k, v in data_dict.items() if k not in exclude_fields}

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
        default="architecture_metadata",
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
