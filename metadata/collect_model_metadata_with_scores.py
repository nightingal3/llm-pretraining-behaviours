from collect_model_scores_hf import get_model_scores
from collect_model_metadata import get_model_metadata
import json
import argparse
import os


if __name__ == "__main__":
    # Define the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="The name of the model to get metadata and collect scores for.",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="The Hugging Face authentication token (only necessary for private "
        "datasets).",
        default=os.getenv("HF_TOKEN"),
    )
    parser.add_argument(
        "--metadata_output_dir",
        type=str,
        help="The directory to output metadata to.",
        default="model_metadata",
    )
    parser.add_argument(
        "--scores_output_dir",
        type=str,
        help="The directory to output scores to.",
        default="model_scores",
    )
    parser.add_argument(
        "--overwrite_metadata",
        action="store_true",
        help="Whether to overwrite the metadata file if it already exists.",
    )
    parser.add_argument(
        "--overwrite_scores",
        action="store_true",
        help="Whether to overwrite the scores file if it already exists.",
    )
    args = parser.parse_args()

    # Define the metadata JSON file name
    metadata_file_name = os.path.join(
        args.metadata_output_dir, f"{args.model_name.replace('/', '_')}.json"
    )
    if os.path.exists(metadata_file_name) and not args.overwrite_metadata:
        raise FileExistsError(
            f"The metadata file '{metadata_file_name}' already exists. "
            "Please pass overwrite=True to overwrite."
        )

    # Define the scores JSON file name
    scores_file_name = os.path.join(
        args.scores_output_dir, f"results_{args.model_name.replace('/', '_')}.json"
    )
    if os.path.exists(scores_file_name) and not args.overwrite_scores:
        raise FileExistsError(
            f"The scores file '{scores_file_name}' already exists. "
            "Please pass overwrite=True to overwrite."
        )

    # Get the model metadata
    model_metadata = get_model_metadata(args.model_name, args.token)

    # Get the model scores
    model_scores = get_model_scores(args.model_name)

    # Write the metadata to a JSON file
    os.makedirs(args.metadata_output_dir, exist_ok=True)
    with open(metadata_file_name, "w") as f:
        json.dump(model_metadata, f, indent=4)
    print(f"Metadata for '{args.model_name}' saved to '{metadata_file_name}'")

    # Write the scores to a JSON file
    os.makedirs(args.scores_output_dir, exist_ok=True)
    with open(scores_file_name, "w") as f:
        json.dump(model_scores, f, indent=4)
    print(f"Scores for '{args.model_name}' saved to {scores_file_name}")
