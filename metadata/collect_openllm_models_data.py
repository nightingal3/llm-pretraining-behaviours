from huggingface_hub import model_info
from typing import Any
from collect_model_scores import get_model_scores
from collect_model_metadata import get_model_metadata
import json
import argparse
import os
import dataclasses
import datetime
import logging
import sys


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Define the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_models",
        type=int,
        help="The number of models to collect metadata and scores for.",
        default=10,
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
    parser.add_argument(
        "--openllm_dir",
        type=str,
        help="Location of directory containing openllm users/models",
    )
    args = parser.parse_args()

    # Set the number of models to fetch
    logging.info(f"Collecting metadata and scores for {args.num_models} models")
    models_fetched = 0

    # Iterate over the models in our local open-llm-leaderboard directory
    for user_name in os.listdir(args.openllm_dir):
        user_path = os.path.join(args.openllm_dir, user_name)
        if os.path.isdir(user_path):
            for model_name in os.listdir(user_path):
                model_path = os.path.join(user_path, model_name)
                if os.path.isdir(model_path):

                    model_name = f"{user_name}/{model_name}"

                    # Define the metadata JSON file name
                    metadata_file_name = os.path.join(
                        args.metadata_output_dir,
                        f"{model_name.replace('/', '_')}.json",
                    )
                    if (
                        os.path.exists(metadata_file_name)
                        and not args.overwrite_metadata
                    ):
                        logging.info(
                            f"Metadata file '{metadata_file_name}' already exists. Skipping."
                        )
                        continue

                    # Define the scores JSON file name
                    scores_file_name = os.path.join(
                        args.scores_output_dir,
                        f"results_{model_name.replace('/', '_')}.json",
                    )
                    if os.path.exists(scores_file_name) and not args.overwrite_scores:
                        logging.info(
                            f"Scores file '{scores_file_name}' already exists. Skipping."
                        )
                        continue

                    # Get the model metadata
                    try:
                        model_metadata = get_model_metadata(model_name, args.token)
                    except Exception as e:
                        logging.info(f"Exception while fetching {model_name} metadata. Skipping.")
                        continue

                    # Get the model scores
                    try:
                        model_scores = get_model_scores(model_name)
                    except Exception as e:
                        logging.info(f"Exception while fetching {model_name} scores. Skipping.")
                        continue

                    # Write the metadata to a JSON file
                    os.makedirs(args.metadata_output_dir, exist_ok=True)
                    with open(metadata_file_name, "w") as f:
                        json.dump(model_metadata, f, indent=4)
                    logging.info(
                        f"Metadata for '{model_name}' saved to '{metadata_file_name}'"
                    )

                    # Write the scores to a JSON file
                    os.makedirs(args.scores_output_dir, exist_ok=True)
                    with open(scores_file_name, "w") as f:
                        json.dump(model_scores, f, indent=4)
                    logging.info(f"Scores for '{model_name}' saved to {scores_file_name}")

                    models_fetched += 1
                    if models_fetched == args.num_models:
                        logging.info(f"Finished collecting metadata and scores for {args.num_models} models.")
                        sys.exit(0)