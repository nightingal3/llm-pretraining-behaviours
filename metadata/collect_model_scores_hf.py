from datasets import load_dataset
import os
import subprocess
import argparse
from datetime import datetime
from typing import Optional
import json
import warnings


def get_latest_results_json(directory: str) -> str:
    """
    Get the latest results JSON file in a directory.
    """
    latest_time = None
    latest_file = None

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                file_time = datetime.strptime(
                    file_path.split("results_")[1].split(".json")[0],
                    "%Y-%m-%dT%H-%M-%S.%f",
                )
            except:
                warnings.warn(f"Ignoring file {file_path} without timestamp")
                continue
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
                latest_file = file_path

    return latest_file, latest_time


def merge_all_json_results(directory: str) -> dict:
    """
    Merge all JSON results files in a directory. Overwrite older results with newer ones.
    """
    results_merged = {}
    latest_time = None
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            file_time = datetime.strptime(
                file_path.split("results_")[1].split(".json")[0],
                "%Y-%m-%dT%H-%M-%S.%f",
            )
        except:
            warnings.warn(f"Ignoring file {file_path} without timestamp")
            continue

        with open(file_path, "r") as f:
            results_json = json.load(f)

        if len(results_merged) == 0:
            results_merged["config_general"] = results_json["config_general"]

        if latest_time is None or file_time > latest_time:
            results_merged["config_general"]["latest_update"] = file_time.strftime(
                "%Y-%m-%d"
            )

        for key, val in results_json["results"].items():
            if "results" not in results_merged:
                results_merged["results"] = {}

            if key not in results_merged["results"]:
                results_merged["results"][key] = val
                results_merged["results"][key]["timestamp"] = file_time.strftime(
                    "%Y-%m-%dT%H-%M-%S.%f"
                )
            else:  # compare timestamps
                if file_time > datetime.strptime(
                    results_merged["results"][key]["timestamp"], "%Y-%m-%dT%H-%M-%S.%f"
                ):
                    results_merged["results"][key] = val
                    results_merged["results"][key]["timestamp"] = file_time.strftime(
                        "%Y-%m-%dT%H-%M-%S.%f"
                    )

    return results_merged


def convert_results_format(results_json: dict) -> dict:
    """
    Convert the results in JSON format to a more readable format.
    """
    results_clean = {
        "model_name": results_json["config_general"]["model_name"],
        "last_updated": results_json["config_general"]["latest_update"],
        "results": {},
    }
    for key, val in results_json["results"].items():
        if key == "all":
            continue
        eval_source, dataset_name, num_shots = key.split("|")
        num_shots = f"{num_shots}-shot"
        if eval_source not in results_clean["results"]:
            results_clean["results"][eval_source] = {}
        if dataset_name not in results_clean["results"][eval_source]:
            results_clean["results"][eval_source][dataset_name] = {}
        if num_shots not in results_clean["results"][eval_source][dataset_name]:
            results_clean["results"][eval_source][dataset_name][num_shots] = [val]
        results_clean["results"][eval_source][dataset_name][num_shots] = val

    return results_clean


def get_model_scores(model_name: str) -> dict:
    openllm_url = "https://huggingface.co/datasets/open-llm-leaderboard/results"
    openllm_prefix = "open-llm-leaderboard-results"
    # Note: this repository contains all results on the openLLM leaderboard but it's not working currently (not loadable):
    # https://huggingface.co/datasets/open-llm-leaderboard/results
    # workaround: download it manually and get results from the local directory
    dir_is_found = any(
        d.startswith(openllm_prefix) for d in os.listdir(".") if os.path.isdir(d)
    )
    if not dir_is_found:
        print("Local results repo not found, cloning...")
        subprocess.run(["git", "clone", openllm_url, openllm_prefix], check=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        os.rename(openllm_prefix, openllm_prefix + "-" + date_str)

    dirname_with_date = [d for d in os.listdir(".") if d.startswith(openllm_prefix)][0]

    for root, _, _ in os.walk(dirname_with_date):
        path_parts = root.split(os.sep)
        if len(path_parts) < 2:
            continue

        # search for either the full name, e.g. meta-llama/Llama-2-7b or just the model's name, e.g. Llama-2-7b
        # this should be the final portion of the path, e.g. if we have a directory someone/gpt2, we should be able to find the model with the query "gpt2" but not "someone"
        if (
            path_parts[-1] == model_name
            or os.path.join(path_parts[-2], path_parts[-1]) == model_name
        ):
            merged_results = merge_all_json_results(root)

            results_clean = convert_results_format(merged_results)

            return results_clean

    return {
        "model_name": model_name,
        "last_updated": None,
        "results": {"harness": {}},
    }


def main(model_name: str, output_dir: str, overwrite: bool) -> tuple[dict, str]:
    output_file = os.path.join(
        output_dir, f"results_{model_name.replace('/', '_')}.json"
    )

    if os.path.exists(output_file) and not overwrite:
        print(
            f"File {output_file} already exists, please pass overwrite=True to overwrite."
        )
        return

    model_scores = get_model_scores(model_name)

    if model_scores == None:
        print(f"No results found for {model_name}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(model_scores, f, indent=4)
        print(f"Results written to {output_file}")

    return model_scores, output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="The model name to collect the scores for. Either give a full name, \
            e.g. meta-llama/Llama-2-7b-chat-hf or just the model's name, e.g. Llama-2-7b-chat-hf",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory to output to.",
        default="model_scores",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the metadata file if it already exists.",
    )
    parser.add_argument
    args = parser.parse_args()
    main(args.model_name, args.output_dir, args.overwrite)
