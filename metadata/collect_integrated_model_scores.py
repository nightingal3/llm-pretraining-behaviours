import subprocess
from collect_model_scores_hf import main as collect_model_scores_hf
from typing import Any
import argparse
import torch
from datetime import datetime
import json
import os
from collections import defaultdict
import os
import yaml

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


collected_metrics = ["acc", "brier_score", "exact_match"]


def get_task_list_from_yaml(path_to_yamls: str, task_name: str) -> list[str]:
    """Search through the yaml files in the given directory for the task list. (by top-level name)"""
    for root, _, files in os.walk(path_to_yamls):
        for file in files:
            if file.endswith(".yaml"):
                with open(os.path.join(root, file), "r") as f:
                    yaml_data = yaml.safe_load(f)
                    if yaml_data["group"] == task_name:
                        return yaml_data["task"]
    return []


def evaluate_with_harness(
    model_name: str, task_name: str, output_filepath: str, include_path: str, **kwargs
) -> dict[str, Any]:
    """
    Evaluate a model on a set of tasks using the eval harness.
    """
    task_list = get_task_list_from_yaml(include_path, task_name)

    output_dir = os.path.join(
        output_filepath, "logits", model_name.replace("/", "__"), task_name
    )
    command = f"""lm_eval --model hf --model_args pretrained={model_name},dtype=float16,trust_remote_code=True --include_path {include_path} --tasks {task_name} --device {DEVICE} --batch_size auto:4 --log_samples --output {output_dir}"""
    new_results = {}

    try:
        print(f"Evaluating {task_name} task(s)")
        result = subprocess.run(
            command.split(" "),
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )

        new_results = parse_harness_group_results(result.stdout, task_list)
    except:
        print(f"Failed to evaluate {task_name} task(s)")

    for task in tasks:
        # Find lines related to the task including its subtasks
        task_lines = [
            line for line in lines if f"| - {task}" in line or f"|  - {task}" in line
        ]
        # If there are no lines for this task, continue to the next
        if not task_lines:
            continue

        # Extract data for each subtask
        for line in task_lines:
            parts = [part.strip() for part in line.split("|")]
            for metric in collected_metrics:
                if metric in parts:
                    # Extracting metric value and stderr
                    metric_index = parts.index(metric)
                    metric_value = parts[metric_index + 1]
                    stderr_index = parts.index("±")
                    stderr_value = parts[stderr_index + 1]

                    task_data[task].update(
                        {
                            metric: float(metric_value),
                            f"{metric}_stderr": float(stderr_value),
                            "timestamp": str(datetime.now()),
                        }
                    )

    return task_data


def parse_harness_group_results(results: str, tasks: list[str]) -> dict[str, Any]:

    lines = results.split("\n")
    task_data = defaultdict(dict)

    # Iterate over each task
    for task in tasks:
        # Find lines related to the task including its subtasks
        task_lines = [
            line for line in lines if f"| - {task}" in line or f"|  - {task}" in line
        ]
        # If there are no lines for this task, continue to the next
        if not task_lines:
            continue

        # Extract data for each subtask
        for line in task_lines:
            parts = [part.strip() for part in line.split("|")]
            for metric in collected_metrics:
                if metric in parts:
                    # Extracting metric value and stderr
                    metric_index = parts.index(metric)
                    metric_value = parts[metric_index + 1]
                    stderr_index = parts.index("±")
                    stderr_value = parts[stderr_index + 1]

                    task_data[task].update(
                        {
                            metric: float(metric_value),
                            f"{metric}_stderr": float(stderr_value),
                            "timestamp": str(datetime.now()),
                        }
                    )

    return task_data


def parse_harness_group_json_files(path: str):
    results_file = [
        file
        for file in os.listdir(path)
        if (file.startswith("results_") & file.endswith(".json"))
    ][0]
    timestamp = results_file.replace("results_", "").replace(".json", "")
    task_data = {}
    with open(os.path.join(path, results_file)) as file:
        results_dict = json.load(file)
        results_scores = results_dict["results"]

        group_names = []
        task_names = []
        for group, task_list in results_dict["group_subtasks"].items():
            group_names.append(group)
            for task in task_list:
                if task not in group_names:
                    scores = results_scores[task]
                    task_names.append(task)
                    scores.pop("alias")
                    task_data[task] = {
                        **{k.split(",")[0]: v for k, v in scores.items()},
                        "timestamp": timestamp,
                    }

    return task_data


def parse_harness_results(
    results: str, dataset_name: str, metric: str = "acc"
) -> dict[str, Any]:
    lines = results.split("\n")
    summary_line = next(
        (
            line
            for line in lines
            if metric in line and dataset_name in line and "| - " not in line
        ),
        None,
    )
    if summary_line:
        try:
            parts = [part.strip() for part in summary_line.split("|")]
            parsed_metric = parts[parts.index(metric) + 1]
            stderr = parts[parts.index("±") + 1]
            print(f"Final Metric: {parsed_metric}, Stderr: {stderr}")
            return {
                metric: float(parsed_metric),
                "acc_stderr": float(stderr),
                "timestamp": str(datetime.now()),
            }
        except:
            return {}
    else:
        return {}


def integrated_eval(
    model_name: str,
    task_name: str,
    output_filename: str,
    overwrite: bool = False,
    update: bool = False,
    eval_harness_only: bool = False,
    include_path: str = "./eval_task_groups",
    results_path: str = None,
) -> None:
    """
    Evaluate a model on a set of tasks in eval harness + any additional tasks found in the open llm leaderboard on huggingface.
    """

    if not eval_harness_only:
        # Collect the model scores
        model_scores, json_path = collect_model_scores_hf(
            model_name, output_filename, overwrite
        )

    else:
        model_scores = {"model_name": model_name, "results": {"harness": {}}}
        json_path = os.path.join(
            output_filename, f"results_{model_name.replace('/', '__')}.json"
        )

    if results_path:
        new_results = parse_harness_group_json_files(results_path)
    else:
        # Evaluate the model on the tasks
        new_results = evaluate_with_harness(
            model_name, task_name, output_filename, include_path
        )

    # Update the model scores with the new results
    model_scores["results"]["harness"].update(new_results)

    if os.path.exists(json_path):

        if overwrite:
            # overwrite the json file
            with open(json_path, "w") as f:
                json.dump(model_scores, f, indent=4)
        elif update:
            # update the json file
            with open(json_path, "r") as f:
                old_model_scores = json.load(f)

            old_model_scores["results"]["harness"].update(new_results)
            old_model_scores["last_updated"] = str(datetime.now())

            with open(json_path, "w") as f:
                json.dump(old_model_scores, f, indent=4)
    else:
        new_model_scores = {
            "model_name": model_name,
            "last_updated": str(datetime.now()),
            **model_scores,
        }
        with open(json_path, "w") as f:
            json.dump(new_model_scores, f, indent=4)

    print(f"Scores for '{model_name}' saved to {json_path}")


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
        default="metadata/model_scores",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the metadata file if it already exists.",
    )
    parser.add_argument(
        "--update", action="store_true", help="Update the metadata file with new tasks."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="The name of the tasks/task group.",
        default="reasoning",
    )
    parser.add_argument(
        "--eval_harness_only",
        action="store_true",
        help="Whether to only evaluate the model on the tasks in the eval harness",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        help="Include the task yamls in this directory as well when looking for tasks",
        default="./eval_task_groups",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Path to eval harness results",
        default="./eval_task_groups",
    )

    args = parser.parse_args()

    assert not (args.overwrite and args.update), "Cannot use both overwrite and update"

    integrated_eval(
        args.model_name,
        args.task_name,
        args.output_dir,
        args.overwrite,
        args.update,
        args.eval_harness_only,
        args.include_path,
        args.results_path,
    )
