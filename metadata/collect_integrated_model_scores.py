import subprocess
from collect_model_scores_hf import main as collect_model_scores_hf
from typing import Any
import argparse
import torch
from datetime import datetime
import json
import os
import torch
from collections import defaultdict
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

task_metric = defaultdict(lambda: "acc")
task_metric.update(
    {
        "minerva_math": "exact_match",
        "minerva_math_algebra": "exact_match",
        "minerva_math_counting_and_prob": "exact_match",
        "minerva_math_geometry": "exact_match",
        "minerva_math_intermediate_algebra": "exact_match",
        "minerva_math_num_theory": "exact_match",
        "minerva_math_prealgebra": "exact_match",
        "minerva_math_precalc": "exact_match",
        "gsm8k": "exact_match",
        "gsm8k_cot": "exact_match",
        "fld": "exact_match",
    }
)


def evaluate_with_harness(
    model_name: str, tasks: set[str], output_filepath: str, **kwargs
) -> dict[str, Any]:
    """
    Evaluate a model on a set of tasks using the eval harness.
    """

    command = """lm_eval --model hf --model_args pretrained={model_name},dtype=float16,trust_remote_code=True --tasks {task} --device {device} --batch_size auto:4 --log_samples --output {output_dir}"""
    new_results = {}
    # note on command: the 'auto' setting for batch size mysteriously causes some tasks to fail
    # setting it to a conservative value that should work in most cases
    for task in tasks:
        output_dir = os.path.join(
            output_filepath, "logits", model_name.replace("/", "_"), task
        )
        print(f"Evaluating {task}")

        try:
            command_task = command.format(
                model_name=model_name, task=task, device=DEVICE, output_dir=output_dir
            )

            result = subprocess.run(
                command_task.split(" "),
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            metric = task_metric[task]
            parsed_results = parse_harness_results(result.stdout, task, metric)
            if parsed_results != {}:  # TODO: how to get num examples
                new_results[task] = {"x-shot": parsed_results}

        except:
            print(f"Failed to evaluate {task}")

    return new_results


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
    tasks: list[str],
    output_filename: str,
    overwrite: bool = False,
    eval_harness_only: bool = False,
) -> None:
    """
    Evaluate a model on a set of tasks + any additional tasks found in the open llm leaderboard on huggingface.
    """

    if not eval_harness_only:
        # Collect the model scores
        model_scores, json_path = collect_model_scores_hf(
            model_name, output_filename, overwrite
        )
        evaled_datasets = set(model_scores["results"]["harness"].keys())
        remaining_tasks = set(tasks) - set(evaled_datasets)
    else:
        model_scores = {"model_name": model_name, "results": {"harness": {}}}
        remaining_tasks = set(tasks)
        json_path = os.path.join(
            output_filename, f"results_{model_name.replace('/', '_')}.json"
        )

    # Evaluate the model on the tasks
    new_results = evaluate_with_harness(model_name, remaining_tasks, output_filename)

    # Update the model scores with the new results
    model_scores["results"]["harness"].update(new_results)

    # overwrite the json file
    with open(json_path, "w") as f:
        json.dump(model_scores, f, indent=4)

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
        "--tasks_file",
        type=str,
        help="The file containing the tasks, one per line",
        default="metadata/eval_harness_tasks.txt",
    )
    parser.add_argument(
        "--eval_harness_only",
        action="store_true",
        help="Whether to only evaluate the model on the tasks in the eval harness",
    )

    args = parser.parse_args()
    with open(args.tasks_file, "r") as f:
        # tasks can also be commented out with #
        tasks = [l.strip() for l in f.readlines() if l[0] != "#"]

    integrated_eval(
        args.model_name, tasks, args.output_dir, args.overwrite, args.eval_harness_only
    )
