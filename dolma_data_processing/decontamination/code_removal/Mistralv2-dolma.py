import time
import argparse
import json
from typing import Iterable, TextIO
import gzip
import os
import pandas as pd


from vllm import LLM, SamplingParams
from Mistralv2_prompt import fixed_prompt_mistral


def read_jsonl(filename, idx, rate):
    """
    Reads the 1000 lines from a JSONL file and returns them as a list of dictionaries.

    :param filename: The path to the JSONL file.
    :return: A list of dictionaries parsed from the 1000 JSON lines.
    """
    ids = []
    texts = []
    data = []
    with open(filename, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i < idx:
                continue

            if i >= idx + rate:
                break

            try:
                json_line = json.loads(line)
                ids.append(json_line["id"])
                texts.append(json_line["text"])
                data.append(json_line)
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {i + 1}")

    return (data, ids, texts)


def get_formatted_prompts(texts):
    """
    Adds the Mistral Prompt formatting to an array
    of texts
    """
    formatted_prompts = [
        f"[INST]{fixed_prompt_mistral}{prompt}[/INST]" for prompt in texts
    ]

    return formatted_prompts


def load_model(model_name, temp=0.1, p=0.95):
    """
    Load the model given the model name and sampling params
    """
    sampling_params = SamplingParams(temperature=temp, top_p=p)
    print(f"Initializing LLM with model: {model_name}", flush=True)
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.9,
        swap_space=16,
        tensor_parallel_size=1,
    )
    print("Loaded model!", flush=True)

    return (llm, sampling_params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        type=str,
        required=False,
        help="LLM",
    )
    parser.add_argument(
        "--input",
        default="data.parquet",
        type=str,
        required=False,
        help="Input prompts",
    )
    parser.add_argument(
        "--output",
        default="continuations",
        type=str,
        required=False,
        help="Output directory",
    )
    parser.add_argument(
        "--max_tokens",
        default=512,
        type=int,
        required=False,
        help="Maximum tokens generated",
    )
    parser.add_argument(
        "--temperature", default=0.7, type=float, required=False, help="Temperature"
    )
    parser.add_argument("--top_p", default=1, type=float, required=False, help="top_p")
    parser.add_argument("--idx", default=0, type=int, required=False, help="index")
    parser.add_argument(
        "--rate", default=1000, type=int, required=False, help="number of prompts"
    )

    args = parser.parse_args()
    print("args:", args, flush=True)

    input_file = args.input
    start_time = time.time()
    data, ids, texts = read_jsonl(input_file, args.idx, args.rate)

    texts = [text.replace("\n", "") for text in texts]
    formatted_prompts = get_formatted_prompts(texts)

    llm_model, sampling_params = load_model(args.model, args.temperature, args.top_p)
    print("Generating outputs...", flush=True)
    outputs = llm_model.generate(formatted_prompts, sampling_params)
    print("Generated outputs...", flush=True)

    decisions_array = []
    for i in range(len(outputs)):
        output = outputs[i]
        prompt = output.prompt
        generated_text = output.outputs[0].text
        contains_code = "True" in generated_text
        if contains_code:
            print(texts[i])
        decision_dict = {"id": ids[i], "contains_code": contains_code}
        decisions_array.append(decision_dict)

    df = pd.DataFrame(decisions_array)

    # Specify your CSV file path/name
    csv_file_path = args.output

    file_exists = os.path.isfile(csv_file_path)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, mode="a", index=False, header=not file_exists)

    print("CSV file has been written using pandas.")

    end_time = time.time()
    print(
        f"Time Taken to inference {len(decisions_array)} prompts is: {end_time-start_time} seconds ",
        flush=True,
    )


if __name__ == "__main__":
    main()
