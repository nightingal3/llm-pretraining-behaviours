import time
from vllm import LLM, SamplingParams
import json
from typing import Iterable, TextIO
import gzip
import os


print("Running Mistralv1-dolma.py for common-crawl", flush=True)

input_dir = "/data/tir/projects/tir7/user_data/ssingha2/dolma_100B_json/common-crawl/"
output_dir = "/data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/dolma_data_processing/decontamination/coderemoval/analysis/mistralv1"


def process(prompt):
    fixed_prompt = "Does the following text contain code? Answer with True or False: "

    formatted_prompt = f"[INST]{fixed_prompt}{prompt}[/INST]"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1", gpu_memory_utilization=0.9, swap_space = 16, tensor_parallel_size=1,)
    outputs = llm.generate(formatted_prompt, sampling_params)
    print(outputs, flush=True)
    generated_text = outputs[0].outputs[0].text
    contains_code = "True" in generated_text
    return((generated_text, contains_code))
    

def _close_when_exhausted(file: TextIO) -> Iterable[str]:
    with file:
        for line in file:
            yield json.loads(line)


def open_read_cleaned(filename, is_gzip=False) -> Iterable[str]:
    file: TextIO = gzip.open(filename, "rt") if is_gzip else open(filename, "r")
    return _close_when_exhausted(file)


start_time = time.time()
for filename in os.listdir(input_dir):
    basename = filename.split(".")[0]
    file_path = os.path.join(input_dir, filename)
    print(f"Processing {filename}...", flush=True)
    corpus = open_read_cleaned(file_path)
        
    with open(f"{output_dir}/{basename}.jsonl", "w+") as f:
        for j, doc in enumerate(corpus, 1):
            text = doc['text']
            generated_text, contains_code = process(text)
            if contains_code:
                print(doc, file=f)
                print("generated text - ", generated_text, file=f)


end_time = time.time()
print("time taken: ", end_time-start_time, flush=True)


