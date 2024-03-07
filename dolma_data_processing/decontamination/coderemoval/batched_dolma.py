import time
from vllm import LLM, SamplingParams
import json
from typing import Iterable, TextIO
import gzip
import os


print("Running Mistralv1-dolma.py for stack-code", flush=True)

input_dir = "/data/tir/projects/tir7/user_data/ssingha2/dolma_100B_json/stack-code/"
output_dir = "/data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/dolma_data_processing/decontamination/coderemoval/analysis/stack-code"


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
    
langs_map = {'java': 0, 'python': 0, 'c':0, 'c++':0, 'go':0, 'html':0, 'javascript':0 }
langs = list(langs_map.keys())
def _close_when_exhausted(file: TextIO) -> Iterable[str]:
    inputs = []
    with file:
        for line in file:
            doc = json.loads(line)
            if doc['lang'] == "text":
                break
            elif (doc['lang'] in langs):
                if(langs_map[doc['lang']] == 4):
                    break
                langs_map[doc['lang']] += 1
                inputs.append(json.loads(line))
    return(inputs)


def open_read_cleaned(filename, is_gzip=False) -> Iterable[str]:
    file: TextIO = gzip.open(filename, "rt") if is_gzip else open(filename, "r")
    return _close_when_exhausted(file)


# start_time = time.time()
# for filename in os.listdir(input_dir):
#     basename = filename.split(".")[0]
#     file_path = os.path.join(input_dir, filename)
#     print(f"Processing {filename}...", flush=True)
#     corpus = open_read_cleaned(file_path)
        
#     with open(f"{output_dir}/{basename}.jsonl", "w+") as f:
#         for j, doc in enumerate(corpus, 1):
#             text = doc['text']
#             generated_text, contains_code = process(text)
#             if contains_code:
#                 print(doc, file=f)
#                 print("generated text - ", generated_text, file=f)


# end_time = time.time()
# print("time taken: ", end_time-start_time, flush=True)


#1. loop over the files in stack-code
#2. for each file, get three lines
#3. Concatenate them into an array of size 1000
#4. Send that array as input to mistralv1

# num_text = 0
# non_text = 0
# for filename in os.listdir(input_dir):

#     basename = filename.split(".")[0]
#     file_path = os.path.join(input_dir, filename)
#     print(f"Processing {filename}...", flush=True)
#     input = open_read_cleaned(file_path)
    
#     if(input[0]['lang'] == "text"):
#         num_text+=1
#     else:
#         print("non_text")
#         non_text+=1

# print("Number of files with Lang == text: ", num_text, flush=True)
# print("Number of files with Lang != text: ", non_text, flush=True)

for filename in os.listdir(input_dir):

    basename = filename.split(".")[0]
    file_path = os.path.join(input_dir, filename)
    print(f"Processing {filename}...", flush=True)
    inputs = open_read_cleaned(file_path)
    for input in inputs:
        print(input, flush=True)

    

    


