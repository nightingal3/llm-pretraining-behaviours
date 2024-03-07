import argparse
import json
from typing import Iterable, TextIO
import gzip
import os
import re
# import torch
# import transformers

# from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModel, AutoTokenizer


print("Running remove_ids.py for c4", flush=True)

input_dir = "/data/tir/projects/tir7/user_data/ssingha2/dolma_100B_json/wiki-en-simple/"
output_dir = "/data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/dolma_data_processing/decontamination/coderemoval/analysis-c4.txt"

# # tokens = ["->", "&&", "||", "::", "while(", "!=", "for(i=0"]

# model_dir = "/data/datasets/models/hf_cache/models--meta-llama--Llama-2-7b-chat-hf/snapshots/94b07a6e30c3292b8265ed32ffdeccfdadf434a8"

# try:
#     tokenizer = LlamaTokenizer.from_pretrained(model_dir)
#     model = LlamaForCausalLM.from_pretrained(model_dir)    
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"An error occurred while loading the model: {e}")

# def process(input_text):
#     prompt = f"Does the following text contain code? \"{input_text}\" Answer with True or False."
#     inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
#     output = model.generate(**inputs, max_length=512, num_return_sequences=1)
#     response = tokenizer.decode(output[0], skip_special_tokens=True)

#     # Interpreting the response
#     contains_code = "True" in response
#     return(contains_code)

def _close_when_exhausted(file: TextIO) -> Iterable[str]:
    with file:
 
        for line in file:
            yield json.loads(line)


def open_read_cleaned(filename, is_gzip=False) -> Iterable[str]:
    file: TextIO = gzip.open(filename, "rt") if is_gzip else open(filename, "r")
    return _close_when_exhausted(file)

i = 1
for filename in os.listdir(input_dir):
    if i == 5:
        break
    file_path = os.path.join(input_dir, filename)
    print(f"Processing {filename}...", flush=True)
    corpus = open_read_cleaned(file_path)
    
    # with open(f"{output_dir}", "w+") as f:
    for j, doc in enumerate(corpus, 1):
        text = doc['text']
        print(doc, flush=True)

    i+=1