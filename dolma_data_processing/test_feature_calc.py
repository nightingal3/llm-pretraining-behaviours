import pytest
from transformers import AutoTokenizer
from calc_feature_utils import *

LLAMA_DIR = "/data/datasets/models/huggingface/meta-llama/Llama-2-70b-hf/"
tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)

