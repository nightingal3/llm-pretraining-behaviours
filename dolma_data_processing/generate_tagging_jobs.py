import os
from collections import defaultdict
import pandas as pd

base_filepath = "/data/tir/projects/tir7/user_data/mchen5/llm-pretraining-behaviours/dolma"
output_base_filepath="/data/tir/projects/tir6/general/mengyan3/dolma-features"
last_valid_file = {
    "c4_1396000000": 3,
    "common-crawl_5186000000": 2,
    "peS2o_796000000": 3,
    "gutenberg-books_231000000": 0,
    "wiki-en-simple": 1,
}
features_lst = ["num_tokens", "char_len", "unique_tokens", "seq_ind_tok"]
input_files = []
output_files = []
all_cmds = defaultdict(list)
for subdir, dirs, files in os.walk(base_filepath):
    for file in files:
        if file.endswith(".arrow"):
            part_num = int(file.split("_")[-1].split(".")[0])
            if part_num > last_valid_file[subdir.split("/")[-1]]:
                continue
            print(f"Processing {subdir}/{file}")
            input_files.append(f"{subdir}/{file}")
            output_files.append(f"{output_base_filepath}/{file.split('.')[0]}.parquet")

# all combinations of features and input/output files
i = 0
for feature in features_lst:
    for input_file, output_file in zip(input_files, output_files):
        domain = input_file.split("/")[-2]
        all_cmds["TaskID"].append(i + 1)
        all_cmds["feature"].append(feature)
        all_cmds["input_file"].append(input_file)
        all_cmds["output_file"].append(output_file)
        all_cmds["domain"].append(domain)
        i += 1

df = pd.DataFrame(all_cmds)
df.to_csv("./tag_features_commands.csv", index=False, sep="\t")