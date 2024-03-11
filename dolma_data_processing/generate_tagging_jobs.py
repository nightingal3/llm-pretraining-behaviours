import os
from collections import defaultdict
import pandas as pd

base_filepath = "/data/tir/projects/tir5/users/mengyan3/dolma_data_processed/dolma_100B_orig_nl_code"

output_base_filepath = "/data/tir/projects/tir6/general/mengyan3/dolma-features"
last_valid_file = {
    "c4": float("inf"),
    "common-crawl": float("inf"),
    "peS2o": float("inf"),
    "gutenberg-books": float("inf"),
    "wiki-en-simple": float("inf"),
    "stack-code": float("inf"),
}
features_lst = ["num_tokens", "char_len", "unique_tokens", "seq_ind_tok"]
sel_domains = ["stack-code"]
# sel_domains = ["c4", "common-crawl", "peS2o", "gutenberg-books", "wiki-en-simple"]
# features_lst = ["dep_parse", "const_parse"]
input_files = []
output_files = []
all_cmds = defaultdict(list)
for subdir, dirs, files in os.walk(base_filepath):
    for file in files:
        if file.endswith(".arrow"):
            if subdir.split("/")[-1] not in sel_domains:
                continue
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
df.to_csv("./slurm_scripts/tag_simple_stack.csv", index=False, sep="\t")
