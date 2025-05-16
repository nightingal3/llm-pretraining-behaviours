import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import gc
import argparse
from typing import List, Dict, Optional

sns.set_theme(style="whitegrid", font_scale=2)

# Define the feature-subfeature relationships
FEATURE_SUBFEATURE_MAP: Dict[str, List[str]] = {
    "char_len": [],
    "num_tokens": [],
    "unique_tokens": [],
    "const_parse": ["const_tree_depth", "num_words_input", "num_sentences_input"],
    "dep_parse": ["dist_to_head", "dist_to_root"],
    "code_features": ["dist_to_def", "tree_depth"]
}

def domain_switch(filename: str, file_path: str, merge_domains: bool = False) -> str:
    if "dolma-c4" in filename:
        domain = "dolma-c4"
    elif "dolma-common-crawl" in filename:
        domain = "dolma-common-crawl"
    elif "dolma-peS2o" in filename:
        domain = "dolma-peS2o"
    elif "dolma-stack-code" in filename:
        domain = "dolma-stack-code"
    elif "dolma-gutenberg" in filename:
        domain = "dolma-gutenberg-books"
    elif "dolma-wikipedia" in filename:
        domain = "dolma-wikipedia"
    else:
        parent_dir = str(os.path.basename(os.path.dirname(file_path)))
        domain = parent_dir
    
    if merge_domains:
        if "cosmopedia" in domain.lower():
            return "cosmopedia"
        elif "smollm" in domain.lower():
            return "smollm"
        elif "dolma" in domain.lower():
            return "dolma"
        
    return domain

def process_dict(d: dict, feature: str = "const_tree_depth") -> dict:
    pooled_subfeats = ["num_words_input", "num_sentences_input", "tree_depth", "const_tree_depth"]
    expanded_subfeats = ["dist_to_head", "dist_to_root", "dist_to_def"]

    if feature in pooled_subfeats:
        try:
            return d[feature].max()
        except AttributeError:
            return d[feature]
    elif feature in expanded_subfeats:
        try:
            return d[feature].mean()
        except:
            return None
    else:
        return d[feature]

def process_base_dir(base_dir: str, sel_feature: str, subfeature: Optional[str] = None, merge_domains: bool = False) -> pd.DataFrame:
    all_features = []
    domains = []

    for filename in os.listdir(base_dir):
        if not filename.endswith(".parquet"):
            continue
        if not sel_feature in filename:
            continue
        
        print(f"Processing file: {filename}, for feature: {sel_feature}, subfeature: {subfeature}")
        file_path = os.path.join(base_dir, filename)
        domain = domain_switch(filename, file_path, merge_domains=merge_domains)
        df = pd.read_parquet(file_path)
        
        if len(df) < 1:  # debugging file likely
            print(f"Short file {filename}, skipping")
            continue
        
        if sel_feature in ["const_parse", "dep_parse", "code_features"] and subfeature:
            df[sel_feature] = df[sel_feature].apply(lambda x: process_dict(x, subfeature))

        all_features.extend(df[sel_feature])
        domains.extend([domain] * len(df))

    df_feature = pd.DataFrame({"feature": all_features, "domain": domains})
    
    # remove inf/nan rows
    df_feature = df_feature[df_feature["feature"].notna()]
    df_feature = df_feature[df_feature["feature"] != float("inf")]
    df_feature = df_feature[df_feature["feature"] != float("-inf")]

    return df_feature

def get_subdirectories(top_dir: str) -> List[str]:
    return [os.path.join(top_dir, d) for d in os.listdir(top_dir) 
            if os.path.isdir(os.path.join(top_dir, d))]

def plot_and_save(df_combined: pd.DataFrame, sel_feature: str, subfeature: Optional[str] = None):
    plt.figure(figsize=(20, 10))

    domain_order = sorted(list(df_combined["domain"].unique()))
    palette = sns.color_palette("muted", len(domain_order))
    color_palette = dict(zip(domain_order, palette))

    sns.histplot(
        data=df_combined,
        x="feature",
        hue="domain",
        bins=100,
        alpha=0.3,
        log_scale=True,
        palette=color_palette,
        hue_order=domain_order,
        stat="density",
        common_norm=False,
    )

    plt.tight_layout()
    os.makedirs("./figures", exist_ok=True)
    feature_name = f"{sel_feature}_{subfeature}" if subfeature else sel_feature
    plt.xlabel(feature_name)
    plt.savefig(f"./figures/{feature_name}_distribution.png")
    plt.gcf().clear()

    df_summary = df_combined.groupby("domain").agg({"feature": ["max", "min", "mean", "std"]})
    print(f"\nSummary for {feature_name}:")
    print(df_summary)

def process_all_features(dirs_to_process: List[str], merge_domains: bool = False):
    for feature, subfeatures in FEATURE_SUBFEATURE_MAP.items():
        if not subfeatures:
            process_single_feature(dirs_to_process, feature, merge_domains=merge_domains)
        else:
            for subfeature in subfeatures:
                process_single_feature(dirs_to_process, feature, subfeature, merge_domains=merge_domains)

def process_single_feature(dirs_to_process: List[str], feature: str, subfeature: Optional[str] = None, merge_domains: bool = False):
    all_data = []
    for base_dir in dirs_to_process:
        if os.path.exists(base_dir):
            df_feature = process_base_dir(base_dir, feature, subfeature, merge_domains=merge_domains)
            all_data.append(df_feature)
        else:
            print(f"Directory {base_dir} does not exist")

    if all_data:
        df_combined = pd.concat(all_data)
        try:
            plot_and_save(df_combined, feature, subfeature)
            del df_combined
        except:
            print(f"Error plotting and saving data for {feature} {subfeature if subfeature else ''}")
        gc.collect()
    else:
        print(f"No data to process for {feature} {subfeature if subfeature else ''}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", help="Base directories to process")
    parser.add_argument("--top-dir", type=str, help="Top-level directory containing subdomains")
    parser.add_argument("--feature", type=str, help="Feature to process", choices=list(FEATURE_SUBFEATURE_MAP.keys()))
    parser.add_argument("--subfeature", type=str, help="For const parse/dep parse/code subfeatures")
    parser.add_argument("--all", action="store_true", help="Process all features and subfeatures")
    parser.add_argument("--merge_domains", action="store_true", help="Merge domains for plotting")
    args = parser.parse_args()

    dirs_to_process = []
    if args.top_dir:
        dirs_to_process.extend(get_subdirectories(args.top_dir))
    if args.dirs:
        dirs_to_process.extend(args.dirs)

    if not dirs_to_process:
        print("No directories specified. Use --dirs and/or --top-dir to specify directories to process.")
        exit(1)

    if args.all:
        process_all_features(dirs_to_process, merge_domains=args.merge_domains)
    elif args.feature:
        if args.subfeature and args.subfeature not in FEATURE_SUBFEATURE_MAP.get(args.feature, []):
            print(f"Invalid subfeature {args.subfeature} for feature {args.feature}")
            exit(1)
        process_single_feature(dirs_to_process, args.feature, args.subfeature, merge_domains=args.merge_domains)
    else:
        print("Please specify either --all or --feature")
        exit(1)