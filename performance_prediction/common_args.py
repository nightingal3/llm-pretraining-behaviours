import argparse
import pandas as pd

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model_feats",
        type=str,
        help="The path to the CSV file containing the training features related to models.",
        default="./performance_prediction/gathered_data/training_model_final.csv",
    )
    parser.add_argument(
        "--data_feats",
        type=str,
        help="The path to the CSV file containing the training features related to datasets.",
        default="./performance_prediction/gathered_data/training_dataset_final_revised.csv",
    )
    parser.add_argument(
        "--train_labels",
        type=str,
        help="The path to the CSV file containing the training labels",
        default="./performance_prediction/gathered_data/training_score_final.csv",
    )
    parser.add_argument(
        "--feat_subset",
        type=str,
        nargs="+",
        help="Subset of features to use for training",
    )
    parser.add_argument(
        "--y_cols",
        type=str,
        help="The name(s) of the column(s) containing the target variable(s) in the train_labels file",
        nargs="+",
        default=["all"]
    )
    parser.add_argument(
        "--drop_instruction_tuned",
        action="store_true",
        help="Whether to drop models that are instruction tuned",
    )
    parser.add_argument(
        "--new_task_only", action="store_true", help="only keep new tasks"
    )
    parser.add_argument(
        "--metric", default="acc", choices=["acc", "brier_score", "perplexity"]
    )

def load_data(args: argparse.Namespace):
    # Load the CSV files into pandas DataFrames
    training_scores = pd.read_csv(args.train_labels)
    
    # Identify duplicates by model_name
    duplicate_model_names = training_scores[training_scores.duplicated(subset=["model_name"], keep=False)]

    if not duplicate_model_names.empty:
        # Group by model_name to handle duplicates
        merged_rows = []
        for model_name, group in duplicate_model_names.groupby("model_name"):
            merged_row = group.iloc[0].copy()  

            # Iterate through the rest of the rows and merge
            for i, row in group.iterrows():
                for col in group.columns:
                    if pd.isna(merged_row[col]):  
                        merged_row[col] = row[col]
                    elif pd.notna(row[col]) and merged_row[col] != row[col]:
                        raise ValueError(
                            f"Mismatch detected for model {model_name} in column {col}. "
                            f"Values: {merged_row[col]} and {row[col]}"
                        )
            merged_rows.append(merged_row)

        # Remove original duplicates and add the merged rows
        training_scores = training_scores.drop_duplicates(subset=["model_name"], keep=False)
        merged_df = pd.DataFrame(merged_rows)
        training_scores = pd.concat([training_scores, merged_df], ignore_index=True)
    
    if args.model_feats and args.data_feats:
        arch_metadata = pd.read_csv(args.model_feats)
        data_metadata = pd.read_csv(args.data_feats)
        metadata_feats = pd.merge(arch_metadata, data_metadata, on="id")
    elif args.model_feats:
        metadata_feats = pd.read_csv(args.model_feats)
    else:
        metadata_feats = pd.read_csv(args.data_feats)

    # Merge the DataFrames based on 'model_name' and 'id', dropping entries without matches
    dataset = pd.merge(
        training_scores,
        metadata_feats,
        how="inner",
        left_on="model_name",
        right_on="id",
    )

    return dataset

def process_data(dataset: pd.DataFrame, args: argparse.Namespace, cols_from_results: set):
    cols_to_drop = [
        "assigned person",
        "notes",
        "link to instruction/sft data",
        "instruction/sft data",
        "base model",
        "pretraining data",
        "is_preference_tuned",
        "merged",
        "link to pretraining data",
    ]

    categorical_variables = [
        "activation",
        "attention_variant",
        "batch_instances",
        "biases",
        "block_type",
        "layer_norm_type",
    ]

    for col in cols_to_drop:
        if col in dataset.columns:
            dataset = dataset.drop(columns=[col])

    if args.drop_instruction_tuned:
        dataset = dataset[dataset["is_instruction_tuned"] != True]

    if args.predictor_type == "scaling_laws":
        # drop all but total params and num tokens
        dataset = dataset[
            [
                "total_params",
                "pretraining_summary:total_tokens_billions",
                "model_name",
                "id",
            ]
            + list(cols_from_results)
        ]
        categorical_variables = []

    if args.predictor_type == "all":
        if "is_instruction_tuned" in dataset.columns:
            dataset["is_instruction_tuned"] = dataset["is_instruction_tuned"].map(
                {True: 1, False: 0, np.nan: -1}
            )

        for var in categorical_variables:
            dataset[var] = dataset[var].astype("category")

    return dataset
