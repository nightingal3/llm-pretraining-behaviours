import argparse
import pandas as pd

from metadata.duckdb.model_metadata_db import AnalysisStore


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
        default="./performance_prediction/gathered_data/curr_model_scores.csv",
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
        default=["all"],
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
        "--metric", default="acc", choices=["acc", "accuracy", "brier_score", "perplexity"]
    )


def load_data(args: argparse.Namespace):
    if args.db_path:
        print("Loading from DB...")
        store = AnalysisStore.from_existing(args.db_path)

        dataset = store.get_analysis_data(format="wide")

        store.con.close()
    else:
        print("Loading from CSV...")
        # Load the CSV files into pandas DataFrames
        training_scores = pd.read_csv(args.train_labels)

        # Identify duplicates by model_name
        duplicate_model_names = training_scores[
            training_scores.duplicated(subset=["id"], keep=False)
        ]

        if not duplicate_model_names.empty:
            # Group by model_name to handle duplicates
            merged_rows = []
            for model_name, group in duplicate_model_names.groupby("id"):
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
            training_scores = training_scores.drop_duplicates(subset=["id"], keep=False)
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
            left_on="id",
            right_on="id",
        )

    return dataset
