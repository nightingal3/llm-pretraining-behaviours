import duckdb
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List
import re
import os
from datetime import datetime
import glob


class AnalysisStore:
    # Map known metric aliases
    METRIC_MAPPING = {
        # Accuracy metrics
        "acc": "accuracy",
        "exact_match": "accuracy",
        "accuracy": "accuracy",
        # Perplexity metrics
        "perplexity": "perplexity",
        "ppl": "perplexity",
        # Other metrics
        "brier_score": "brier_score",
        "f1": "f1",
        "rouge1": "rouge1",
        "rouge2": "rouge2",
        "rougeL": "rougeL",
        "pass@1,create_test": "accuracy",  # humaneval
        "exact_match": "accuracy",  # we use strict match only
    }

    # Map stderr suffixes
    STDERR_SUFFIXES = [
        "_stderr",
        "_std",
        "_stddev",
        "_error",
        "_stderr,none",
        "_stderr,strict-match",
    ]

    BENCHMARK_DEFAULTS = {
        "arc_challenge": "25-shot",
        "hellaswag": "10-shot",
        "mmlu": "5-shot",
        "truthfulqa": "0-shot",
        "winogrande": "5-shot",
        "lambada": "0-shot",
        "drop": "3-shot",
        "gsm8k": "5-shot",
        "arithmetic": "5-shot",
        "minerva": "5-shot",
        "mathqa": "0-shot",
        "xnli": "0-shot",
        "anli": "0-shot",
        "logiqa2": "0-shot",
        "fld": "0-shot",
        "asdiv": "5-shot",
    }

    @classmethod
    def from_existing(cls, db_path: str):
        """Initialize from an existing duckdb directory without modifying schema"""
        db_path = Path(db_path)

        # Create a new memory connection first
        store = cls.__new__(cls)
        store.con = duckdb.connect(":memory:")

        try:
            # Import the database into this connection
            store.con.execute(f"IMPORT DATABASE '{db_path}'")

            # Verify the import worked
            tables = store.con.execute("SHOW TABLES").fetchall()
            print(f"Loaded existing database from {db_path}")
            print(f"Available tables: {[t[0] for t in tables]}")

            return store

        except Exception as e:
            raise RuntimeError(f"Failed to import database from {db_path}: {e}")

    @classmethod
    def extract_metrics(cls, data: dict, exclude_keys: List = []) -> dict:
        """Extract all metrics and their stderr from data"""
        # First clean up all keys by removing ",none" suffix
        cleaned_data = {
            k.replace(",none", "").replace(",strict-match", ""): v
            for k, v in data.items()
        }
        # remove flexible extract
        cleaned_data = {k: v for k, v in cleaned_data.items() if "flexible" not in k}

        metrics = {}
        for key, value in cleaned_data.items():
            # Skip timestamp and other non-metric fields
            if key == "timestamp" or not isinstance(value, (int, float, str)):
                continue
            if key in exclude_keys:
                continue

            # Convert string numbers to float
            if isinstance(value, str) and value.replace(".", "").isdigit():
                value = float(value)

            # Check if this is a stderr value
            is_stderr = any(suffix in key for suffix in cls.STDERR_SUFFIXES)
            if is_stderr:
                continue

            # Get base metric name
            metric_name = cls.METRIC_MAPPING.get(key, key)

            # Find corresponding stderr if exists
            stderr_value = None
            for suffix in cls.STDERR_SUFFIXES:
                stderr_key = f"{key}{suffix}"
                if stderr_key in cleaned_data:
                    stderr_val = cleaned_data[stderr_key]
                    if (
                        isinstance(stderr_val, str)
                        and stderr_val.replace(".", "").isdigit()
                    ):
                        stderr_val = float(stderr_val)
                    stderr_value = stderr_val
                    break

            metrics[metric_name] = {"value": value, "stderr": stderr_value}

        return metrics

    @classmethod
    def _extract_metrics(cls, data: dict, exclude_keys: List = []) -> dict:
        """Extract all metrics and their stderr from data"""
        metrics = {}
        for key, value in data.items():
            # Skip timestamp and other non-metric fields
            if key == "timestamp" or not isinstance(value, (int, float, str)):
                continue
            if key in exclude_keys:
                continue

            # Convert string numbers to float
            if isinstance(value, str) and value.replace(".", "").isdigit():
                value = float(value)

            # Check if this is a stderr value
            is_stderr = any(suffix in key for suffix in cls.STDERR_SUFFIXES)
            if is_stderr:
                continue

            # Get base metric name
            metric_name = cls.METRIC_MAPPING.get(key, key)

            # Find corresponding stderr if exists
            stderr_value = None
            for suffix in cls.STDERR_SUFFIXES:
                stderr_key = f"{key}{suffix}"
                if stderr_key in data:
                    stderr_val = data[stderr_key]
                    if (
                        isinstance(stderr_val, str)
                        and stderr_val.replace(".", "").isdigit()
                    ):
                        stderr_val = float(stderr_val)
                    stderr_value = stderr_val
                    break

            metrics[metric_name] = {"value": value, "stderr": stderr_value}

        return metrics

    @staticmethod
    def transform_to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the long-format database data to wide format for analysis.
        One row per model, with columns for each benchmark/setting combination.
        """
        # First get unique model features and metadata (one row per model)
        feature_cols = [
            "id",
            "dimension",
            "num_heads",
            "mlp_ratio",
            "layer_norm_type",
            "positional_embeddings",
            "attention_variant",
            "biases",
            "block_type",
            "activation",
            "sequence_length",
            "batch_instances",
            "batch_tokens",
            "weight_tying",
            "is_instruction_tuned",
            "is_preference_tuned",
            "total_params",
            "pretraining_summary_total_tokens_billions",
            "pretraining_summary_percentage_web",
            "pretraining_summary_percentage_code",
            "pretraining_summary_percentage_books",
        ]
        model_features = df[feature_cols].drop_duplicates("id")

        # Create benchmark result columns
        benchmark_data = df[
            ["id", "benchmark", "setting", "accuracy", "accuracy_stderr"]
        ].dropna(subset=["benchmark"])

        def format_benchmark_name(benchmark: str, suffix: str) -> str:
            """Format benchmark name without duplicating shot settings"""
            # Remove any trailing underscores
            return f"{benchmark}_{suffix}".rstrip("_")

        # Pivot accuracy scores
        acc_wide = benchmark_data.pivot(
            index="id", columns="benchmark", values="accuracy"
        )

        acc_wide.columns = [format_benchmark_name(b, "acc") for b in acc_wide.columns]

        # Pivot stderr values
        stderr_wide = benchmark_data.pivot(
            index="id", columns="benchmark", values="accuracy_stderr"
        )
        stderr_wide.columns = [
            format_benchmark_name(b, "acc_stderr") for b in stderr_wide.columns
        ]

        # Combine all parts
        result = pd.merge(model_features, acc_wide.reset_index(), on="id", how="left")
        result = pd.merge(result, stderr_wide.reset_index(), on="id", how="left")

        # Rename id to model_name to match original format
        # result = result.rename(columns={"id": "model_name"})

        return result

    def get_analysis_data(self, format: str = "long") -> pd.DataFrame:
        """
        Get joined data for analysis.

        Args:
            format: Either 'long' (default) or 'wide' format

        Returns:
            DataFrame with all model features and evaluation results
        """
        df = self.con.execute(
            """
            SELECT 
                m.*,
                d.pretraining_summary_total_tokens_billions,
                d.pretraining_summary_percentage_web,
                d.pretraining_summary_percentage_code,
                d.pretraining_summary_percentage_books,
                e.benchmark,
                e.setting,
                e.accuracy,
                e.accuracy_stderr
            FROM model_annotations m
            LEFT JOIN dataset_info d ON m.id = d.id
            LEFT JOIN evaluation_results e ON m.id = e.id
        """
        ).df()

        if format == "wide":
            return self.transform_to_wide_format(df)
        return df

    def __init__(self, db_path="analysis_store.duckdb", create_new=False):
        self.con = duckdb.connect(db_path)
        self.setup_schema()

    def setup_schema(self):
        self.con.execute("DROP TABLE IF EXISTS model_annotations")
        self.con.execute("DROP TABLE IF EXISTS dataset_info")
        self.con.execute("DROP TABLE IF EXISTS evaluation_results")
        self.con.execute(
            """
            -- Manual annotations table
            CREATE TABLE model_annotations (
                id VARCHAR PRIMARY KEY,
                dimension INTEGER,
                num_heads INTEGER,
                mlp_ratio DOUBLE,
                layer_norm_type VARCHAR,
                positional_embeddings VARCHAR,
                attention_variant VARCHAR,
                biases VARCHAR,
                block_type VARCHAR,
                activation VARCHAR,
                sequence_length INTEGER,
                batch_instances INTEGER,
                batch_tokens INTEGER,
                weight_tying BOOLEAN,
                is_instruction_tuned BOOLEAN,
                is_preference_tuned BOOLEAN,
                total_params BIGINT
            );

            -- Evaluation results from JSONs
            CREATE TABLE evaluation_results (
                id VARCHAR,
                benchmark VARCHAR,
                setting VARCHAR,
                accuracy DOUBLE,
                accuracy_stderr DOUBLE,
                timestamp TIMESTAMP,
                metadata JSON,
                PRIMARY KEY(id, benchmark, setting)
            );
            
            -- Dataset information
            CREATE TABLE dataset_info (
                id VARCHAR PRIMARY KEY,
                pretraining_summary_total_tokens_billions DOUBLE,
                pretraining_summary_percentage_web DOUBLE,
                pretraining_summary_percentage_code DOUBLE,
                pretraining_summary_percentage_books DOUBLE,
                pretraining_summary_percentage_reference DOUBLE,
                pretraining_summary_percentage_academic DOUBLE,
                pretraining_summary_percentage_english DOUBLE,
                metadata JSON
            );
        """
        )

    def preview_changes(self, temp_db_path: str = ":memory:") -> dict:
        """Preview changes between current database and a temporary one.
        Returns a dictionary of changes by table."""

        if not os.path.exists(temp_db_path) or temp_db_path == ":memory:":
            print("No previous database found. Skipping comparison.")
            return {}

        temp_con = duckdb.connect(temp_db_path)
        changes = {}
        tables = ["model_annotations", "evaluation_results", "dataset_info"]

        for table in tables:
            # Check if table exists in both databases
            main_exists = self.con.execute(
                f"SELECT * FROM information_schema.tables WHERE table_name = '{table}'"
            ).fetchone()
            temp_exists = temp_con.execute(
                f"SELECT * FROM information_schema.tables WHERE table_name = '{table}'"
            ).fetchone()

            if main_exists and temp_exists:
                # Find differing rows between the main and temp databases
                comparison = self.con.execute(
                    f"""
                    SELECT * FROM (
                        SELECT *, 'old' AS source FROM {table}
                        EXCEPT
                        SELECT *, 'new' AS source FROM temp_db.{table}
                    ) AS differences
                """
                )
                changes[table] = comparison.df()
            elif main_exists:
                changes[table] = "Table exists only in current database"
            elif temp_exists:
                changes[table] = "Table exists only in previous database"

        temp_con.close()
        return changes

    def save_database(self, output_path: str, preview: bool = True) -> None:
        """Save the current database to a new file, optionally previewing changes first."""

        if preview:
            changes = self.preview_changes()

            print("\nDatabase Changes Preview:")
            print("-------------------------")

            for table, df in changes.items():
                if df.empty:
                    print(f"\n{table}: No changes")
                    continue

                print(f"\n{table}:")
                print(f"Total changes: {len(df)}")

                # Group by change type
                changes_by_type = df.groupby("change_type").size()
                for change_type, count in changes_by_type.items():
                    print(f"- {change_type}: {count}")

                # Show detailed changes
                for _, row in df.iterrows():
                    print(f"\n{row['change_type'].upper()}: {row['id']}")
                    for col in df.columns:
                        if col.endswith("_change") and pd.notna(row[col]):
                            changes = row[col]
                            col_name = col.replace("_change", "")
                            print(f"  {col_name}: {changes['old']} -> {changes['new']}")

            # Ask for confirmation
            response = input("\nDo you want to save these changes? (y/N): ")
            if response.lower() != "y":
                print("Save cancelled.")
                return

        # Perform the save
        try:
            # drop temporary data - can cause issues importing
            temp_names = [
                "temp_model_data",
                "new_scores",
                "non_conflicts",
                "temp_scores",
                "conflicts",
            ]

            for name in temp_names:
                try:
                    self.con.unregister(name)
                    print(f"Unregistered {name}")
                except:
                    pass

            self.con.execute(f"EXPORT DATABASE '{output_path}'")
            print(f"\nDatabase saved successfully to {output_path}")
        except Exception as e:
            print(f"Error saving database: {e}")

    def import_model_features_from_csv(self, csv_path: str):
        """Import model features with simplified preprocessing in pandas."""
        print(f"Importing annotations from {csv_path}")

        # Read the CSV into a pandas DataFrame
        df = pd.read_csv(csv_path)

        df["safetensors:total"] = pd.to_numeric(
            df["safetensors:total"], errors="coerce"
        )
        df["total_params"] = pd.to_numeric(df["total_params"], errors="coerce")

        # Replace NaNs with zeros and convert to integer
        df["safetensors:total"] = df["safetensors:total"].fillna(0).astype("Int64")
        df["total_params"] = df["total_params"].fillna(0).astype("Int64")

        # Replace zeros with None to represent them as NULL in the database
        df["safetensors:total"].replace(0, None, inplace=True)
        df["total_params"].replace(0, None, inplace=True)

        # Merge 'safetensors:total' into 'total_params' if it exists
        df["total_params"] = df["safetensors:total"].combine_first(df["total_params"])

        # Drop 'safetensors:total' as it is no longer needed
        df.drop(columns=["safetensors:total"], inplace=True)

        # Register the DataFrame as a temporary table in DuckDB
        self.con.register("temp_model_data", df)

        result_df = self.con.execute(
            """
            SELECT id, total_params
            FROM temp_model_data
            WHERE id IN ('EleutherAI/pythia-410m', 'cerebras/Cerebras-GPT-2.7B', 'allenai/OLMo-7B')
            """
        ).df()

        # Insert data into the database using the temporary table
        self.con.execute(
            """
            INSERT INTO model_annotations
            SELECT * FROM temp_model_data
            ON CONFLICT (id) DO UPDATE SET
                dimension = COALESCE(EXCLUDED.dimension, model_annotations.dimension),
                num_heads = COALESCE(EXCLUDED.num_heads, model_annotations.num_heads),
                mlp_ratio = COALESCE(EXCLUDED.mlp_ratio, model_annotations.mlp_ratio),
                layer_norm_type = COALESCE(EXCLUDED.layer_norm_type, model_annotations.layer_norm_type),
                positional_embeddings = COALESCE(EXCLUDED.positional_embeddings, model_annotations.positional_embeddings),
                attention_variant = COALESCE(EXCLUDED.attention_variant, model_annotations.attention_variant),
                biases = COALESCE(EXCLUDED.biases, model_annotations.biases),
                block_type = COALESCE(EXCLUDED.block_type, model_annotations.block_type),
                activation = COALESCE(EXCLUDED.activation, model_annotations.activation),
                sequence_length = COALESCE(EXCLUDED.sequence_length, model_annotations.sequence_length),
                batch_instances = COALESCE(EXCLUDED.batch_instances, model_annotations.batch_instances),
                batch_tokens = COALESCE(EXCLUDED.batch_tokens, model_annotations.batch_tokens),
                weight_tying = COALESCE(EXCLUDED.weight_tying, model_annotations.weight_tying),
                total_params = COALESCE(EXCLUDED.total_params, model_annotations.total_params),
                is_instruction_tuned = COALESCE(EXCLUDED.is_instruction_tuned, model_annotations.is_instruction_tuned),
                is_preference_tuned = COALESCE(EXCLUDED.is_preference_tuned, model_annotations.is_preference_tuned);
            """
        )

        self.con.execute(
            """
            SELECT id, total_params
            FROM model_annotations
            WHERE id IN ('EleutherAI/pythia-410m', 'cerebras/Cerebras-GPT-2.7B', 'allenai/OLMo-7B')
            """
        ).df()

        self.con.unregister("temp_model_data")

        print(f"Imported annotations from {csv_path} successfully.")

    def import_dataset_features_from_csv(self, csv_path: str):
        """Import dataset features from CSV"""
        print(f"Importing dataset features from {csv_path}")

        # First get the actual column names from the CSV
        self.con.execute(
            """
            INSERT INTO dataset_info 
            SELECT 
                id,
                "pretraining_summary:total_tokens_billions" as pretraining_summary_total_tokens_billions,
                "pretraining_summary:percentage_web" as pretraining_summary_percentage_web,
                "pretraining_summary:percentage_code" as pretraining_summary_percentage_code,
                "pretraining_summary:percentage_books" as pretraining_summary_percentage_books,
                "pretraining_summary:percentage_reference" as pretraining_summary_percentage_reference,
                "pretraining_summary:percentage_academic" as pretraining_summary_percentage_academic,
                "pretraining_summary:percentage_english" as pretraining_summary_percentage_english,
                NULL as metadata
            FROM read_csv_auto(?)
            ON CONFLICT (id) DO UPDATE SET
                pretraining_summary_total_tokens_billions = EXCLUDED.pretraining_summary_total_tokens_billions,
                pretraining_summary_percentage_web = EXCLUDED.pretraining_summary_percentage_web,
                pretraining_summary_percentage_code = EXCLUDED.pretraining_summary_percentage_code,
                pretraining_summary_percentage_books = EXCLUDED.pretraining_summary_percentage_books,
                pretraining_summary_percentage_reference = EXCLUDED.pretraining_summary_percentage_reference,
                pretraining_summary_percentage_academic = EXCLUDED.pretraining_summary_percentage_academic,
                pretraining_summary_percentage_english = EXCLUDED.pretraining_summary_percentage_english
        """,
            [csv_path],
        )

    def import_scores_from_csv(self, csv_path: str):
        """Import scores from CSV with flexible columns"""
        print(f"Importing scores from {csv_path}")

        # Read CSV to get column names
        df = pd.read_csv(csv_path)
        score_columns = [col for col in df.columns if col != "id"]

        # Process each score column into components
        processed_benchmarks = set()
        for col in score_columns:
            parts = col.split("_")
            if len(parts) < 2:
                continue

            metric = parts[-1]
            base_name = "_".join(parts[:-1])

            # Extract setting if exists
            setting = None
            benchmark = base_name
            for part in base_name.split("_"):
                if "shot" in part:
                    setting = part
                    benchmark = benchmark.replace(f"_{setting}", "")
                    break

            # Skip benchmarks without settings
            if setting is None:
                print(f"Skipping {benchmark} - no setting found")
                continue

            benchmark_key = (benchmark, setting)
            if benchmark_key in processed_benchmarks:
                continue

            processed_benchmarks.add(benchmark_key)
            print(f"Importing {benchmark} - {setting}")

            # Find related columns
            acc_col = next(
                (
                    c
                    for c in score_columns
                    if c.startswith(f"{base_name}_") and c.endswith("_acc")
                ),
                None,
            )
            acc_stderr_col = next(
                (
                    c
                    for c in score_columns
                    if c.startswith(f"{base_name}_") and c.endswith("_acc_stderr")
                ),
                None,
            )
            brier_col = next(
                (
                    c
                    for c in score_columns
                    if c.startswith(f"{base_name}_") and c.endswith("_brier_score")
                ),
                None,
            )
            perplexity_col = next(
                (
                    c
                    for c in score_columns
                    if c.startswith(f"{base_name}_") and c.endswith("_perplexity")
                ),
                None,
            )

            # Build metadata
            metadata_parts = []
            if brier_col:
                metadata_parts.append(f"'brier_score', \"{brier_col}\"::DOUBLE")
            if perplexity_col:
                metadata_parts.append(f"'perplexity', \"{perplexity_col}\"::DOUBLE")

            metadata_json = (
                f"json_object({', '.join(metadata_parts)})"
                if metadata_parts
                else "NULL"
            )

            query = f"""
                INSERT INTO evaluation_results 
                (id, benchmark, setting, accuracy, accuracy_stderr, metadata, timestamp)
                SELECT 
                    id as id,
                    ? as benchmark,
                    ? as setting,
                    {f'"{acc_col}"::DOUBLE' if acc_col else 'NULL'} as accuracy,
                    {f'"{acc_stderr_col}"::DOUBLE' if acc_stderr_col else 'NULL'} as accuracy_stderr,
                    {metadata_json} as metadata,
                    CURRENT_TIMESTAMP as timestamp
                FROM read_csv_auto(?)
                ON CONFLICT (id, benchmark, setting) DO UPDATE SET
                    accuracy = EXCLUDED.accuracy,
                    accuracy_stderr = EXCLUDED.accuracy_stderr,
                    metadata = EXCLUDED.metadata
            """

            self.con.execute(query, [benchmark, setting, csv_path])

    def check_for_conflicts(self) -> pd.DataFrame:
        """Check for any potential conflicts in evaluation results by examining duplicates."""
        return self.con.execute(
            """
            WITH duplicate_checks AS (
                SELECT 
                    id,
                    benchmark,
                    setting,
                    COUNT(*) as occurrence_count,
                    GROUP_CONCAT(accuracy) as accuracy_values,
                    GROUP_CONCAT(accuracy_stderr) as stderr_values,
                    GROUP_CONCAT(timestamp) as timestamps
                FROM evaluation_results
                GROUP BY id, benchmark, setting
                HAVING COUNT(*) > 1
            )
            SELECT * FROM duplicate_checks
            ORDER BY id, benchmark, setting
        """
        ).df()

    def safe_import_scores(
        self, scores_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Safely import scores while checking for conflicts.
        Returns two DataFrames: successfully imported and conflicts.
        """

        # First deduplicate the input data
        def combine_rows(x):
            non_null = x.dropna()
            return non_null.iloc[0] if len(non_null) > 0 else x.iloc[0]

        scores_df_dedup = scores_df.groupby("id", as_index=False).agg(
            {col: combine_rows for col in scores_df.columns}
        )

        # Get accuracy and stderr columns
        acc_cols = [
            col
            for col in scores_df_dedup.columns
            if col.endswith("_acc") and not col.endswith("_acc_stderr")
        ]
        stderr_cols = [
            col for col in scores_df_dedup.columns if col.endswith("_acc_stderr")
        ]

        # Melt accuracy columns
        melted_acc = pd.melt(
            scores_df_dedup,
            id_vars=["id"],
            value_vars=acc_cols,
            var_name="benchmark_setting",
            value_name="accuracy",
        )

        # Melt stderr columns
        melted_stderr = pd.melt(
            scores_df_dedup,
            id_vars=["id"],
            value_vars=stderr_cols,
            var_name="benchmark_setting",
            value_name="accuracy_stderr",
        )

        # Clean up stderr benchmark names to match accuracy names
        melted_stderr["benchmark_setting"] = melted_stderr[
            "benchmark_setting"
        ].str.replace("_stderr", "")

        # Merge accuracy and stderr data
        merged_df = pd.merge(
            melted_acc, melted_stderr, on=["id", "benchmark_setting"], how="left"
        )

        # Extract benchmark and setting from benchmark_setting
        merged_df["setting"] = merged_df["benchmark_setting"].str.extract(
            r"_(\d+[-]?shot)_acc$", flags=re.IGNORECASE
        )
        merged_df["setting"] = merged_df["setting"].fillna("")
        merged_df["benchmark"] = merged_df["benchmark_setting"].apply(
            lambda x: x.split("_acc")[0]
        )

        # Add timestamp and metadata
        merged_df["timestamp"] = pd.Timestamp.now()
        merged_df["metadata"] = None

        # Select and reorder columns
        merged_df = merged_df[
            [
                "id",
                "benchmark",
                "setting",
                "accuracy",
                "accuracy_stderr",
                "timestamp",
                "metadata",
            ]
        ]

        # Drop rows where accuracy is null
        merged_df = merged_df.dropna(subset=["accuracy"])

        # Create temporary table for conflict checking
        self.con.execute("DROP TABLE IF EXISTS temp_scores")
        self.con.execute(
            """
            CREATE TEMP TABLE temp_scores (
                id VARCHAR,
                benchmark VARCHAR,
                setting VARCHAR,
                accuracy DOUBLE,
                accuracy_stderr DOUBLE,
                timestamp TIMESTAMP,
                metadata JSON
            )
        """
        )

        # Register and insert the processed data
        self.con.register("new_scores", merged_df)
        self.con.execute("INSERT INTO temp_scores SELECT * FROM new_scores")

        # Find conflicts and non-conflicts
        conflicts = self.con.execute(
            """
            SELECT 
                n.id,
                n.benchmark,
                n.setting,
                n.accuracy as new_accuracy,
                n.accuracy_stderr as new_stderr,
                n.timestamp as new_timestamp,
                e.accuracy as existing_accuracy,
                e.accuracy_stderr as existing_stderr,
                e.timestamp as existing_timestamp,
                ABS(n.accuracy - e.accuracy) as accuracy_diff
            FROM temp_scores n
            JOIN evaluation_results e 
                ON n.id = e.id 
                AND n.benchmark = e.benchmark 
                AND n.setting = e.setting
            WHERE n.accuracy IS DISTINCT FROM e.accuracy
            OR n.accuracy_stderr IS DISTINCT FROM e.accuracy_stderr
            ORDER BY accuracy_diff DESC
        """
        ).df()

        non_conflicts = self.con.execute(
            """
            SELECT n.*
            FROM temp_scores n
            LEFT JOIN evaluation_results e 
                ON n.id = e.id
                AND n.benchmark = e.benchmark 
                AND n.setting = e.setting
            WHERE e.id IS NULL
        """
        ).df()

        self.con.execute("DROP TABLE IF EXISTS temp_scores")
        return non_conflicts, conflicts

    def resolve_conflicts(
        self, conflicts_df: pd.DataFrame, resolution: str = "newer"
    ) -> None:
        """
        Resolve conflicts using specified strategy.

        Parameters:
        - conflicts_df: DataFrame of conflicts from safe_import_scores
        - resolution: Strategy to resolve conflicts:
            - 'newer': Use newer timestamp
            - 'older': Use older timestamp
            - 'larger': Use larger accuracy value
            - 'smaller': Use smaller accuracy value
        """
        if resolution not in ["newer", "older", "larger", "smaller"]:
            raise ValueError("Invalid resolution strategy")

        # Register conflicts dataframe
        self.con.register("conflicts", conflicts_df)

        # Build resolution query based on strategy
        if resolution == "newer":
            condition = "new_timestamp > existing_timestamp"
        elif resolution == "older":
            condition = "new_timestamp < existing_timestamp"
        elif resolution == "larger":
            condition = "new_accuracy > existing_accuracy"
        else:  # smaller
            condition = "new_accuracy < existing_accuracy"

        # Update records based on resolution strategy
        self.con.execute(
            f"""
            UPDATE evaluation_results e
            SET 
                accuracy = c.new_accuracy,
                accuracy_stderr = c.new_stderr,
                timestamp = c.new_timestamp
            FROM conflicts c
            WHERE e.id = c.id 
                AND e.benchmark = c.benchmark 
                AND e.setting = c.setting
                AND {condition}
        """
        )

    # Example usage
    def import_scores_safely(
        self,
        scores_df: pd.DataFrame,
        auto_resolve: bool = False,
        resolution_strategy: str = "newer",
    ) -> None:
        """
        Import scores with conflict handling.

        Parameters:
        - scores_df: DataFrame with new scores
        - auto_resolve: If True, automatically resolve conflicts using resolution_strategy
        - resolution_strategy: How to resolve conflicts if auto_resolve is True
        """
        non_conflicts, conflicts = self.safe_import_scores(scores_df)

        print(f"Found {len(non_conflicts)} new records to import")
        print(f"Found {len(conflicts)} conflicts")

        if len(conflicts) > 0:
            print("\nConflicts found:")
            for _, row in conflicts.iterrows():
                print(f"\nModel: {row['id']}")
                print(f"Benchmark: {row['benchmark']}, Setting: {row['setting']}")
                print(
                    f"Existing: {row['existing_accuracy']:.4f} ± {row['existing_stderr']:.4f} ({row['existing_timestamp']})"
                )
                print(
                    f"New: {row['new_accuracy']:.4f} ± {row['new_stderr']:.4f} ({row['new_timestamp']})"
                )
                print(f"Difference: {row['accuracy_diff']:.4f}")

            if auto_resolve:
                print(
                    f"\nAutomatically resolving conflicts using strategy: {resolution_strategy}"
                )
                self.resolve_conflicts(conflicts, resolution_strategy)
            else:
                response = input(
                    "\nHow would you like to resolve conflicts?\n"
                    "1. Use newer values\n"
                    "2. Use older values\n"
                    "3. Use larger values\n"
                    "4. Use smaller values\n"
                    "5. Skip conflicts\n"
                    "Enter choice (1-5): "
                )

                if response == "1":
                    self.resolve_conflicts(conflicts, "newer")
                elif response == "2":
                    self.resolve_conflicts(conflicts, "older")
                elif response == "3":
                    self.resolve_conflicts(conflicts, "larger")
                elif response == "4":
                    self.resolve_conflicts(conflicts, "smaller")
                elif response == "5":
                    print("Skipping conflicts")
                else:
                    print("Invalid choice, skipping conflicts")

        # Import non-conflicting records
        if len(non_conflicts) > 0:
            self.con.register("non_conflicts", non_conflicts)
            self.con.execute(
                """
                INSERT INTO evaluation_results 
                SELECT * FROM non_conflicts
            """
            )
            print(f"\nSuccessfully imported {len(non_conflicts)} new records")

    def verify_scores(self):
        """Verify score import"""
        print("\nScore import verification:")
        print("\nSample of scores with accuracy:")
        print(
            self.con.execute(
                """
            SELECT id, benchmark, setting, accuracy, accuracy_stderr
            FROM evaluation_results
            WHERE accuracy IS NOT NULL
            LIMIT 5
        """
            ).df()
        )

        print("\nSample of scores with other metrics:")
        print(
            self.con.execute(
                """
            SELECT id, benchmark, setting, 
                json_extract_string(metadata, '$.brier_score') as brier_score,
                json_extract_string(metadata, '$.perplexity') as perplexity
            FROM evaluation_results
            WHERE metadata IS NOT NULL
            LIMIT 5
        """
            ).df()
        )

    def get_model_profile(self, model_id: str):
        """Get complete profile for a specific model"""
        # Get model features
        model_features = self.con.execute(
            """
            SELECT *
            FROM model_annotations
            WHERE id = ?
        """,
            [model_id],
        ).df()

        # Get dataset features
        dataset_features = self.con.execute(
            """
            SELECT *
            FROM dataset_info
            WHERE id = ?
        """,
            [model_id],
        ).df()

        # Get benchmark scores
        benchmark_scores = self.con.execute(
            """
            SELECT 
                benchmark,
                setting,
                accuracy,
                accuracy_stderr,
                timestamp
            FROM evaluation_results
            WHERE id = ?
            ORDER BY benchmark, setting
        """,
            [model_id],
        ).df()

        return {
            "model_features": model_features,
            "dataset_features": dataset_features,
            "benchmark_scores": benchmark_scores,
        }

    def _standardize_model_id(self, model_id: str) -> str:
        """Standardize model ID to use '/' instead of '__'"""
        return model_id.replace("__", "/")

    def import_scores_from_lm_eval_json(
        self, json_path: str, excludes: List[str] = [], alternate_name_map: dict = {}
    ):
        """Import from lm-eval output format (different from our score format)"""
        with open(json_path) as f:
            data = json.load(f)

        model_name = data.get("model_name")
        assert model_name, "Model name not found in JSON"
        model_name = alternate_name_map.get(model_name, model_name)

        all_results = data.get("results")

        imported = 0
        skipped = 0
        errors = 0
        for benchmark in all_results:
            if benchmark in excludes:
                continue  # to skip aggregates and other non-benchmarks
            try:
                setting = data.get("configs").get(benchmark).get("num_fewshot")
                setting = f"{setting}-shot" if setting else "0-shot"
                metrics_data = all_results[benchmark]

                # Create a new cleaned metrics dictionary
                cleaned_metrics = {}
                for k, v in metrics_data.items():
                    if v == "N/A":
                        continue
                    try:
                        cleaned_metrics[k] = float(v)
                    except (ValueError, TypeError):
                        cleaned_metrics[k] = v

                metrics = AnalysisStore.extract_metrics(
                    cleaned_metrics, exclude_keys=["alias"]
                )
                self._insert_score(
                    model_id=model_name,
                    benchmark=benchmark,
                    setting=setting,
                    metrics=metrics,
                )
                imported += 1
            except Exception as e:
                print(f"Error processing {benchmark}: {e}")
                errors += 1

        print(f"\nImport summary:")
        print(f"Imported {imported} scores")
        print(f"Skipped {skipped} items")
        print(f"Encountered {errors} errors")

    def import_scores_from_json_dir(
        self, json_dir: str, benchmark_defaults: dict = None
    ):
        """Import scores with flexible metric handling

        Handles different metric types and naming conventions:
        - accuracy: acc, exact_match, etc.
        - perplexity: perplexity, ppl
        - brier score: brier_score
        etc.
        """
        json_dir = Path(json_dir)
        print(f"Importing scores from {json_dir}")
        benchmark_defaults = (
            self.BENCHMARK_DEFAULTS
            if benchmark_defaults is None
            else benchmark_defaults
        )

        imported = 0
        skipped = 0
        errors = 0

        for json_path in json_dir.glob("*.json"):
            try:
                with open(json_path) as f:
                    data = json.load(f)

                model_id = data.get("model_name")
                if not model_id:
                    print(f"Skipping {json_path.name} - no model_name found")
                    skipped += 1
                    continue

                results = data.get("results", {})

                # Handle harness section separately
                if "harness" in results:
                    harness_results = results.pop("harness")
                    for benchmark, content in harness_results.items():
                        if isinstance(content, dict):
                            # Check if it has shot settings
                            if any("shot" in setting for setting in content.keys()):
                                # New format with explicit shot settings
                                for setting, metrics_data in content.items():
                                    if "shot" not in setting:
                                        continue
                                    metrics = self.extract_metrics(metrics_data)
                                    timestamp = metrics_data.get("timestamp")
                                    self._insert_score(
                                        model_id=model_id,
                                        benchmark=benchmark,
                                        setting=setting,
                                        metrics=metrics,
                                        timestamp=timestamp,
                                    )
                                    imported += 1
                            else:
                                # Need to add shot setting
                                default_setting = None
                                for prefix, setting in benchmark_defaults.items():
                                    if benchmark.startswith(prefix):
                                        default_setting = setting
                                        break

                                # Special case for minerva_math benchmarks
                                if benchmark.startswith("minerva_math"):
                                    default_setting = "5-shot"

                                if not default_setting:
                                    print(
                                        f"Skipping {benchmark} - no default shot setting found"
                                    )
                                    skipped += 1
                                    continue

                                metrics = extract_metrics(content)
                                timestamp = content.get("timestamp")
                                self._insert_score(
                                    model_id=model_id,
                                    benchmark=benchmark,
                                    setting=default_setting,
                                    metrics=metrics,
                                    timestamp=timestamp,
                                )
                                imported += 1

                # Handle non-harness results
                for benchmark, content in results.items():
                    if isinstance(content, dict) and any(
                        "shot" in setting for setting in content.keys()
                    ):
                        # New format with explicit shot settings
                        for setting, metrics_data in content.items():
                            if "shot" not in setting:
                                continue

                            metrics = extract_metrics(metrics_data)
                            timestamp = metrics_data.get("timestamp")

                            self._insert_score(
                                model_id=model_id,
                                benchmark=benchmark,
                                setting=setting,
                                metrics=metrics,
                                timestamp=timestamp,
                            )
                            imported += 1
                    else:
                        # Old format without shot settings
                        default_setting = None
                        for prefix, setting in benchmark_defaults.items():
                            if benchmark.startswith(prefix):
                                default_setting = setting
                                break

                        if not default_setting:
                            print(
                                f"Skipping {benchmark} - no default shot setting found"
                            )
                            skipped += 1
                            continue

                        metrics = extract_metrics(content)
                        timestamp = content.get("timestamp")

                        self._insert_score(
                            model_id=model_id,
                            benchmark=benchmark,
                            setting=default_setting,
                            metrics=metrics,
                            timestamp=timestamp,
                        )
                        imported += 1

                print(f"Imported scores for {model_id}")

            except Exception as e:
                print(f"Error processing {json_path.name}: {e}")
                errors += 1

        print(f"\nImport summary:")
        print(f"Imported {imported} scores")
        print(f"Skipped {skipped} items")
        print(f"Encountered {errors} errors")

    def _insert_score(
        self,
        model_id: str,
        benchmark: str,
        setting: str,
        metrics: dict,
        timestamp: str = None,
    ):
        """Insert score with multiple metrics into database"""
        # Convert metrics from {metric_name: {value: x, stderr: y}}
        # to list of (metric_name, value, stderr)
        for metric_name, values in metrics.items():
            value = values.get("value")
            stderr = values.get("stderr")

            self.con.execute(
                """
                INSERT INTO evaluation_results 
                (id, benchmark, setting, metric, metric_value, metric_stderr)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (id, benchmark, setting, metric) DO UPDATE SET
                    metric_value = EXCLUDED.metric_value,
                    metric_stderr = EXCLUDED.metric_stderr
            """,
                [
                    model_id,
                    benchmark,
                    setting,
                    metric_name,
                    value,
                    stderr,
                ],
            )

    def verify_data(self):
        """Print data verification"""
        print("\nData Verification:")
        for table in ["model_annotations", "evaluation_results", "dataset_info"]:
            count = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"\n{table} count: {count}")
            print("\nSample data:")
            print(self.con.execute(f"SELECT * FROM {table} LIMIT 3").df())


def load_table_from_db(
    db_path: str, table_to_load: str, metric: Optional[str] = None
) -> pd.DataFrame:
    """Load and join data from DuckDB database"""

    store = AnalysisStore.from_existing(db_path)

    if table_to_load == "model":
        query = """
            SELECT * FROM model_annotations
        """
    elif table_to_load == "dataset":
        query = """
            SELECT * FROM dataset_info
        """
    elif table_to_load == "evaluation":
        if metric:
            query = """
                SELECT * FROM evaluation_results
                WHERE metric = ?
            """
        else:
            query = """
                SELECT * FROM evaluation_results
            """
    else:
        raise ValueError(
            "Invalid table_to_load value. Choose from 'model', 'dataset', 'evaluation'"
        )

    df = store.con.execute(query, [metric]).df()
    store.con.close()
    return df


def update_specific_columns(
    store, csv_path, columns_to_update, table="model_annotations"
):
    """Update only specific columns from CSV while preserving other data"""
    df = pd.read_csv(csv_path)

    # Keep only columns we want to update plus 'id'
    columns_to_keep = ["id"] + columns_to_update
    df = df[columns_to_keep]

    # Register temporary table with exact schema
    store.con.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS temp_model_data (
            id VARCHAR,
            {}
        )
    """.format(
            ",".join(f"{col} VARCHAR" for col in columns_to_update)
        )
    )

    store.con.register("temp_df", df)
    store.con.execute("INSERT INTO temp_model_data SELECT * FROM temp_df")

    # Build dynamic UPDATE query
    update_cols = [
        f"{col} = COALESCE(EXCLUDED.{col}, {table}.{col})" for col in columns_to_update
    ]
    update_stmt = ", ".join(update_cols)

    # Execute update
    store.con.execute(
        f"""
        INSERT INTO {table} ({','.join(columns_to_keep)})
        SELECT * FROM temp_model_data
        ON CONFLICT (id) DO UPDATE SET
            {update_stmt}
    """
    )

    # Cleanup
    store.con.execute("DROP TABLE IF EXISTS temp_model_data")
    store.con.unregister("temp_df")


# Example usage
if __name__ == "__main__":
    # store = AnalysisStore()
    store = AnalysisStore.from_existing(
        "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2025_01_26.duckdb"
    )
    # update_specific_columns(store, "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/updated_data_1_24.csv", ["dimension", "num_heads", "mlp_ratio", "sequence_length"], table="model_annotations")
    # update_specific_columns(store, "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/updated_data_1_24_2.csv", ["pretraining_summary_total_tokens_billions", "pretraining_summary_percentage_web", "pretraining_summary_percentage_code", "pretraining_summary_percentage_books", "pretraining_summary_percentage_reference", "pretraining_summary_percentage_academic", "pretraining_summary_percentage_english"], table="dataset_info")
    # date_str = datetime.now().strftime("%Y_%m_%d")
    # store.save_database(f"/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/{date_str}.duckdb")

    # test_json = "/data/tir/projects/tir5/users/mengyan3/lm_eval_outputs/mmlu/mmlu_Salesforce/codegen-350M-multi.json/Salesforce__codegen-350M-multi/results_2025-01-21T00-01-38.841414.json"
    # store.import_scores_from_lm_eval_json(test_json, ["mmlu"])
    base_dir = "/data/tir/projects/tir5/users/mengyan3/lm_eval_outputs/lambada"
    json_files = glob.glob(f"{base_dir}/**/*.json", recursive=True)

    total_imported = 0
    for json_path in json_files:
        # Skip temp/backup files
        if "temp" in json_path or "backup" in json_path:
            continue

        try:
            store.import_scores_from_lm_eval_json(json_path, excludes=[])
            total_imported += 1
        except Exception as e:
            print(f"Error importing {json_path}: {e}")

    print(f"Successfully imported scores from {total_imported} files")

    breakpoint()
    date_str = datetime.now().strftime("%Y_%m_%d")
    store.save_database(
        f"/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/{date_str}.duckdb"
    )
    # # Import manual annotations
    # store.import_model_features_from_csv(
    #     "./performance_prediction/gathered_data/training_model_final.csv"
    # )
    # store.import_dataset_features_from_csv(
    #     "./performance_prediction/gathered_data/training_dataset_final_revised.csv"
    # )
    # print("\nImporting scores...")
    # # scores_df = pd.read_csv(
    # #     "./performance_prediction/gathered_data/curr_model_scores.csv"
    # # )

    # # store.import_scores_safely(
    # #     scores_df, auto_resolve=False  # Set to True if you want automatic resolution
    # # )
    # store.import_scores_from_json_dir(
    #     "/data/tir/projects/tir6/general/mengyan3/tower-llm-training/metadata/model_scores",
    # )

    # # Verify the imports
    # print("\nVerifying data...")
    # store.verify_data()

    # model_id = "EleutherAI/pythia-410m"
    # profile = store.get_model_profile(model_id)

    # print("\nModel Features:")
    # if not profile["model_features"].empty:
    #     print(profile["model_features"].iloc[0].to_dict())
    # else:
    #     print("No model features found")

    # print("\nDataset Features:")
    # if not profile["dataset_features"].empty:
    #     print(profile["dataset_features"].iloc[0].to_dict())
    # else:
    #     print("No dataset features found")

    # print(f"\nScores for {model_id}:")
    # scores = store.con.execute(
    #     """
    #     SELECT
    #         benchmark,
    #         setting,
    #         metric,
    #         metric_value,
    #         metric_stderr
    #     FROM evaluation_results
    #     WHERE id = ?
    #         AND metric_value IS NOT NULL
    #         AND metric_value = metric_value
    #     ORDER BY benchmark, setting
    # """,
    #     [model_id],
    # ).df()
    # print(scores)
    # date_str = datetime.now().strftime("%Y_%m_%d")
    # store.save_database(f"./metadata/duckdb/{date_str}.duckdb")
