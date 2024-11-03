import duckdb
import json
import pandas as pd
from pathlib import Path
from typing import Optional

class AnalysisStore:
    def __init__(self, db_path='analysis_store.duckdb'):
        self.con = duckdb.connect(db_path)
        self.setup_schema()
    
    def setup_schema(self):
        self.con.execute("""
            -- Manual annotations table
            CREATE TABLE IF NOT EXISTS model_annotations (
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
                safetensors_total BIGINT,
                is_instruction_tuned BOOLEAN,
                is_preference_tuned BOOLEAN,
                total_params BIGINT
            );

            -- Evaluation results from JSONs
            CREATE TABLE IF NOT EXISTS evaluation_results (
                model_id VARCHAR,
                benchmark VARCHAR,
                setting VARCHAR,
                accuracy DOUBLE,
                accuracy_stderr DOUBLE,
                timestamp TIMESTAMP,
                metadata JSON,
                PRIMARY KEY(model_id, benchmark, setting)
            );
            
            -- Dataset information
            CREATE TABLE IF NOT EXISTS dataset_info (
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
        """)
            
    def import_annotations(self, csv_path: str):
        """Import manually annotated model data"""
        print(f"Importing annotations from {csv_path}")
        self.con.execute("""
            INSERT INTO model_annotations 
            SELECT * FROM read_csv_auto(?)
            ON CONFLICT (id) DO UPDATE SET
                dimension = EXCLUDED.dimension,
                num_heads = EXCLUDED.num_heads,
                -- ... update other fields
                is_instruction_tuned = EXCLUDED.is_instruction_tuned
        """, [csv_path])

    def import_dataset_features(self, csv_path: str):
        """Import dataset features from CSV"""
        print(f"Importing dataset features from {csv_path}")
        
        # First get the actual column names from the CSV
        df = pd.read_csv(csv_path)
        print("CSV columns:", df.columns.tolist())
        
        self.con.execute("""
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
        """, [csv_path])
        
    def import_scores(self, csv_path: str):
        """Import scores from CSV with flexible columns"""
        print(f"Importing scores from {csv_path}")
        
        # Read CSV to get column names
        df = pd.read_csv(csv_path)
        score_columns = [col for col in df.columns if col != 'id']
        
        # Process each score column into components
        processed_benchmarks = set()
        for col in score_columns:
            parts = col.split('_')
            if len(parts) < 2:
                continue
                
            metric = parts[-1]
            base_name = '_'.join(parts[:-1])
            
            # Extract setting if exists
            setting = None
            benchmark = base_name
            for part in base_name.split('_'):
                if 'shot' in part:
                    setting = part
                    benchmark = benchmark.replace(f"_{setting}", '')
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
            acc_col = next((c for c in score_columns if c.startswith(f"{base_name}_") and c.endswith("_acc")), None)
            acc_stderr_col = next((c for c in score_columns if c.startswith(f"{base_name}_") and c.endswith("_acc_stderr")), None)
            brier_col = next((c for c in score_columns if c.startswith(f"{base_name}_") and c.endswith("_brier_score")), None)
            perplexity_col = next((c for c in score_columns if c.startswith(f"{base_name}_") and c.endswith("_perplexity")), None)
            
            # Build metadata
            metadata_parts = []
            if brier_col:
                metadata_parts.append(f"'brier_score', \"{brier_col}\"::DOUBLE")
            if perplexity_col:
                metadata_parts.append(f"'perplexity', \"{perplexity_col}\"::DOUBLE")
            
            metadata_json = f"json_object({', '.join(metadata_parts)})" if metadata_parts else "NULL"
            
            query = f"""
                INSERT INTO evaluation_results 
                (model_id, benchmark, setting, accuracy, accuracy_stderr, metadata, timestamp)
                SELECT 
                    id as model_id,
                    ? as benchmark,
                    ? as setting,
                    {f'"{acc_col}"::DOUBLE' if acc_col else 'NULL'} as accuracy,
                    {f'"{acc_stderr_col}"::DOUBLE' if acc_stderr_col else 'NULL'} as accuracy_stderr,
                    {metadata_json} as metadata,
                    CURRENT_TIMESTAMP as timestamp
                FROM read_csv_auto(?)
                ON CONFLICT (model_id, benchmark, setting) DO UPDATE SET
                    accuracy = EXCLUDED.accuracy,
                    accuracy_stderr = EXCLUDED.accuracy_stderr,
                    metadata = EXCLUDED.metadata
            """
            
            self.con.execute(query, [benchmark, setting, csv_path])

    def verify_scores(self):
        """Verify score import"""
        print("\nScore import verification:")
        print("\nSample of scores with accuracy:")
        print(self.con.execute("""
            SELECT model_id, benchmark, setting, accuracy, accuracy_stderr
            FROM evaluation_results
            WHERE accuracy IS NOT NULL
            LIMIT 5
        """).df())
        
        print("\nSample of scores with other metrics:")
        print(self.con.execute("""
            SELECT model_id, benchmark, setting, 
                   json_extract_string(metadata, '$.brier_score') as brier_score,
                   json_extract_string(metadata, '$.perplexity') as perplexity
            FROM evaluation_results
            WHERE metadata IS NOT NULL
            LIMIT 5
        """).df())

    def get_model_profile(self, model_id: str):
        """Get complete profile for a specific model"""
        # Get model features
        model_features = self.con.execute("""
            SELECT *
            FROM model_annotations
            WHERE id = ?
        """, [model_id]).df()
        
        # Get dataset features
        dataset_features = self.con.execute("""
            SELECT *
            FROM dataset_info
            WHERE id = ?
        """, [model_id]).df()
        
        # Get benchmark scores
        benchmark_scores = self.con.execute("""
            SELECT 
                benchmark,
                setting,
                accuracy,
                accuracy_stderr,
                timestamp
            FROM evaluation_results
            WHERE model_id = ?
            ORDER BY benchmark, setting
        """, [model_id]).df()
        
        return {
            'model_features': model_features,
            'dataset_features': dataset_features,
            'benchmark_scores': benchmark_scores
        }
        
    def import_evaluation_json(self, json_path: str):
        """Import evaluation results from JSON"""
        with open(json_path) as f:
            data = json.load(f)
            
        model_id = data['model_name']
        for benchmark, results in data['results'].items():
            for setting, metrics in results.items():
                self.con.execute("""
                    INSERT INTO evaluation_results 
                    (model_id, benchmark, setting, accuracy, accuracy_stderr, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (model_id, benchmark, setting) DO UPDATE SET
                        accuracy = EXCLUDED.accuracy,
                        accuracy_stderr = EXCLUDED.accuracy_stderr,
                        timestamp = EXCLUDED.timestamp,
                        metadata = EXCLUDED.metadata
                """, [
                    model_id, 
                    benchmark,
                    setting,
                    metrics.get('acc'),
                    metrics.get('acc_stderr'),
                    metrics.get('timestamp'),
                    json.dumps(metrics)
                ])

    def get_analysis_data(self) -> pd.DataFrame:
        """Get joined data for analysis"""
        return self.con.execute("""
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
            LEFT JOIN evaluation_results e ON m.id = e.model_id
        """).df()

    def verify_data(self):
        """Print data verification"""
        print("\nData Verification:")
        for table in ['model_annotations', 'evaluation_results', 'dataset_info']:
            count = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"\n{table} count: {count}")
            print("\nSample data:")
            print(self.con.execute(f"SELECT * FROM {table} LIMIT 3").df())

# Example usage
if __name__ == "__main__":
    store = AnalysisStore()
    
    # Import manual annotations
    store.import_annotations("./performance_prediction/gathered_data/training_model_final.csv")
    store.import_dataset_features("./performance_prediction/gathered_data/training_dataset_final_revised.csv")
    store.import_scores("./performance_prediction/gathered_data/curr_model_scores.csv")

    model_id = "EleutherAI/pythia-410m"
    profile = store.get_model_profile(model_id)
    
    print("\nModel Features:")
    if not profile['model_features'].empty:
        print(profile['model_features'].iloc[0].to_dict())
    else:
        print("No model features found")
        
    print("\nDataset Features:")
    if not profile['dataset_features'].empty:
        print(profile['dataset_features'].iloc[0].to_dict())
    else:
        print("No dataset features found")
        
    print(f"\nScores for {model_id}:")
    scores = store.con.execute("""
        SELECT 
            benchmark,
            setting,
            accuracy,
            accuracy_stderr
        FROM evaluation_results
        WHERE model_id = ?
            AND accuracy IS NOT NULL
            AND accuracy = accuracy
        ORDER BY benchmark, setting
    """, [model_id]).df()
    print(scores)