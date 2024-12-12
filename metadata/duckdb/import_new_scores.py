from pathlib import Path
from model_metadata_db import AnalysisStore
from datetime import datetime

def update_database():
    # Paths
    base_dir = Path("/data/tir/projects/tir6/general/mengyan3/tower-llm-training")
    existing_db = base_dir / "metadata/duckdb/2024_11_12.duckdb"
    json_dir = base_dir / "metadata/model_scores"
    
    print(f"Loading existing database from {existing_db}")
    store = AnalysisStore.from_existing(str(existing_db))
    
    # Get existing benchmark settings from the database
    benchmark_defaults = store.con.execute("""
        SELECT DISTINCT benchmark, setting
        FROM evaluation_results
        WHERE setting IS NOT NULL
    """).fetchall()
    
    # Convert to dictionary
    benchmark_settings = {b: s for b, s in benchmark_defaults}
    print(f"\nFound {len(benchmark_settings)} benchmark settings from existing database:")
    for b, s in benchmark_settings.items():
        print(f"  {b}: {s}")
    
    print(f"\nImporting new scores from {json_dir}")
    store.import_scores_from_json_dir(
        str(json_dir),
        benchmark_defaults=benchmark_settings
    )
    
    # Create new date-stamped database path
    date_str = datetime.now().strftime("%Y_%m_%d")
    new_db = base_dir / f"metadata/duckdb/{date_str}.duckdb"
    
    print(f"\nSaving updated database to {new_db}")
    store.save_database(str(new_db))
    
    # Print some verification info
    print("\nVerifying data:")
    store.verify_data()

if __name__ == "__main__":
    update_database()