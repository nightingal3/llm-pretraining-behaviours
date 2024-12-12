import os
from pathlib import Path

def inspect_db_directory(path):
    path = Path(path)
    print(f"\nInspecting directory: {path}")
    print("\nFiles found:")
    for item in path.iterdir():
        print(f"\n{item.name}:")
        if item.is_file():
            # Print first few lines of text files
            if item.suffix in ['.sql', '.csv']:
                try:
                    with open(item) as f:
                        content = f.read(500)  # Read first 500 chars
                        print("Content preview:")
                        print(content[:200] + "..." if len(content) > 200 else content)
                except UnicodeDecodeError:
                    print("Binary file or different encoding")
            print(f"Size: {item.stat().st_size} bytes")

# Let's also try connecting directly with DuckDB to see any errors
def try_import_db(path):
    import duckdb
    con = duckdb.connect(':memory:')
    try:
        print(f"\nTrying to import database from {path}")
        con.execute(f"IMPORT DATABASE '{path}'")
        print("Success! Tables found:")
        print(con.execute("SHOW TABLES").fetchall())
    except Exception as e:
        print(f"Error importing database: {e}")
        print("Type:", type(e))
    finally:
        con.close()

# Run both checks
inspect_db_directory("/data/tir/projects/tir6/general/mengyan3/tower-llm-training/metadata/duckdb/try.duckdb")
try_import_db("/data/tir/projects/tir6/general/mengyan3/tower-llm-training/metadata/duckdb/try.duckdb")