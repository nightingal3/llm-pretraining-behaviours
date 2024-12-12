COPY dataset_info FROM './metadata/duckdb/2024_12_04.duckdb/dataset_info.csv' (FORMAT 'csv', force_not_null 'id', quote '"', delimiter ',', header 1);
COPY evaluation_results FROM './metadata/duckdb/2024_12_04.duckdb/evaluation_results.csv' (FORMAT 'csv', force_not_null ('id', 'benchmark', 'setting', 'metric'), quote '"', delimiter ',', header 1);
COPY model_annotations FROM './metadata/duckdb/2024_12_04.duckdb/model_annotations.csv' (FORMAT 'csv', force_not_null 'id', quote '"', delimiter ',', header 1);
