COPY dataset_info FROM './metadata/duckdb/try.duckdb/dataset_info.csv' (FORMAT 'csv', force_not_null 'id', quote '"', delimiter ',', header 1);
COPY evaluation_results FROM './metadata/duckdb/try.duckdb/evaluation_results.csv' (FORMAT 'csv', force_not_null ('id', 'benchmark', 'setting'), quote '"', delimiter ',', header 1);
COPY model_annotations FROM './metadata/duckdb/try.duckdb/model_annotations.csv' (FORMAT 'csv', force_not_null 'id', quote '"', delimiter ',', header 1);
