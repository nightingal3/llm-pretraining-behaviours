


CREATE TABLE dataset_info(id VARCHAR PRIMARY KEY, pretraining_summary_total_tokens_billions DOUBLE, pretraining_summary_percentage_web DOUBLE, pretraining_summary_percentage_code DOUBLE, pretraining_summary_percentage_books DOUBLE, pretraining_summary_percentage_reference DOUBLE, pretraining_summary_percentage_academic DOUBLE, pretraining_summary_percentage_english DOUBLE, metadata JSON);
CREATE TABLE evaluation_results(id VARCHAR, benchmark VARCHAR, setting VARCHAR, metric VARCHAR, metric_value DOUBLE, metric_stderr DOUBLE, PRIMARY KEY(id, benchmark, setting, metric));
CREATE TABLE model_annotations(id VARCHAR PRIMARY KEY, dimension INTEGER, num_heads INTEGER, mlp_ratio DOUBLE, layer_norm_type VARCHAR, positional_embeddings VARCHAR, attention_variant VARCHAR, biases VARCHAR, block_type VARCHAR, activation VARCHAR, sequence_length INTEGER, batch_instances INTEGER, batch_tokens INTEGER, weight_tying BOOLEAN, is_instruction_tuned BOOLEAN, is_preference_tuned BOOLEAN, total_params BIGINT);




