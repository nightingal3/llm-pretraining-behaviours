These scripts aggregate collected metadata about models (stored under `metadata`) in order to generate features about the model architecture, training process, and data that can be used to train performance predictors. Currently, this is an xgboost model. 

A list of model features and data features can be found below (please update this when adding new features!):

## Model Features
- activation
- attention_variant
- batch_instances
- batch_tokens
- biases
- block_type
- dimension
- layer_norm_type
- mlp_ratio
- num_heads
- positional_embeddings
- sequence_length
- weight_tying

## Data Features
(These are for each training stage, e.g. pretraining, finetuning, instruction tuning, but can also be aggregated under `summary_<feature>`)

For "tagged features", see `dolma_data_processing` for definitions.

Composition features: 
- total tokens trained on (in billions)
- percentage web data
- percentage book data
- percentage reference data
- percentage academic paper data
- percentage code data
- percentage contaminated data
- percentage instruction data
- is_instruction_tuned: whether or not the model was instruction tuned

Tagged features:
- mean dependency length
- stdev depdendency length
- mean unique tokens per document
- stdev unique tokens per document
- mean tree depth 
- stdev tree depth
- mean entropy over next token
- stdev entropy over next token
# TODO: should complete the rest of the global info taggers and list them here

