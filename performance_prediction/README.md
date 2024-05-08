These scripts aggregate collected metadata about models (stored under `metadata`) in order to generate features about the model architecture, training process, and data that can be used to train performance predictors. Currently, this is an xgboost model. 

A list of model features and data features can be found below (please update this when adding new features!):

[*] indicates a scaling law feature.

## Model Features
- activation (the activation function used)
- attention_variant (the attention variant used. Some abbreviations - mqa (multi query attention), gqa (grouped query attention))
- batch_instances (num seqs in a batch)
- batch_tokens (number of tokens in a batch)
- biases (where bias terms exist in the model)
- block_type (whether there are any parallel blocks, a vanilla transformer is "sequential")
- dimension (the embedding dimension)
- mlp_ratio (ratio of FFN's hidden dimension to embedding dimension)
- num_heads (num attention heads)
- positional_embeddings (type of positional embedding)
- sequence_length (sequence length in tokens)
- weight_tying (whether weight tying was used)
- total_params [*]

## Data Features
(These are for each training stage, e.g. pretraining, finetuning, instruction tuning, but can also be aggregated under `summary_<feature>`)

For "tagged features", see `dolma_data_processing` for definitions.

Composition features: 
- total tokens trained on (in billions) [*]
- percentage web data
- percentage book data
- percentage reference data
- percentage academic paper data
- percentage code data
- percentage contaminated data
- percentage instruction data
- is_instruction_tuned: whether or not the model was instruction tuned
- is_preference_tuned: whether or not the model is RLHF'ed/DPOed/etc with preference data

Tagged features:
- mean dependency length
- stdev dependency length
- mean unique tokens per document
- stdev unique tokens per document
- mean tree depth 
- stdev tree depth
- mean entropy over next token
- stdev entropy over next token
\# TODO: should complete the rest of the global info taggers and list them here

