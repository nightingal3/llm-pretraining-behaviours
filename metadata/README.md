# Model and Dataset Metadata

This directory contains model and dataset metadata.

## Model Metadata

This metadata contains information regarding the model that may be useful in our research.
For a new model, you can seed this by running the `collect_model_metadata.py` script to
grab some data (such as parameter count and architecture) from the hugging face hub.
We will also probably want to gather other data from papers, etc. regarding things
like the model's training data, more detailed architecture information, etc.
The detailed format will be decided at a later date.

## Dataset Metadata

One aspect of model metadata is what data it is trained on. We can link models to datasets
and document the dataset's metdata in a separate directory. The format will be decided
at a later date.

## Validation

This directory validates that the metadata is in the correct format using JSONSchema. To validate all of the schemas, run `pytest`, which will execute `validate_metadata_test.py`, which validates all of the schemas in the `model_metadata` directory.

If the schema check is not passing, you can install `check-jsonschema`

```bash
pip install check-jsonschema
```

and run the following command to check the schema

```bash
check-jsonschema --schemafile model_metadata_schema.json model_metadata/*.json
```

to get more information about where things are breaking.