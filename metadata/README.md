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
