# Model and Dataset Metadata

This directory contains model and dataset metadata.

## Model Metadata

This metadata contains information regarding the model that may be useful in our research.

For a new model, you can seed this:

* For models on the huggingface hub, run `python collect_model_metadata.py [model_name]` to
  grab some data (such as parameter count and architecture) from the hugging face hub.
  It will be written to the `model_metadata` directory in a json file following the model name
  (but with slashes replaced by underbars).
* For models not on the huggingface hub, you can create a new json file in the `model_metadata`
  directory by hand following a similar format.

We will also probably want to gather other data from papers, etc. regarding things
like the model's training data, more detailed architecture information, etc.
The detailed format will be decided at a later date.

## Dataset Metadata

One aspect of model metadata is what data it is trained on. We can link models to datasets
and document the dataset's metdata in a separate directory. The format will be decided
at a later date.

## Details

**Collecting Data from Gated Models/Datasets:**
If you want to run `collect_model_metadata.py` on a private or gated model/dataset, you can
set the `HF_TOKEN` environmental variable to you hugging face token. For public datasets this
is not necessary.
