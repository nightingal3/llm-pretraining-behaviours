# Model and Dataset Metadata

This directory contains model and dataset metadata.

## Model Metadata

This metadata contains information regarding the model that may be useful in our research.

For a new model, you can seed this:

* For models on the huggingface hub, see the "Collecting Data from Hugging Face Hub" section below
  to seed files in the `model_metadata` directory.
* For models not on the huggingface hub, you can create a new json file in the `model_metadata`
  directory by hand following a similar format.

We will also probably want to gather other data from papers, etc. regarding things
like the model's training data, more detailed architecture information, etc.
The detailed format will be decided at a later date.

## Dataset Metadata

One aspect of model metadata is what data it is trained on. We can link models to datasets
and document the dataset's metdata in a separate directory. Model results link to a dataset, and a dataset can be defined based on other datasets as well. Here is our taxonomy of training data sources (not all of these may be used for real pretraining data).

web/
├─ social_media/
├─ news/
├─ blogs/
├─ forums/
books/
├─ literary/
│  ├─ fiction/
│  ├─ nonfiction/
├─ textbooks/
reference/
├─ encyclopedic/
├─ dictionaries/
academic_papers/
├─ sciences/
├─ humanities/
code/
├─ source_code/
├─ documentation/
├─ forums/
media/
├─ podcasts/
├─ subtitles/
specific_datasets/
├─ <focus of the dataset here, e.g. "finance", "health">/

## Results Metadata

We can also get model results on datasets evaluated by the Huggingface OpenLLM leaderboard in order to track changes in model features and data against final performance. We will also add evaluation options via the EleutherAI evaluation harness for models that have not been uploaded to the leaderboard at a later date.

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

## Collecting Data from Hugging Face Hub

First, you will need to install the requirements in your environment.

```bash
pip install -r requirements.txt
```

Then you can run

```bash
python collect_model_metadata.py [model_name]
```

to grab some data (such as parameter count and architecture) from the hugging face hub.
It will be written to the `model_metadata` directory in a json file following the model name
(but with slashes replaced by underbars).

**Collecting Data from Gated Models/Datasets:**
If you want to run `collect_model_metadata.py` on a private or gated model/dataset, you can
set the `HF_TOKEN` environmental variable to you hugging face token. For public datasets this
is not necessary.

## Collecting evaluation data

In order to collect data about model performance on the Huggingface Open LLM leaderboard, you can use
the following command:

```bash
python collect_model_scores.py [model_name]
```

This will download the current version of the leaderboard data and output scores to `model_scores` by default in json format.
