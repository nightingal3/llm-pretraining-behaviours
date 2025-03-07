# Not-Just-Scaling Laws

This is the repository associated with the paper [Not-Just-Scaling Laws: Towards a Better Understanding of the Downstream Impact of Language Model Design Decisions](https://arxiv.org/abs/2503.03862). 

For more information, see these sections:
- [Loading the database of model info/evals](#model-database) 
- [Contributing to the DB](#contributions)
- [Pretraining Llama2-architecture models on data mixes](#model-pretraining)
- [Using taggers/classifiers on pretrain data/generations](#data-tagging)

To recreate the results/visualizations from the paper, see [here](#reproducing-results).

## Model database

To load the latest version of the DB, you can read it with duckdb:

```python
import duckdb

con = duckdb.connect(":memory:")
con.execute("IMPORT DATABASE 'metadata/duckdb/2025_03_03.duckdb'")

# three tables - architecture details, pretrain data details, evals
tables = con.execute("SHOW TABLES").fetchall()
print("tables:" )
for t in tables:
    table = t[0]
    print (f"=== {table} ===")
    # show schemas
    print("- schema: ", con.execute(f"DESCRIBE {table}").fetchall())
```

You can also load the datastore with our `AnalysisStore` class which provides some helper functions. 

```python
from model_metadata_db import AnalysisStore

# See metadata/duckdb/model_metadata_db.py for helper fns
store = AnalysisStore.from_existing('metadata/duckdb/2025_03_03.duckdb')
```


## Contributions

We welcome contributions from the community! If you want to contribute information on a new model or evals of existing models, please follow the following instructions:

### How to contribute

Please check back for details of how to contribute! 

#TODO: create PR format

## Installation Instructions

First, clone this repository and its submodules:

```bash
git clone --recurse-submodules https://github.com/nightingal3/llm-pretraining-behaviours.git
```

Then, to create a new conda environment with all the necessary dependencies, run the following command:

```bash
export CONDA_HOME="/path/to/(mini)conda3"
bash setup/conda.sh
```

# Model Pretraining
### Running pretraining

We're currently using bash scripts (which have roughly the same functionality as the ducttape tapes), stored in `./demo_scripts`.

The three main scripts are (`preprocess_data.sh`, `train_model.sh`, and `eval_model.sh (not finalized)`). Usage is as follows:

Preprocessing:
```bash
preprocess_data.sh <arrow file or path to arrow file dir> <output dir for bin dataset> <output dir for dataset json (intermediate output)> <tokenizer> <num cpu workers>
```

Training:
```bash
train_model.sh <path to output checkpoints> <model config yaml> <bin dataset location> <tokenizer>
```

Models can be customized through `./demo_scripts/configs/`. 

Your personal environment variables (like the repo path, wandb username, etc) can be changed through making a copy of `./demo_scripts/configs/.env.template` at `./demo_scripts/configs/.env`, and filling in your information (don't share this with others). 


## Data Tagging

### Domain classification

To run the 4o-mini domain classifier, format your input file in jsonl format and run the following:
```sh
export OPENAI_API_KEY=<API_KEY>
python ../freegens/classify_domains_multistage.py \
    --input $input_file \
    --output $output_file \
    # if text is in an array under 'resps' rather than in a field called 'text'
    [--resps_format] 
```

### Document-level features

As above, format your input file in jsonl format and run the following:
```sh
python get_doc_level_features.py \
    --feature $feature \
    --input $input_file \
    --output $output_file
```


### Keyword-based features

#TODO: standardize this so it works more like the others

### N-gram entropy

This is the only non-document level tagger as of now, it accepts the same input format and can be called like this:

```sh
python whole_corpus_measures/entropy.py \
        --input $input_file \
        --output $output_file \
        --text_column <col_text_is_in> \
        # order ngram to use (1 == use prev 1 token as context, so bigram entropy)
        [--ngram=1] \
        # limit # docs to process
        [--num_docs] \ 
```
## Reproducing Results

### Feature selection

You can see the selected features for each benchmark under `performance_prediction/selected_feats`. To rerun this process, you can run `performance_prediction/feat_selection.sh`.

### Predicting performance

#TODO: add more here about misc scripts/generating figures

To run performance prediction for a task:
```sh
python performance_prediction/performance_predict_from_db.py \
    # accuracy or brier_score
    --metric accuracy \
    --db_path ./metadata/duckdb/2025_03_03.duckdb \
    --predictor_type all \
    --merge_arithmetic \
    --hyperparam_search \
    --merge_mmlu \
    --drop_instruction_tuned \
    # feats from freegens
    --pseudo_feats_csv ./all_models_feature_stats_3_03_with_ratios.csv \
    # if you only want to run one task
    [--sel_tasks] <task_name> \
    # if you only want to use a subset of features (see the selected features for each task for reference)
    [--sel_features] <feats_separated_by_space> \
    # seed (defaults to 42)
    [--seed] <seed>
```

To run the aggregate/significance test across many seeds, you can use:

```sh
python performance_prediction/performance_predict_from_db_gradual.py \
    <same_args_as_above>
    --test_significance \
    --initial_features_csv performance_prediction/selected_feats/forward_generation_results_303.csv \
    # to do only one task
    [--selected_task] <task>
```
## Citation

```bibtex
@misc{liu2025notjustscalinglawsbetterunderstanding,
      title={Not-Just-Scaling Laws: Towards a Better Understanding of the Downstream Impact of Language Model Design Decisions}, 
      author={Emmy Liu and Amanda Bertsch and Lintang Sutawika and Lindia Tjuatja and Patrick Fernandes and Lara Marinov and Michael Chen and Shreya Singhal and Carolin Lawrence and Aditi Raghunathan and Kiril Gashteovski and Graham Neubig},
      year={2025},
      eprint={2503.03862},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.03862}, 
}
```