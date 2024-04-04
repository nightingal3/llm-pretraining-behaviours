# TowerLLM: Training Repository

## Installation Instructions

As a pre-requisite, make sure you have [ducttape](https://github.com/CoderPat/ducttape) and [(mini)conda](https://docs.conda.io/en/latest/miniconda.html) installed.

First, clone this repository and its submodules:

```bash
git clone --recurse-submodules git@github.com:deep-spin/tower-llm-training.git
```

Then, to create a new conda environment with all the necessary dependencies, run the following command:

```bash
export CONDA_HOME="/path/to/(mini)conda3"
bash setup/conda.sh
```

A second optional set up is for vllm inferencing on dolma. If you wish to run inferencing on dolma, set up a vllm environment. 
There are 3 options to do this (be mindful of CUDA versions) - 
1. (Recommended) vllm source code - https://docs.vllm.ai/en/latest/getting_started/installation.html
2. pip install vllm
3. (Easiest) - Download the vllm env from - https://huggingface.co/shreyasinghal/vllm_env

Finally, Add your hf_cache path to config.env

## Running reasoning pretraining (temp)

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


## Running pipelines

The core experimentation and training pipelines rely on ducttape, and are defined in `main.tape`. 
Configuration files for different models and datasets are defined in `configs/`.

Start by creating a configuration with user-dependent variables (like the output folder) in associated `configs/*_uservars.conf` associated with your chosen `.tconf`. E.g, for the `configs/tower_llm.tconf` configuration, create a `configs/tower_llm_uservars.conf` file with the following content:
```
global {
    ducttape_output=/mnt/data/patrick/towerllm-outs/
    repo=/home/patrick/tower-llm-training

    (...)
    # use a simple shell submitter 
    # we are forced to explicitly set the submitter parameters
    # to make it compatible with other submitters (ie the slurm submitter)
    submitter=shell
    dump_account=none
    dump_partition=none
    (...)
}
```

Then, you can ran the one of the specified pipelines in `main.tape` by running ducttape with the corresponding configuration file:

```bash
conda activate towerllm-env
ducttape main.tape -C configs/tower_llm.conf 
```
