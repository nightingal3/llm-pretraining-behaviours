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
bash setup_scripts/conda.sh
```

## Running pipelines

The core experimentation and training pipelines rely on ducttape, and are defined in `main.tape`. 
Configuration files for different models and datasets are defined in `configs/`.

For example, to train a small Llama-2 model on a small subset of C4, edit the relevant variables in `configs/llama2_c4small.tconf` and run:

```bash
ducttape main.tape -C configs/llama2_c4small.tconf
```