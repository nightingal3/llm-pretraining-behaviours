#!/bin/zsh
# Script for setting up a conda environment with for launching servers
# It sidesteps system-wide installations by relying on conda for most packages
# and by building openssl from source
# TODO: only got it to work with a static build of OpenSSL, which is not ideal
ENV_NAME=towerllm-env

# get the directory of this script, and go one up to get the root directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR="$(dirname "$DIR")"

set -eo pipefail

# check if CONDA_HOME is set and create environment
if [ -z "$CONDA_HOME" ]
then
    echo "Please set CONDA_HOME to the location of your conda installation"
    exit 1
fi
source ${CONDA_HOME}/etc/profile.d/conda.sh
conda create -y -n ${ENV_NAME} python=3.9
conda activate ${ENV_NAME}
# python can't handle this dependency madness, switch to C++
conda install -y -c conda-forge mamba

# install gcc, CUDA and set environment variables
mamba install -y "gxx<10.0" -c conda-forge
mamba install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit cuda-nvcc cuda-cudart

export PATH=${CONDA_HOME}/envs/${ENV_NAME}/bin:$PATH
export LD_LIBRARY_PATH=${CONDA_HOME}/envs/${ENV_NAME}/lib:$LD_LIBRARY_PATH

# # install pytorch
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 \
              -c pytorch -c nvidia

# install apex
pip install ninja 
rm -rf .apex && git clone https://github.com/NVIDIA/apex .apex
cd .apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
               --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./


# install pdsh
git clone git@github.com:chaos/pdsh.git .pdsh
cd .pdsh
autoreconf -i 
./configure --with-ssh --prefix ${DIR}/.pdsh
make -j 8 && make install

# install other dependencies
cd $DIR
pip install -r setup_scripts/pip_reqs.txt

conda env config vars set PATH=$PATH
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH