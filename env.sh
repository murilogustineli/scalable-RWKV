#!/bin/bash

# Create and activate the conda environment
conda create -y --name rwkv_4neo python=3.10
source activate rwkv_4neo

# Install CUDA toolkits
conda install -y -c conda-forge cudatoolkit=11.7 cudatoolkit-dev=11.7 

# Install PyTorch 1.13.1 with CUDA 11.7
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install deepspeed, pytorch-lightning, ninja, wandb, and transformers
python -m pip install deepspeed==0.7.0 pytorch-lightning==1.9.5
python -m pip install ninja wandb transformers

