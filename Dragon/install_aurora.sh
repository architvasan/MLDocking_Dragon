#!/bin/bash

# Load modules
module load tensorflow

# Create venv
python -m venv _dragon_env --system-site-packages
. _dragon_env/bin/activate
pip install dragonhpc
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64

# Install other packages
pip install -i https://pypi.anaconda.org/OpenEye/simple OpenEye-toolkits
pip install SmilesPE
pip install MDAnalysis rdkit
pip install pydantic_settings
pip install pandas
pip install transformers
pip install mpi4py
pip install scikit-learn scikit-learn-intelex
