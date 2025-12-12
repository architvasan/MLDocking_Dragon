#!/bin/bash

WDIR=/flare/hpe_dragon_collab/csimpson

# Load modules
module load tensorflow

# Set env vars
export NUMEXPR_MAX_THREADS=208
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

# Create venv
. ${WDIR}/_dragon_env/bin/activate


