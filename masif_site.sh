#!/bin/bash

VENV="/lrlhps/data/compbio_shared/apps/masif/venv"
PROF="/lrlhps/data/compbio_shared/conda.profile"

source ${PROF}
conda activate ${VENV}
python source/data_preparation/compute_surface_features.py "$@"