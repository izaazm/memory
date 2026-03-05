#!/bin/bash
set -e

echo "Setting up Conda environment for Cartridges TOFU Experiments..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "conda could not be found. Please install Miniconda or Anaconda."
    exit 1
fi

# Create environment from environment.yml
conda env create -f environment.yml

echo ""
echo "========================================================"
echo "Environment 'cartridges-tofu' created successfully."
echo "Activate it using: conda activate cartridges-tofu"
echo "========================================================"
