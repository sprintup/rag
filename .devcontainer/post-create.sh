#!/usr/bin/env bash
# Runs once when the dev container is created: installs the libraries and downloads the models.
set -e

# Use the CPU-only build of PyTorch. Codespaces has no GPU, and the default Linux build
# bundles about 3 GB of NVIDIA CUDA libraries the app would never use.
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Download the models into models/ (skipped if they're already there)
[ -d models/TinyLlama ] || hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/TinyLlama
[ -d models/all-MiniLM-L6-v2 ] || hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2

echo "Setup complete. Start the app with: streamlit run src/main.py"
