#!/usr/bin/env bash
# Runs once when the dev container is created: installs the libraries and downloads the models.
set -e

# Use the CPU-only build of PyTorch. Codespaces has no GPU, and the default Linux build
# bundles about 3 GB of NVIDIA CUDA libraries the app would never use.
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Download the models into models/ (skipped if they're already there)
[ -d models/TinyLlama ] || hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/TinyLlama
[ -d models/all-MiniLM-L6-v2 ] || hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2

# Ollama serves Qwen for the live Q&A (src/live_qa.py). The version is pinned, so a new release
# isn't installed without anyone noticing (readme-safety.md, section 8).
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.4 sh

# Ollama has to be running to download a model: start it in the background if it isn't
if ! curl -s http://127.0.0.1:11434/api/version >/dev/null; then
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    for _ in $(seq 1 30); do curl -s http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
fi
ollama pull qwen3.5:4b

echo "Setup complete. Start the app with: streamlit run src/main.py"
echo "or the live Q&A with: streamlit run src/live_qa.py"
