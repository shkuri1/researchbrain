#!/usr/bin/env bash
# Sets up the local SLM environment using Ollama.
# Run this after setup.sh (which handles the base Python deps and datasets).
set -e

echo "=== Local SLM Setup (Ollama) ==="
echo ""

# ── Install Ollama ─────────────────────────────────────────────────────────────
if command -v ollama &>/dev/null; then
    echo "[1/2] Ollama already installed: $(ollama --version 2>&1 | head -1)"
else
    echo "[1/2] Installing Ollama..."
    if [[ "$(uname)" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            brew install --cask ollama
        else
            echo "      Homebrew not found."
            echo "      Download Ollama from: https://ollama.com/download"
            exit 1
        fi
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
fi

# ── Install Python ollama package ─────────────────────────────────────────────
echo "[2/2] Installing ollama Python package..."
pip install -q "ollama>=0.3"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Start Ollama:         ollama serve"
echo "     (keep this running in a separate terminal)"
echo ""
echo "  2. Pull a model (pick one):"
echo "     ollama pull llama3.2        # ~2.0 GB — fast, good quality"
echo "     ollama pull mistral         # ~4.1 GB — strong reasoning"
echo "     ollama pull phi3:mini       # ~2.3 GB — efficient"
echo "     ollama pull llama3.1:8b     # ~4.7 GB — larger Llama"
echo ""
echo "  3. Ad hoc chat:"
echo "     python run_adhoc_slm.py"
echo "     python run_adhoc_slm.py --model mistral"
echo "     python run_adhoc_slm.py --once 'What is the capital of France?'"
echo ""
echo "  4. Sycophancy eval (compare to Claude):"
echo "     python run_sycophancy_slm.py"
echo "     python run_sycophancy_slm.py --model mistral --n 20"
echo "     python run_sycophancy_slm.py --challenge 'Are you absolutely certain?'"
