#!/usr/bin/env bash
# Sets up the Python research environment for replicating:
#   - meg-tong/sycophancy-eval
#   - anthropic-experimental/agentic-misalignment
set -e

echo "=== Python Research Environment Setup ==="

# ── Clone repos ──────────────────────────────────────────────────────────────
mkdir -p vendors

if [ ! -d "vendors/sycophancy-eval" ]; then
    echo "[1/4] Cloning meg-tong/sycophancy-eval..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
        https://github.com/meg-tong/sycophancy-eval \
        vendors/sycophancy-eval
else
    echo "[1/4] vendors/sycophancy-eval already exists — skipping clone."
fi

if [ ! -d "vendors/agentic-misalignment" ]; then
    echo "[2/4] Cloning anthropic-experimental/agentic-misalignment..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
        https://github.com/anthropic-experimental/agentic-misalignment \
        vendors/agentic-misalignment
else
    echo "[2/4] vendors/agentic-misalignment already exists — skipping clone."
fi

# ── Python deps ───────────────────────────────────────────────────────────────
echo "[3/4] Installing Python dependencies..."
pip install -q -r requirements.txt

# ── .env ──────────────────────────────────────────────────────────────────────
echo "[4/4] Checking .env..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "      Created .env from .env.example."
    echo "      ⚠️  Edit .env and set ANTHROPIC_API_KEY before running examples."
else
    echo "      .env already exists."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env — set ANTHROPIC_API_KEY=sk-ant-..."
echo "  2. python run_sycophancy_example.py          # are-you-sure sycophancy eval"
echo "  3. python run_agentic_misalignment_example.py # blackmail misalignment eval"
