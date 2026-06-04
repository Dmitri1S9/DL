#!/usr/bin/env bash
# One-shot environment setup for fine-tuning on Colab / Kaggle.
#
# Run from the repo root, AFTER cloning and cd-ing into it:
#     !bash scripts/colab_setup.sh
#
# Note: a script cannot export PYTHONPATH into the notebook's shell, so prefix the
# run command with it yourself:
#     !PYTHONPATH=src python -m model.train
set -euo pipefail

echo "Installing dependencies (this can take a couple of minutes)..."
pip install -q -r requirements.txt

echo
echo "Setup complete. GPU visible to PyTorch:"
python -c "import torch; print('  cuda available:', torch.cuda.is_available())"

echo
echo "Next, run training with src on the import path:"
echo "    PYTHONPATH=src python -m model.train"
