#!/usr/bin/env bash
# Creates the "ppo_rl" conda env and installs everything the config files and the tutorial notebooks need.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
eval "$(conda shell.bash hook)"
if ! conda env list | grep -qE '^ppo_rl\s'; then
  conda create -n ppo_rl python=3.10 -y
fi
conda activate ppo_rl
python -m pip install --upgrade pip
python -m pip install "setuptools<=66.1.1" wheel
# PyTorch with CUDA 12.8 (required for RTX 50-series GPUs; falls back to CPU if no GPU)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$HERE/requirements.txt"
python -m ipykernel install --user --name ppo_rl --display-name "Python (ppo_rl)"
python - <<'PY'
import torch, gym, ding, easydict, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gym", gym.__version__, "DI-engine", ding.__version__, "numpy", numpy.__version__)
PY
