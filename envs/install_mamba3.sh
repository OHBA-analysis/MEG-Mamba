#!/bin/bash
#
# Build the `mamba3` env.
#
# PyTorch 2.11.0 (cu128) + CUDA 12.8 nvcc + mamba_ssm (pinned commit) + causal-conv1d.
# TileLang JITs the MIMO kernel for A100 (sm_80) and L4 (sm_89).
#
# mamba_ssm is pinned to commit 316ed60 (Apr 2026; reports version 2.3.1).
# Later commits regressed the NON-varlen MIMO kernel latency ~2.7x (measured A/B
# on identical hardware): #937 (May) made batch/heads/groups dynamic TileLang dims,
# and #962 (Jun, "heavy-tail A") added per-step A compute. #965 only fixed the
# *varlen* kernels (not the path we use, cu_seqlens=None). 316ed60 predates all
# of them, so the kernels are statically specialized and fast.

set -eo pipefail
eval "$(conda shell.bash hook)"

# Minimal base — building from a conda-forge environment.yml breaks torch._dynamo.
conda create -y -n mamba3 python=3.10 pip
conda activate mamba3
conda install -y -c nvidia/label/cuda-12.8.0 cuda-nvcc cuda-cudart-dev

# torch hard-pinned to 2.11.0 (PIP_CONSTRAINT, every pip call): 2.12.0's
# torch._dynamo is broken here (import Mamba3 fails with NP_SUPPORTED_MODULES).
# tilelang pinned to 0.1.8: main may declare a newer tilelang whose bundled CUDA
# headers churn (see the README "Kernel constraints"); 0.1.8 is the known-good kernel JIT.
CONSTRAINTS=$(mktemp)
printf 'torch==2.11.0\ntransformers==5.5.4\ntilelang==0.1.8\n' > "$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"
export TORCH_CUDA_ARCH_LIST="8.0 8.9"

# +cu128 wheel so torch's CUDA major matches the 12.8 nvcc at build time.
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install ninja setuptools packaging numpy joblib threadpoolctl
MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation --no-cache-dir \
    "git+https://github.com/state-spaces/mamba.git@316ed60"

# mamba_ssm drifts torch to +cu130; force it back to +cu128 so causal-conv1d's
# torch-extension build matches the 12.8 nvcc. --no-deps keeps quack's cuda-bindings.
pip install --force-reinstall --no-deps torch==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation --no-deps \
    --no-cache-dir causal-conv1d

unset PIP_CONSTRAINT; rm -f "$CONSTRAINTS"

python -c 'import torch, mamba_ssm, causal_conv1d; from mamba_ssm import Mamba3; \
print("mamba3 OK |", torch.__version__, "| mamba_ssm", mamba_ssm.__version__)'
echo "Done: conda activate mamba3"
