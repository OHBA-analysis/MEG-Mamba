#!/bin/bash
#
# Train MEG-Mamba for N epochs on the 24h A100 queue, saving a
# snapshot every epoch (--snapshot-every 1 -> snapshot_e1, _e2, …).
#
#   sbatch slurm/train.slurm [EPOCHS=3] [outdir=model] [seed=42]
#
#SBATCH --job-name=mm-train
#SBATCH --partition=gpu_a100_80gb
#SBATCH --account=gpu_woolrich.prj
#SBATCH --qos=gpu_bmrc_24hr
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/train-%j.out
#SBATCH --error=slurm/logs/train-%j.err
#SBATCH --mail-type=END,FAIL

set -eo pipefail
EPOCHS="${1:-3}"
OUTDIR="${2:-model}"
SEED="${3:-42}"

source /apps/eb/el8/2023a/skylake/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate mamba3
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd /gpfs3/well/woolrich/users/wlo995/packages/MEG-Mamba

echo "Node: $(hostname) | Env: $CONDA_DEFAULT_ENV | outdir: $OUTDIR | epochs: $EPOCHS | seed: $SEED | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train.py \
    --manifest data/manifest.pt \
    --output-dir "$OUTDIR" \
    --session-features data/session_features.npz --holdout-sessions test \
    --d-model 256 --n-layers 4 --d-state 64 --headdim 64 --expand 2 \
    --mimo-rank 4 \
    --seq-len 1000 --loss-start 50 \
    --batch-size 512 \
    --lr 1e-4 --warmup-steps 2000 --weight-decay 0.01 --grad-clip 1.0 \
    --seed "$SEED" \
    --epochs "$EPOCHS" --snapshot-every 1 --resume
