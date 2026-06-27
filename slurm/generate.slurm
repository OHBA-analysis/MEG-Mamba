#!/bin/bash
#
# Zero-shot generation for the held-out sessions on the L4 (Ada, sm_89) node.
#
# Usage:
#   sbatch slurm/generate.slurm \
#       --checkpoint model/snapshot_e2.pt \
#       --output-dir ~/projects/meg-mamba/analysis/generated \
#       --session-batch 4 --n-generate 15000
#
#SBATCH --job-name=mm-gen
#SBATCH --partition=gpu_interactive
#SBATCH --account=gpu_woolrich.prj
#SBATCH --nodelist=compg047
#SBATCH --gres=gpu:l4-24gb:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/gen-%j.out
#SBATCH --error=slurm/logs/gen-%j.err
#SBATCH --mail-type=END,FAIL

set -eo pipefail

source /apps/eb/el8/2023a/skylake/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate mamba3
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /gpfs3/well/woolrich/users/wlo995/packages/MEG-Mamba

echo "Node: $(hostname) | Env: $CONDA_DEFAULT_ENV | nvcc: $(which nvcc) | Time: $(date)"
nvcc --version | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python generate.py "$@"
