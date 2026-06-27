# Environments

MEG-Mamba uses the `mamba3` conda environment.

### mamba3

Build with [install_mamba3.sh](install_mamba3.sh):

```bash
bash envs/install_mamba3.sh  # creates the mamba3 env

conda activate mamba3
python -c 'from mamba_ssm import Mamba3; print("OK")'
```

Stack: PyTorch 2.11/cu128 + [mamba_ssm @ commit 316ed60](https://github.com/state-spaces/mamba/tree/316ed6036538405f767782132f76caf342256d33) + tilelang==0.1.8 + causal-conv1d. Ships its own CUDA 12.8 nvcc.

Also see [Kernel constraints](../README.md#kernel-constraints).

### Oxford BMRC Cluster

You do not need to do `module load CUDA` to use the `mamba3` environment.

In SLURM scripts on the BMRC cluster, use:

```bash
source /apps/eb/el8/2023a/skylake/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate mamba3
export PATH="$CONDA_PREFIX/bin:$PATH"
```
