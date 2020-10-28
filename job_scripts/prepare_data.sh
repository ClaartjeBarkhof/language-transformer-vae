#!/bin/bash
#SBATCH -p gpu_short
#SBATCH -t 0:50:00
#SBATCH --output /home/cbarkhof/slurm-logs/%j-slurm-log.out
#SBATCH --mem 20G

echo $HOME

module purge  # unload all that are active
module load 2019  # load 2019 software module for good python versions
module load Anaconda3  # load anacoda
module load CUDA/10.0.130  # load cuda
module load cuDNN/7.6.3-CUDA-10.0.130  # load cudnn
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.6.3-CUDA-10.0.130/lib64:$LD_LIBRARY_PATH

CONDA_PREFIX=$(conda info --base)
source $CONDA_PREFIX/etc/profile.d/conda.sh

conda deactivate # just to make sure other envs are not active
conda activate thesisenv # activat environment

python /home/cbarkhof/code-thesis/NewsVAE/NewsData.py