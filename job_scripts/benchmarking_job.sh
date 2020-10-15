#!/bin/bash
#SBATCH -p gpu_short
#SBATCH -t 0:50:00
#SBATCH --mem=60G

module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load CUDA/11.0.3-GCC-9.3.0
module load cuDNN/8.0.3.33-gcccuda-2020a

#export PATH="/home/cbarkhof/code-thesis/Experimentation/Optimus/code:$PATH"
#export PATH="/home/cbarkhof/code-thesis/Experimentation/Optimus/claartje:$PATH"
#export PATH="/home/cbarkhof/code-thesis/Experimentation/Optimus/code/examples/big_ae:$PATH"

source /home/cbarkhof/code-thesis/venv/bin/activate
python /home/cbarkhof/code-thesis/Experimentation/Benchmarking/benchmarking-models-claartje.py