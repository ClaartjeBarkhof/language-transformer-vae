#!/bin/bash
#SBATCH -p gpu_short
#SBATCH -t 0:50:00
#SBATCH --mail-type=END
#SBATCH --mail-user=claartjebarkhof@hotmail.com

module load python/3.6.0
source /home/cbarkhof/venv/bin/activate