#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0:50:00
#SBATCH --mem=30G

module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load CUDA/11.0.3-GCC-9.3.0
module load cuDNN/8.0.3.33-gcccuda-2020a

export PATH="/home/cbarkhof/code-thesis/Experimentation/Optimus/code:$PATH"
export PATH="/home/cbarkhof/code-thesis/Experimentation/Optimus/claartje:$PATH"
export PATH="/home/cbarkhof/code-thesis/Experimentation/Optimus/code/examples/big_ae:$PATH"

#source /home/cbarkhof/code-thesis/venv/bin/activate
python /home/cbarkhof/code-thesis/Experimentation/Optimus/claartje/autoregressive_versus_teacher_forcing.py

##Copy input file to scratch
#cp $HOME/big_input_file "$TMPDIR"
#
##Create output directory on scratch
#mkdir "$TMPDIR"/output_dir
#
##Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
#python $HOME/my_program.py "$TMPDIR"/big_input_file "$TMPDIR"/output_dir
#
##Copy output directory from scratch to home
#cp -r "$TMPDIR"/output_dir $HOME