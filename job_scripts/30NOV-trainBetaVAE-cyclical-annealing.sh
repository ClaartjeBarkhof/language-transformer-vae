#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 40:00:00
#SBATCH -c 12
#SBATCH --output /home/cbarkhof/slurm-logs/%j-slurm-log.out

module purge  # unload all that are active
module load 2019  # load 2019 software module for good python versions
module load Anaconda3  # load anacoda
module load CUDA/10.0.130  # load cuda
module load cuDNN/7.6.3-CUDA-10.0.130  # load cudnn
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.6.3-CUDA-10.0.130/lib64:$LD_LIBRARY_PATH

CONDA_PREFIX=$(conda info --base)
source $CONDA_PREFIX/etc/profile.d/conda.sh

conda deactivate # just to make sure other envs are not active
conda activate thesisenv # activate environment

python /home/cbarkhof/code-thesis/NewsVAE/trainNewsVAE.py \
          --overwrite_args=False \
          \
          --run_name_prefix="30NOV-BETA-VAE-Cylical-annealing" \
          \
          --objective="beta-vae" \
          --KL_annealing_steps=33333 \
          --KL_annealing=True \
          --beta=1.0 \
          --mmd_lambda=10000. \
          --hinge_loss_lambda=0.5 \
          \
          --max_train_steps_epoch=5000 \
          --max_valid_steps_epoch=-1 \
          --max_global_train_steps=100000 \
          \
          --batch_size=32 \
          --accumulate_n_batches_grad=2 \
          \
          --lr=2e-5 \
          --lr_scheduler=False \
          \
          --gradient_checkpointing=False \
          --use_amp=True \
          --n_gpus=4 \
          --ddp=True \
          --n_nodes=1 \
          \
          --logging=True \
          --log_every_n_steps=20 \
          --print_stats=True \
          --print_every_n_steps=100 \
          --time_batch=False \
          --checkpoint=True \
          --checkpoint_every_n_steps=1000 \
          --load_from_checkpoint=False \
          \
          --num_workers=8 \
          --debug_data=False \
          --max_seq_len=64 \
          \
          --seed=1 \
          --deterministic=True \
          \
          --evaluate_every_n_epochs=1 \
          --n_batches_to_evaluate_on=24 \
          --iw_nsamples_evaluation=300 \
          \
          --do_tie_weights=True