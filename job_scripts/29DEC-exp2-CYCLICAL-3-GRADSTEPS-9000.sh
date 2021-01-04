#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 14:00:00
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

python /home/cbarkhof/code-thesis/NewsVAE/train.py \
          --overwrite_args=False \
          \
          --run_name_prefix="29DEC-exp2-CYCLICAL-3-GRADSTEPS-9000" \
          \
          --batch_size=32 \
          --max_train_steps_epoch_per_rank=-1 \
          --max_valid_steps_epoch_per_rank=-1 \
          --max_global_train_steps=55000 \
          --accumulate_n_batches_grad=2 \
          \
          --lr=1.0 \
          --lr_scheduler=True \
          --lr_warmup_updates=5000 \
          \
          --gradient_checkpointing=False \
          --use_amp=True \
          --n_gpus=4 \
          --n_nodes=1 \
          --ddp=True \
          \
          --logging=True \
          --log_every_n_steps=50 \
          --wandb_project="thesis" \
          \
          --tokenizer_name="roberta" \
          --dataset_name="cnn_dailymail" \
          --num_workers=10 \
          --debug_data=False \
          --debug_data_len=10 \
          --max_seq_len=64 \
          \
          --print_stats=True \
          --print_every_n_steps=100 \
          \
          --checkpoint=True \
          --checkpoint_every_n_steps=1000 \
          --load_from_checkpoint=False \
          --checkpoint_file="/path/to/file.pth" \
          --continue_train_after_checkpoint_loading=False \
          \
          --seed=0 \
          --deterministic=True \
          \
          --hinge_loss_lambda=0.0 \
          --beta=1.0 \
          --KL_linear_annealing=False \
          --KL_annealing_grad_steps_linear=0 \
          --KL_cyclical_annealing=True \
          --KL_annealing_grad_steps_per_cycle=9000 \
          --objective="beta-vae" \
          --mmd_lambda=1000 \
          \
          --latent_size=768 \
          --add_latent_via_memory=True \
          --add_latent_via_embeddings=True \
          --do_tie_weights=True \
          --code_dir_path="/home/cbarkhof/code-thesis/NewsVAE"

