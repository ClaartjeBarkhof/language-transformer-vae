import torch
import utils_train
import torch.multiprocessing as mp
from arguments import preprare_parser

import os
import datetime
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import wandb
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
# import NewsVAEArguments
from pytorch_lightning import seed_everything
from modules.vae import NewsVAE
from dataset_wrappper import NewsData
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from modules.decoder import DecoderNewsVAE
from modules.encoder import EncoderNewsVAE

def get_optimizer(vae_model, learning_rate=2e-5):
    """
    Return a simple AdamW optimiser.

    Args:
        vae_model: nn.module
        learning_rate: float

    Returns:
        optimizer: torch.optizer
    """

    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=learning_rate)

    return optimizer


def get_scheduler(optimizer, warmup_updates=1000, lr_scheduler=True):
    """
    Returns a learning rate scheduler with linear warm-up (if lr_scheduler).
    Otherwise it returns the unscaled learning rate.

    Args:
        optimizer: torch.optim
        warmup_updates: int:
            How many steps to warm-up (linearly) to the initial learning rate
        lr_scheduler: bool:
            Whether or not to actually apply the scheduler.
    Returns:
        scheduler: LambdaLR scheduler
            This is the learning rate scheduler that may be updated during training by calling .step()
    """
    def get_lr_scale(step):
        step += 1

        # If using a scheduler, calculate the scaling coefficient
        if lr_scheduler:
            # First linear warm-up
            if step <= warmup_updates:
                lr_scale = step / warmup_updates
            # Then square root decay
            else:
                lr_scale = 1 / np.sqrt(step - warmup_updates)

            return lr_scale

        # No scaling if no scheduler
        else:
            return 1.0

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=get_lr_scale
    )

    return scheduler


def get_dataloader(phases, ddp=False, batch_size=12, num_workers=8, debug_data=False,
                   debug_data_len=200, max_seq_len=64, world_size=4,
                   dataset_name="cnn_dailymail", tokenizer_name="roberta",
                   device_name="cuda:0", verbose=True, gpu_rank=0):
    """
    Get data loaders for distributed sampling for different phases (eg. train and validation).

    Args:
        phases: list:
            List of phases to make dataloaders for (train, validation, test)
        ddp: bool:
            Whether or not to use Distributed Data Parallel
        batch_size: int
        num_workers: int
        debug_data: bool
            Whether or not to use a fraction of the data for debugging
        debug_data_len: int:
            How many data points to use for debugging if debug mode is on.
        max_seq_len: int
        world_size: int
            Number of nodes x number of GPUs
        dataset_name: str:
            The name of the dataset ("cnn_dailymail")
        tokenizer_name: str:
            The name of the tokenizer ("roberta")
        device_name: str:
            Which device is active.
        verbose: bool:
            Whether or not to print.
        gpu_rank: int:
            If cuda, what the rank of the GPU is.

    Returns:
        loaders: dict[str, torch.utils.data.DataLoader]
            A dict with dataloaders for every named phase.
        data:
            NewsData object with the data # TODO: change this
        samplers: dict[str. torch.utils.data.distributed.DistributedSampler]
            A dict with samplers for the GPU (if ddp).
    """

    if verbose: print("Get dataloaders...")

    pin_memory = True if "cuda" in device_name else False

    loaders = {}
    samplers = {}

    # Get data
    data = NewsData(dataset_name, tokenizer_name,
                    batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, debug=debug_data,
                    debug_data_len=debug_data_len, max_seq_len=max_seq_len,
                    device=device_name)

    for phase in phases:
        # DDP
        if ddp:
            sampler = DistributedSampler(data.datasets[phase], num_replicas=world_size,
                                         shuffle=True, rank=gpu_rank)
            samplers[phase] = sampler

            # With distributed sampling, shuffle must be false
            loaders[phase] = DataLoader(data.datasets[phase], batch_size=batch_size,
                                        sampler=sampler, pin_memory=True,
                                        num_workers=num_workers, collate_fn=data.collate_fn)
        # Single GPU and CPU
        else:
            # Without distributed sampling
            loaders[phase] = DataLoader(data.datasets[phase], batch_size=batch_size,
                                        shuffle=True, pin_memory=pin_memory,
                                        num_workers=num_workers, collate_fn=data.collate_fn)

    if verbose: print("Done getting dataloaders...")

    return loaders, data, samplers


def get_model_on_device(device_name="cuda:0", latent_size=768, gradient_checkpointing=False,
                        add_latent_via_memory=True, add_latent_via_embeddings=True,
                        do_tie_weights=True, verbose=True):
    """
    Get a VAE model on correct device.

    Args:
        device_name: str
        latent_size: int
        gradient_checkpointing: bool
        add_latent_via_embeddings: bool
        add_latent_via_memory: bool
        do_tie_weights: bool
        verbose: bool:
    Returns:
        vae_model: NewsVAE object
    """

    if verbose: print("Loading model...")

    decoder = DecoderNewsVAE(gradient_checkpointing=gradient_checkpointing)
    encoder = EncoderNewsVAE(gradient_checkpointing=gradient_checkpointing, latent_size=latent_size)

    vae_model = NewsVAE(encoder, decoder, latent_size=latent_size,
                        add_latent_via_memory=add_latent_via_memory,
                        add_latent_via_embeddings=add_latent_via_embeddings,
                        do_tie_weights=do_tie_weights)

    vae_model = vae_model.to(device_name)

    if verbose: print("Done model...")

    return vae_model


def train(device_rank, config, run_name):
    # Device
    device_name = utils_train.set_device(device_rank)

    # Verbosity depending on device
    verbose = utils_train.set_verbosity(device_name)

    # Initiate process group and specify backend configurations
    if config.ddp:
        if verbose: print("Init process group...")
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=int(config.n_gpus * config.n_nodes), rank=device_rank)

    # Seed everything
    seed_everything(config.seed)

    # Model
    vae_model = get_model_on_device(device_name=device_name, latent_size=config.latent_size,
                                    gradient_checkpointing=config.gradient_checkpointing,
                                    add_latent_via_memory=config.add_latent_via_memory,
                                    add_latent_via_embeddings=config.add_latent_via_embeddings,
                                    do_tie_weights=config.do_tie_weights, verbose=verbose)

    # Logging
    if config.logging and verbose: utils_train.init_logging(vae_model, run_name, config)

    # Data loaders / data set / samplers (if ddp)
    data_loaders, data, samplers = get_dataloader(device_name=device_name, latent_size=config.latent_size,
                                                  gradient_checkpointing=config.gradient_checkpointing,
                                                  add_latent_via_memory=config.add_latent_via_memory,
                                                  add_latent_via_embeddings=config.add_latent_via_embeddings,
                                                  do_tie_weights=config.do_tie_weights, verbose=verbose)

    # Optimizer, learning rate scheduler and amp scaler
    optimizer = get_optimizer(vae_model, config.lr)
    scheduler = get_scheduler(optimizer, config.warmup_updates, config.lr_scheduler)
    scaler = GradScaler(enabled=config.use_amp)

    # Set-up DDP
    if config.ddp:
        vae_model = torch.nn.parallel.DistributedDataParallel(vae_model, device_ids=[device_rank])

    # Zero grads
    vae_model.zero_grad()

    # Initialise the stats to keep track of
    stats = utils_train.make_nested_dict()
    finished_training = False

    epoch, global_step, best_valid_loss = 0, 0, 1000
    max_train_steps_epoch, max_valid_steps_epoch = determine_max_epoch_steps(args, data)

    if args.load_from_checkpoint:
        optimizer, scheduler, VAE_model, scaler, global_step, epoch, best_valid_loss = load_from_checkpoint(VAE_model,
                                                                                                            args,
                                                                                                            optimizer=optimizer,
                                                                                                            scheduler=scheduler,
                                                                                                            scaler=scaler,
                                                                                                            args=args)

    if gpu_rank == 0: print("Start or resume training!")



def main(config):
    # Init folders & get unique run name
    run_name = utils_train.get_run_name(config.run_name_prefix)
    utils_train.prepare_folders(run_name, config.prefix_NewsVAE_path)

    # DDP
    if config.ddp:
        print("Using DDP")
        utils_train.set_DDP_environment_vars()
        mp.spawn(train, nprocs=int(config.n_gpus * config.n_nodes), args=(config, run_name))

    # Single GPU
    elif not config.ddp and config.use_gpu:
        print(f"Not using DDP, only using device: {torch.cuda.current_device()}")
        train(torch.cuda.current_device(), config, run_name)

    # CPU
    else:
        print("Using CPU, you're warned!")
        train("cpu", config, run_name)

if __name__ =="__main__":
    parsed_args = preprare_parser()
    main(parsed_args)