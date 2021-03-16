import torch.multiprocessing as mp
import argparse
import sys; sys.path.append("/home/cbarkhof/code-thesis/NewsVAE")
from dataset_wrappper import NewsData
from utils_train import transfer_batch_to_device
from pytorch_lightning import seed_everything
from pathlib import Path
import torch
import numpy as np
import pickle
import os
import copy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from utils_train import set_ddp_environment_vars
from utils import load_model_for_eval, load_pickle, dump_pickle


def combine_result_files(result_dir):
    # TODO: write combine results function

    # either combine per run separately, or all runs from all GPUs together

    for f in result_dir:
        if "cuda" in f:


def get_dist_validation_loader(batch_size=12, num_workers=8, max_seq_len=64, world_size=4,
                               dataset_name="cnn_dailymail", tokenizer_name="roberta",
                               device_name="cuda:0", gpu_rank=0):
    # Get data
    data = NewsData(dataset_name, tokenizer_name,
                    batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, max_seq_len=max_seq_len,
                    device=device_name)

    # Get distributed sampler
    sampler = DistributedSampler(data.datasets["validation"], num_replicas=world_size,
                                 shuffle=True, rank=gpu_rank)

    # Get data loader
    loader = DataLoader(data.datasets["validation"], batch_size=batch_size,
                        sampler=sampler, pin_memory=True, shuffle=False,
                        num_workers=num_workers, collate_fn=data.collate_fn)

    return loader


def evaluation_function(device_rank, config):

    # Prepare some variables & result directory
    device_name = f"cuda:{device_rank}"
    latent_size = 32 if "latent32" in config.model_path else 64
    result_dir = Path()
    os.makedirs(result_dir, exist_ok=True)
    run_name = config.model_path.split("/")[-2]
    result_file = result_dir / f"{device_name}_{run_name}_max_batches_{config.max_batches}.pickle"

    if os.path.isfile(result_file):

        print('_' * 80)
        print('_' * 80)
        print("Have done this one already!")
        print('_' * 80)
        print('_' * 80)

    else:

        print("-" * 30)
        print("run_name:", run_name)
        print("batch size:", config.batch_size)
        print("max_batches:", config.max_batches)
        print("latent size:", latent_size)
        print("device name:", device_name)
        print("-" * 30)

        # Get model
        vae_model = load_model_for_eval(run_name=run_name, path=config.model_path)

        # Get distributed validation data loader of PTB data set
        loader = get_dist_validation_loader(batch_size=config.batch_size, num_workers=2, max_seq_len=64,
                                            world_size=4, dataset_name=config.dataset_name, tokenizer_name="roberta",
                                            device_name=device_name, gpu_rank=device_rank)

        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=4, rank=device_rank)

        # Seed everything
        seed_everything(0)

        print(f"Len data loader on device {device_name}: {len(loader)}")
        N = config.max_batches if config.max_batches > 0 else len(loader)

        # TODO: initialise result dict or list
        result = {
            "key": []
        }

        # For all batches in the validation set
        for batch_i, batch in enumerate(loader):
            print(f"{batch_i:3d}/{N}")



            results = dict(...)

            # Dump the results for this device
            pickle.dump(results, open(result_file, "wb"))


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True, type=str,
                        help="Path to the model needed to calculate importance weighted PPL (iw ppl).")
    parser.add_argument("--result_dir_path", required=False, type=str,
                        default="/home/cbarkhof/code-thesis/NewsVAE/evaluation/result-files/result_dir",
                        help="Path to directory to store the results.")
    parser.add_argument("--dataset_name", required=False, type=str,
                        default="ptb_text_only",
                        help="The name of the dataset (default: ptb_text_only).")
    parser.add_argument("--batch_size", required=False, type=int, default=64,
                        help="Batch size (default: 64).")
    parser.add_argument("--max_batches", required=False, type=int, default=-1,
                        help="Maximum validation batches to process (default: -1, means all).")

    # TODO: Add other options here

    config = parser.parse_args()

    return config


def main(config):
    # INIT DDP
    print(f"*** Using DDP, spawing 4 processes")
    set_ddp_environment_vars(port_nr=1235)
    seed_everything(0)
    mp.spawn(evaluation_function, nprocs=4, args=(config))
    combine_result_files(config.result_dir)

if __name__ == "__main__":
    args = get_config()
    main(args)
