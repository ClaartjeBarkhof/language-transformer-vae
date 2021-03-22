import torch.multiprocessing as mp
import argparse
import sys;

sys.path.append("/home/cbarkhof/code-thesis/NewsVAE")
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
from utils_train import set_ddp_environment_vars, load_from_checkpoint
from utils import load_pickle, dump_pickle
import distutils


def combine_result_files(result_dir_path, run_name, max_batches, batch_size, n_samples):
    all_res = dict()

    # write away the accumulated results
    result_dir = Path(result_dir_path)

    # accumulate all results from cuda 0 to 3
    for f in os.listdir(result_dir_path):
        if "cuda" in f and run_name in f:
            r = load_pickle(str(result_dir / f))
            for k, v in r.items():
                if k in all_res:
                    all_res[k].append(v)
                else:
                    all_res[k] = [v]

    # cat all tensors in list
    for k, v in all_res.items():
        all_res[k] = torch.cat(v)

    result_file = result_dir / f"{run_name}_max_batches_{max_batches}_batch_size_{batch_size}_n_samples_{n_samples}.pickle"
    dump_pickle(all_res, result_file)


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


def evaluation_function(device_rank, run_name, model_path, max_batches,
                        result_dir_path, batch_size, dataset_name, objective,
                        world_size, num_workers):
    # Prepare some variables & result directory
    device_name = f"cuda:{device_rank}"
    latent_size = 32 if "latent32" in model_path else 64
    result_dir = Path(result_dir_path)
    os.makedirs(result_dir, exist_ok=True)

    result_file = result_dir / f"{device_name}_{run_name}_max_batches_{max_batches}.pickle"

    if os.path.isfile(result_file):

        print('_' * 80)
        print('_' * 80)
        print("Have done this one already!")
        print('_' * 80)
        print('_' * 80)

    else:

        print("-" * 30)
        print("run_name:", run_name)
        print("batch size:", batch_size)
        print("max_batches:", max_batches)
        print("latent size:", latent_size)
        print("device name:", device_name)
        print("-" * 30)

        # Get model
        # vae_model = #(path=model_path, device_name=device_name)
        vae_model = load_from_checkpoint(path=model_path, device_name=device_name, latent_size=latent_size,
                                         do_tie_embedding_spaces=True,
                                         add_decoder_output_embedding_bias=False, do_tie_weights=True,
                                         add_latent_via_embeddings=False,
                                         add_latent_via_memory=True, objective=objective, evaluation=True)
        vae_model = vae_model.to(device_name)

        # Get distributed validation data loader of PTB data set
        loader = get_dist_validation_loader(batch_size=batch_size, num_workers=num_workers, max_seq_len=64,
                                            world_size=world_size, dataset_name=dataset_name, tokenizer_name="roberta",
                                            device_name=device_name, gpu_rank=device_rank)

        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=device_rank)

        # Seed everything
        seed_everything(0)

        print(f"Len data loader on device {device_name}: {len(loader)}")
        N = max_batches if max_batches > 0 else len(loader)

        results = {}

        for batch_i, batch in enumerate(loader):
            print(f"{batch_i:3d}/{N} - {device_name}")
            batch = transfer_batch_to_device(batch, device_name=device_name)

            with torch.no_grad():
                out = vae_model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                auto_regressive=False,

                                return_latents=False,
                                return_mu_logvar=False,

                                return_exact_match=True,
                                return_cross_entropy=True,
                                return_reconstruction_loss=True,

                                return_posterior_stats=True,

                                reduce_seq_dim_ce="mean",
                                reduce_seq_dim_exact_match="mean",
                                reduce_batch_dim_exact_match="mean",
                                reduce_batch_dim_ce="mean")

                for k, v in out.items():
                    if k not in results:
                        if torch.is_tensor(v):
                            results[k] = [v.item()]
                        else:
                            results[k] = [v]
                    else:
                        if torch.is_tensor(v):
                            results[k].append(v.item())
                        else:
                            results[k].append(v)

                if batch_i + 1 == max_batches:
                    break

            # Dump the results for this device
            pickle.dump(results, open(result_file, "wb"))

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True, type=str,
                        help="Path to the model needed to evaluate it.")
    parser.add_argument("--result_dir_path", required=False, type=str,
                        default="/home/cbarkhof/code-thesis/NewsVAE/evaluation/result-files/beta-tc-vae-stats-older-runs",
                        help="Path to directory to store the results.")
    parser.add_argument("--dataset_name", required=False, type=str,
                        default="ptb_text_only", help="The name of the dataset (default: ptb_text_only).")
    parser.add_argument("--batch_size", required=False, type=int, default=3,
                        help="Batch size (default: 64).")
    parser.add_argument("--max_batches", required=False, type=int, default=1,
                        help="Maximum validation batches to process (per GPU!) (default: -1, means all).")
    parser.add_argument("--with_grad", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to enable gradients (default: False)")
    parser.add_argument("--num_workers", required=False, type=int, default=2,
                        help="Num workers for data loading (default: 8).")
    parser.add_argument("--world_size", required=False, type=int, default=2,
                        help="Number of GPUs to use (default: 4).")
    parser.add_argument("--objective", required=False, type=str,
                        default="beta-tc-vae",
                        help="Which objective to use.")

    config = parser.parse_args()

    return config


def main(config):
    # INIT DDP
    # print(f"*** Using DDP, spawing 4 processes")
    set_ddp_environment_vars(port_nr=1236)
    seed_everything(0)
    run_name = config.model_path.split("/")[-2]
    mp.spawn(evaluation_function, nprocs=4, args=(run_name, config.model_path, config.max_batches,
                                                  config.result_dir_path, config.batch_size, config.dataset_name, config.objective,
                                                  config.world_size, config.num_workers))
    combine_result_files(config.result_dir_path, run_name, config.max_batches, config.batch_size, config.n_samples)


if __name__ == "__main__":
    args = get_config()
    main(args)
