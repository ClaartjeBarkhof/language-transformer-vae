import torch.multiprocessing as mp
import argparse
import sys
sys.path.append("/home/cbarkhof/code-thesis/NewsVAE")
from dataset_wrappper import NewsData
from pytorch_lightning import seed_everything
from pathlib import Path
import torch
import pickle
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from utils_train import set_ddp_environment_vars, load_from_checkpoint
import distutils
from utils_evaluation import iw_log_p_x_generated, iw_log_p_x_dataset, dump_pickle, load_pickle

# only target rate of 0.5 per dimension
# only drop-out rates of 0.0, 0.4, 0.8
RUNS_TO_EVALUATE = [
    '2021-03-24-24MAR_AE-run-15:54:46', # AE
    '2021-03-24-24MAR_BetaVAE_FB_0.5-run-12:32:07', # FB-0.5
    '2021-03-25-25MAR_BetaVAE_MDR_0.5_checkpointing-run-14:44:42', # MDR-0.5
    '2021-03-24-24MAR_HoffmanTest-run-14:34:49', # VAE
    '2021-03-24-24MAR_Hoffman_Test_ConstraintOptim-run-15:08:49',  # Hoffman16
    '2021-03-30-30MAR_BetaVAE_FB_0.5_DropOut0.4-run-09:58:10',
    '2021-03-30-30MAR_BetaVAE_MDR_0.5_DropOut0.4-run-09:58:09',
    '2021-03-30-30MAR_BetaVAE_FB_0.0_DropOut0.4-run-18:16:23', # VAE - DROP-0.4
    '2021-04-02-2APR_EmbedOnly_BetaVAE_FB_0.5-run-15:27:53', # FB-0.5 | EMBED
    '2021-04-02-2APR_EmbedOnly_VAE_FB_0.0-run-17:49:57', # VAE-0.5 | EMBED
    '2021-04-02-2APR_LongTestCross_BetaVAE_FB_0.5-run-20:05:16', # FB-0.5 | CROSS
    '2021-04-02-2APR_MemOnly_BetaVAE_FB_0.5-run-16:59:34', # FB-0.5 | MEM
    '2021-04-05-EXP - CROSS - FB-0.5 - DROP-0.8-run-18:01:02', # FB-0.5 | CROSS
    '2021-04-05-EXP - CROSS - VAE - DROP-0.4-run-18:00:31', # VAE | CROSS | DROP-0.4
    '2021-04-05-EXP - CROSS - VAE - DROP-0.8-run-18:01:02', # VAE | CROSS | DROP-0.8
    '2021-04-05-EXP - MEM - FB-0.5 - DROP-0.8-run-18:01:32', # FB-0.5 | MEM  | DROP-0.8
    '2021-04-05-EXP - MEM+EMB - FB-0.5 - DROP-0.8-run-18:02:02', # FB-0.5 | MEM+EMB | DROP-0.8
    '2021-04-05-EXP - MEM+EMB - VAE - DROP-0.8-run-19:42:57', # VAE | MEM+EMB | DROP-0.8
    '2021-04-06-EXP - MDR-0.5 - CROSS - DROP-0.4-run-11:02:41', # MDR-0.5 | CROSS | DROP-0.4
    '2021-04-06-EXP - MDR-0.5 - CROSS - DROP-0.8-run-11:02:37', # MDR-0.5 | CROSS | DROP-0.8
    '2021-04-06-EXP - MDR-0.5 - MEM-EMB - DROP-0.8-run-11:03:21', # MDR-0.5 | MEM+EMB | DROP-0.8
]

def combine_results_N_gpus(result_dir_path, run_name, max_batches, batch_size, n_samples, world_size, train_validation):
    result_dir = Path(result_dir_path) / run_name
    all_res = dict()

    for i in range(world_size):
        device_name = f"cuda:{i}"

        result_file = result_dir / f"{device_name}_{run_name}_max_batches_{max_batches}_{train_validation}.pickle"
        res = load_pickle(result_file)
        for k, v in res.items():
            if k in all_res:
                all_res[k].append(v)
            else:
                all_res[k] = [v]

    # cat all tensors in list
    for k, v in all_res.items():
        all_res[k] = torch.cat(v)

    result_file = result_dir / f"{run_name}_world_size_{world_size}_max_batches_{max_batches}_" \
                               f"batch_size_{batch_size}_n_samples_{n_samples}_{train_validation}.pickle"
    dump_pickle(all_res, result_file)

    for f in os.listdir(result_dir):
        if "cuda" in str(f):
            os.remove(result_dir / f)

def get_dist_validation_loader(batch_size=12, num_workers=8, max_seq_len=64, world_size=4,
                               dataset_name="ptb_text_only", tokenizer_name="roberta",
                               device_name="cuda:0", gpu_rank=0, train_validation="validation"):
    # Get data
    data = NewsData(dataset_name, tokenizer_name,
                    batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, max_seq_len=max_seq_len,
                    device=device_name)

    # Get distributed sampler
    sampler = DistributedSampler(data.datasets[train_validation], num_replicas=world_size,
                                 shuffle=True, rank=gpu_rank)

    # Get data loader
    loader = DataLoader(data.datasets[train_validation], batch_size=batch_size,
                        sampler=sampler, pin_memory=True, shuffle=False,
                        num_workers=num_workers, collate_fn=data.collate_fn)

    return loader


def dist_iw_log_likelihood_x_obs_x_gen(device_rank, run_name, model_path, max_batches,
                               result_dir_path, batch_size, dataset_name,
                               world_size, num_workers, n_samples, n_chunks, max_seq_len_gen, train_validation):
    # Prepare some variables & result directory
    device_name = f"cuda:{device_rank}"

    result_dir = Path(result_dir_path) / run_name
    os.makedirs(result_dir, exist_ok=True)

    result_file = result_dir / f"{device_name}_{run_name}_max_batches_{max_batches}_{train_validation}.pickle"

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
        print("device name:", device_name)
        print("-" * 30)

        # Get model
        vae_model = load_from_checkpoint(model_path, world_master=True, ddp=False, device_name=device_name,
                                        evaluation=True, return_loss_term_manager=False)

        # Get distributed validation data loader of PTB data set
        loader = get_dist_validation_loader(batch_size=batch_size, num_workers=num_workers, max_seq_len=64,
                                            world_size=world_size, dataset_name=dataset_name, tokenizer_name="roberta",
                                            device_name=device_name, gpu_rank=device_rank,
                                            train_validation=train_validation)

        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=device_rank)

        # Seed everything
        seed_everything(0)

        print(f"Len data loader on device {device_name}: {len(loader)}")
        N = max_batches if max_batches > 0 else len(loader)

        with torch.no_grad():
            log_p_x_obs, log_p_x_w_obs, lens_obs = iw_log_p_x_dataset(loader, model=vae_model, path=None,
                                                                      n_samples=n_samples, n_chunks=n_chunks,
                                                                      verbose=True, ddp=False, device_name=device_name,
                                                                      max_batches=N)

            log_p_x_gen, log_p_x_w_gen, lens_gen = iw_log_p_x_generated(model=vae_model, path=None, n_batches=N,
                                                                        batch_size=batch_size, n_samples=n_samples,
                                                                        n_chunks=n_chunks,
                                                                        verbose=True, ddp=False, device_name=device_name,
                                                                        max_seq_len_gen=max_seq_len_gen)

        results = dict(
            log_p_x_obs=log_p_x_obs.cpu(),
            log_p_x_w_obs=log_p_x_w_obs.cpu(),
            lens_obs=lens_obs.cpu(),
            log_p_x_gen=log_p_x_gen.cpu(),
            log_p_x_w_gen=log_p_x_w_gen.cpu(),
            lens_gen=lens_gen.cpu()
        )

        # Dump the results for this device
        pickle.dump(results, open(result_file, "wb"))

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_i", required=False, type=int, default=-1,
                        help="Which i of RUNS_TO_EVALUATE to evaluate.")
    parser.add_argument("--model_path", required=False, type=str, default='dummy.pth',
                        help="Path to the model needed to evaluate it.")
    parser.add_argument("--result_dir_path", required=False, type=str,
                        default="/home/cbarkhof/code-thesis/NewsVAE/iw_log_likelihood_x_obs_x_gen",
                        help="Path to directory to store the results.")
    parser.add_argument("--dataset_name", required=False, type=str,
                        default="ptb_text_only", help="The name of the dataset (default: ptb_text_only).")
    parser.add_argument("--batch_size", required=False, type=int, default=64,
                        help="Batch size (default: 64).")
    parser.add_argument("--max_batches", required=False, type=int, default=4,
                        help="Maximum validation batches to process (per GPU!) (default: -1, means all).")
    parser.add_argument("--num_workers", required=False, type=int, default=8,
                        help="Num workers for data loading (default: 8).")
    parser.add_argument("--world_size", required=False, type=int, default=4,
                        help="Number of GPUs to use (default: 4).")
    parser.add_argument("--n_chunks", required=False, type=int, default=3,
                        help="Number of chunks to divide the samples from posterior "
                             "in for importance weighting (default: 2).")
    parser.add_argument("--n_samples", required=False, type=int, default=600,
                        help="Number of samples to take per posterior or data point (default: 600).")
    parser.add_argument("--max_seq_len_gen", required=False, type=int, default=64,
                        help="Maximum sequence length for ancestral sampling generation (default: 64).")
    parser.add_argument("--train_validation", required=False, type=str, default='validation',
                        help="Whether to use the train or the validation set to evaluate on (default: validation).")

    config = parser.parse_args()

    return config

def main(config):
    # INIT DDP
    set_ddp_environment_vars(port_nr=1236)
    seed_everything(0)

    # array job on RUNS_TO_EVALUATE
    if config.run_i > -1:
        index = config.run_i - 1
        run_name = RUNS_TO_EVALUATE[index]
        model_path = "/home/cbarkhof/code-thesis/NewsVAE/Runs/" + run_name + "/checkpoint-best.pth"
    # single model evaluation with config.model_path
    else:
        model_path = config.model_path
        run_name = config.model_path.split("/")[-2]

    assert os.path.isfile(model_path), "give a valid model path, not valid: {}".format(model_path)
    assert config.train_validation in ["train", "validation"], \
        "enter a valid mode, currently {}, must be in ['train', 'validation']".format(config.train_validation)

    mp.spawn(dist_iw_log_likelihood_x_obs_x_gen, nprocs=config.world_size,
             args=(run_name, model_path, config.max_batches, config.result_dir_path,
                   config.batch_size, config.dataset_name, config.world_size, config.num_workers,
                   config.n_samples, config.n_chunks, config.max_seq_len_gen, config.train_validation))
    combine_results_N_gpus(config.result_dir_path, run_name, config.max_batches, config.batch_size, config.n_samples,
                           config.world_size, config.train_validation)


if __name__ == "__main__":
    args = get_config()
    main(args)
