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
from loss_and_optimisation import sample_log_likelihood


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


def get_eos_mask(predictions, eos_token_id=2):
    # Make a tensor with position indices per row
    ind = torch.arange(predictions.shape[1]).repeat(predictions.shape[0], 1)

    # Check where the eos_token is in the predictions, if not there set to max_len
    lens = torch.tensor(
        [a.index(eos_token_id) if eos_token_id in a else len(a) for a in predictions.tolist()]).unsqueeze(1)

    # Mask everything after the eos_token_id (set to 0.0)
    mask = (ind <= lens).float()

    return mask


def get_sampled_log_probs_sequence(vae_model, z, max_seq_len, device_name):
    # Auto-regressive decoding of posterior samples
    dec_out = vae_model.decoder.autoregressive_decode(z,
                                                      max_seq_len=max_seq_len,
                                                      device_name=device_name,
                                                      labels=None,
                                                      return_cross_entropy=False,
                                                      nucleus_sampling=True,
                                                      top_k=0, top_p=1.0,
                                                      return_logits=False,
                                                      return_predictions=True,
                                                      return_log_probs=True)

    preds_log_probs = dec_out["log_probs"].cpu()
    mask = get_eos_mask(dec_out["predictions"], eos_token_id=2)
    preds_log_probs = preds_log_probs * mask  # mask everything after the eos token
    preds_log_probs_seq = preds_log_probs.sum(dim=-1)

    return preds_log_probs_seq


def batch_log_x_gen_vs_log_x_obs(vae_model, batch, n_samples, latent_size, batch_size, device_name, n_chunks,
                                 max_seq_len):
    # Encode these input ids and sample <n_samples> for each x
    enc_out = vae_model.encoder.encode(batch["input_ids"], batch["attention_mask"],
                                       n_samples=n_samples,
                                       return_log_q_z_x=True,
                                       return_log_q_z=False,
                                       return_log_p_z=True,
                                       return_embeddings=False)

    # Unpack the tensors we need
    # [batch, n_samples, latent_dim], [batch, n_samples], [batch, n_samples]
    post_samples, post_log_p_z, post_log_q_z_x = enc_out["latent_z"], \
                                                 enc_out["log_p_z"], enc_out["log_q_z_x"]

    # [n_samples, latent_dim]
    prior_samples = vae_model.sample_from_prior(latent_size=latent_size, n_samples=n_samples * batch_size,
                                                device_name=device_name)
    prior_samples = prior_samples.reshape(batch_size, n_samples, -1)
    prior_log_p_z = sample_log_likelihood(prior_samples, reduce_batch_dim=False, reduce_latent_dim=True)

    # Now we need to loop again because our batch size was multiplied by n_samples
    post_log_p_x_z = []
    prior_log_p_x_z = []

    # For all samples x in batch
    for sample_i in range(batch_size):
        print(f"sample i: {sample_i:3d}")

        post_samples_i = post_samples[sample_i, :, :]
        prior_samples_i = prior_samples[sample_i, :, :]

        post_samples_i_chunked = list(torch.chunk(post_samples_i, n_chunks, dim=0))
        prior_samples_i_chunked = list(torch.chunk(prior_samples_i, n_chunks, dim=0))

        for (post_z, prior_z) in zip(post_samples_i_chunked, prior_samples_i_chunked):
            prior_preds_log_probs = get_sampled_log_probs_sequence(vae_model, prior_z, max_seq_len,
                                                                   device_name=device_name)
            prior_log_p_x_z.append(prior_preds_log_probs)
            post_preds_log_probs = get_sampled_log_probs_sequence(vae_model, post_z, max_seq_len,
                                                                  device_name=device_name)
            post_log_p_x_z.append(post_preds_log_probs)

    post_log_p_x_z = torch.cat(post_log_p_x_z).reshape(batch_size, n_samples)
    prior_log_p_x_z = torch.cat(prior_log_p_x_z).reshape(batch_size, n_samples)

    post_frac = post_log_p_x_z.cpu() + post_log_p_z.cpu() - post_log_q_z_x.cpu()
    prior_frac = prior_log_p_z.cpu() + prior_log_p_x_z.cpu()

    post_likelihood = torch.logsumexp(post_frac, dim=-1) - np.log(n_samples)
    prior_likelihood = torch.logsumexp(prior_frac, dim=-1) - np.log(n_samples)

    return post_likelihood, prior_likelihood


def evaluation_function(device_rank, run_name, model_path, max_batches, max_seq_len, with_grad,
                        result_dir_path, batch_size, dataset_name, n_samples, n_chunks,
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
        #vae_model = #(path=model_path, device_name=device_name)
        vae_model = load_from_checkpoint(path=model_path, device_name=device_name, latent_size=latent_size, do_tie_embedding_spaces=True,
                                         add_decoder_output_embedding_bias=False, do_tie_weights=True, add_latent_via_embeddings=False,
                                         add_latent_via_memory=True, objective="vae", evaluation=True)
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

        log_likelihood_res = {
            "log_like_prior_generation": [],
            "log_like_posterior_generation": []
        }

        with torch.set_grad_enabled(with_grad):
            # For all batches in the validation set
            for batch_i, batch in enumerate(loader):
                print(f"{batch_i:3d}/{N} - {device_name}")

                # Send to right device
                batch = transfer_batch_to_device(batch, device_name)
                batch_size = batch["input_ids"].shape[0]

                # Execute the actual function on a batch
                post_likelihood, prior_likelihood = batch_log_x_gen_vs_log_x_obs(vae_model, batch, n_samples,
                                                                                 latent_size, batch_size,
                                                                                 device_name, n_chunks, max_seq_len)

                log_likelihood_res["log_like_posterior_generation"].append(post_likelihood)
                log_likelihood_res["log_like_prior_generation"].append(prior_likelihood)

                if batch_i + 1 == max_batches:
                    break

            res_dict = dict()
            for k, v in log_likelihood_res.items():
                res_dict[k] = torch.cat(v)

            # Dump the results for this device
            pickle.dump(res_dict, open(result_file, "wb"))


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True, type=str,
                        help="Path to the model needed to calculate importance weighted PPL (iw ppl).")
    parser.add_argument("--result_dir_path", required=False, type=str,
                        default="/home/cbarkhof/code-thesis/NewsVAE/evaluation/result-files/log_x_gen_vs_log_x_obs",
                        help="Path to directory to store the results.")
    parser.add_argument("--dataset_name", required=False, type=str,
                        default="ptb_text_only",
                        help="The name of the dataset (default: ptb_text_only).")
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
    parser.add_argument("--max_seq_len", required=False, type=int, default=32,
                        help="Maximum sequence length for auto-regressive decoding (default: 32).")
    parser.add_argument("--n_samples", required=False, type=int, default=100,
                        help="How many z samples to take for every q(z|x) eval (default: 500).")
    parser.add_argument("--n_chunks", required=False, type=int, default=2,
                        help="In how many chunks to divide the latent samples (default: 2).")

    config = parser.parse_args()

    return config


def main(config):
    # INIT DDP
    # print(f"*** Using DDP, spawing 4 processes")
    set_ddp_environment_vars(port_nr=1236)
    seed_everything(0)
    run_name = config.model_path.split("/")[-2]
    mp.spawn(evaluation_function, nprocs=4, args=(run_name, config.model_path, config.max_batches,
                                                  config.max_seq_len, config.with_grad,
                                                  config.result_dir_path, config.batch_size,
                                                  config.dataset_name, config.n_samples,
                                                  config.n_chunks, config.world_size, config.num_workers))
    combine_result_files(config.result_dir_path, run_name, config.max_batches, config.batch_size, config.n_samples)


if __name__ == "__main__":
    args = get_config()
    main(args)
