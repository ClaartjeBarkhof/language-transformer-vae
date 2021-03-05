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
from train import get_model_on_device
from utils_train import load_from_checkpoint, set_ddp_environment_vars


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


def get_model(device_name, latent_size, path):
    vae_model = get_model_on_device(device_name=device_name,
                                    latent_size=latent_size,
                                    gradient_checkpointing=False,
                                    add_latent_via_memory=True,
                                    add_latent_via_embeddings=False,
                                    do_tie_weights=True,
                                    do_tie_embedding_spaces=True,
                                    world_master=True,
                                    add_decoder_output_embedding_bias=False)

    _, _, vae_model, _, _, _, _ = load_from_checkpoint(vae_model, path, device_name=device_name)
    vae_model.eval()
    return vae_model


def iw_ppl(device_rank, path, batch_size=64, importance_weight_samples=1000,
           latent_chunk_size=250, max_batches=-1, mode="teacherforced"):

    # Prepare some variables
    device_name = f"cuda:{device_rank}"
    latent_size = 32 if "latent32" in path else 64

    result_dir = Path("/home/cbarkhof/code-thesis/NewsVAE/evaluation/result-files/ppl-results-lisa")
    os.makedirs(result_dir, exist_ok=True)
    r = path.split("/")[-2]
    result_file = result_dir / f"{device_name}_{mode}_{r}_max_batches_{max_batches}.pickle"

    if os.path.isfile(result_file):
        print('_' * 80)
        print('_' * 80)
        print("Have done this one already!")
        print('_' * 80)
        print('_' * 80)

    else:

        print("*" * 30)
        print(mode.upper())
        print("*" * 30)

        print("-" * 30)
        print("path:", path)
        print("batch size:", batch_size)
        print("importance_weight_samples:", importance_weight_samples)
        print("latent_chunk_size:", latent_chunk_size)
        print("max_batches:", max_batches)
        print("latent size:", latent_size)
        print("device name:", device_name)
        print("-" * 30)

        # Get model
        vae_model = get_model(device_name, latent_size, path)

        # Get distributed validation data loader of PTB data set
        loader = get_dist_validation_loader(batch_size=batch_size, num_workers=2, max_seq_len=64,
                                            world_size=4, dataset_name="ptb_text_only", tokenizer_name="roberta",
                                            device_name=device_name, gpu_rank=device_rank)

        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=4, rank=device_rank)

        # Seed everything
        seed_everything(0)

        print(f"Len data loader on device {device_name}: {len(loader)}")

        n_samples = importance_weight_samples
        chunk_size = latent_chunk_size

        autoregressive = True if mode == "autoregressive" else False

        with torch.no_grad():
            N = len(loader)

            ppls = []
            ds = []
            log_p_xs = []
            log_p_x_p_ws = []
            ce_p_ws = []

            log_p_x_zs, log_q_z_xs, log_p_zs = [], [], []

            # For all batches in the validation set
            for batch_i, batch in enumerate(loader):
                print(f"{batch_i:3d}/{N}")

                # Send to right device
                batch = transfer_batch_to_device(batch, device_name)
                batch_size = batch["input_ids"].shape[0]

                # Get label mask (where input ids are not 1, which is the padding ID)
                labels = copy.deepcopy(batch["input_ids"])[:, 1:].contiguous()
                label_mask = (labels != 1).float()
                avg_len = label_mask.sum(dim=-1).mean()  # sum where is 1 and average over batch

                print("Average length:", avg_len)

                # Encode these input ids and sample <n_samples> for each x
                enc_out = vae_model.encoder.encode(batch["input_ids"], batch["attention_mask"], n_samples=n_samples,
                                                   hinge_kl_loss_lambda=0.5,
                                                   return_log_q_z_x=True,
                                                   return_log_p_z=True,
                                                   return_embeddings=False)

                # Unpack the tensors we need
                latent_z, log_p_z, log_q_z_x = enc_out["latent_z"], enc_out["log_p_z"], enc_out["log_q_z_x"]

                # Now we need to loop again because our batch size was multiplied by n_samples
                log_p_x_z = []
                ce_per_word = []
                distortion = []

                # For all samples x in batch
                for sample_i in range(batch_size):
                    print(f"sample i: {sample_i:3d}")

                    # Gather all n_samples z belonging to that x_i
                    latent_z_sample_i = latent_z[sample_i, :, :]

                    # Chunk those samples into batches and copy inputs and attention masks to match x (repeat)
                    input_ids = batch['input_ids'][sample_i, :].repeat(chunk_size, 1)
                    attention_mask = batch['attention_mask'][sample_i, :].repeat(chunk_size, 1)
                    n_chunks = int(n_samples / chunk_size)

                    # Get the mask for this sequence and its length
                    label_mask_i = label_mask[sample_i, :].repeat(chunk_size, 1)
                    len_i = label_mask[sample_i, :].sum()

                    for chunk_i, z_b in enumerate(torch.chunk(latent_z_sample_i, n_chunks, dim=0)):
                        # Teacher forced decoding
                        if autoregressive is False:
                            dec_out = vae_model.decoder.forward(z_b, input_ids, attention_mask,
                                                                labels=copy.deepcopy(input_ids),
                                                                return_cross_entropy=True,
                                                                reduce_seq_dim_ce="none",
                                                                reduce_batch_dim_ce="none")

                        # Auto-regressive decoding
                        else:
                            dec_out = vae_model.decoder.autoregressive_decode(z_b,
                                                                              max_seq_len=input_ids.shape[1],
                                                                              device_name=device_name,
                                                                              labels=copy.deepcopy(input_ids),
                                                                              return_cross_entropy=True,
                                                                              reduce_seq_dim_ce="none",
                                                                              reduce_batch_dim_ce="none")

                        # Collect cross entropy per word: multiply with mask, get average over seq and over batch
                        ce_per_word.append(((dec_out["cross_entropy"] * label_mask_i).sum(dim=-1) / len_i).mean())

                        # Collect distortion
                        distortion.append((dec_out["cross_entropy"] * label_mask_i).sum(dim=-1).mean())

                        # CE = - log p_x_z (not averaged over batch yet)
                        ce = (dec_out['cross_entropy'] * label_mask_i).sum(dim=-1)

                        log_p_x_z.append(- ce)  # we need the log likelihood, not the negative log likelihood

                # Get mean CE per word and mean distortion
                ce_per_word = torch.stack(ce_per_word).mean().item()
                distortion = torch.stack(distortion).mean().item()

                # Calculate importance weighted perplexity
                # log p(x) = log 1/N sum_i^N ( p(x|z_i) * p(z_i) ) / q(z_i|x)
                # log p(x) = log sum_i^N exp( log( p(x|z_i) * p(z_i) ) / q(z_i|x) )) + log 1/N
                # log p(x) = log sum_i^N exp( log p(x|z_i) + log p(z_i) - log q(z_i|x)) + log 1/N
                log_p_x_z = torch.cat(log_p_x_z, dim=0).reshape(-1, n_samples)
                log_frac = log_p_x_z + log_p_z - log_q_z_x
                log_p_x = torch.logsumexp(log_frac, dim=-1) + np.log(1 / n_samples)
                # log likelihood per word
                log_p_x_p_w = log_p_x.mean() / avg_len
                # take the exponent of the negative log likelihood per word
                ppl = torch.exp(- log_p_x_p_w)

                ppls.append(ppl.item())
                ds.append(distortion)
                log_p_xs.append(log_p_x.mean().item())
                log_p_x_p_ws.append(log_p_x_p_w.item())
                ce_p_ws.append(ce_per_word)

                log_p_x_zs.append(log_p_x_z.cpu().numpy())
                log_q_z_xs.append(log_q_z_x.cpu().numpy())
                log_p_zs.append(log_p_z.cpu().numpy())

                # print("log_p_x_z.shape", log_p_x_z.shape)
                # print("log_q_z_x.shape", log_q_z_x.shape)
                # print("log_p_z.shape", log_p_z.shape)

                print(f"ce per word: {ce_per_word:.2f} | D: {distortion:.2f}")
                print(f"log p x p w: {log_p_x_p_w:.2f} | ppl: {ppl:6f}")

                if batch_i == max_batches - 1:
                    break

            log_p_x_zs = np.concatenate(log_p_x_zs, axis=0)
            log_q_z_xs = np.concatenate(log_q_z_xs, axis=0)
            log_p_zs = np.concatenate(log_p_zs, axis=0)

            results = dict(PPL=ppls, distortion=ds, log_p_x=log_p_xs, log_p_x_p_w=log_p_x_p_ws, ce_p_w=ce_p_ws,
                           log_p_x_zs=log_p_x_zs, log_q_z_xs=log_q_z_xs, log_p_zs=log_p_zs)

            # Dump the results for this device
            pickle.dump(results, open(result_file, "wb"))


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", required=True, type=str,
                        help="Path to the model needed to calculate importance weighted PPL (iw ppl).")
    parser.add_argument("--batch_size", required=False, type=int, default=64,
                        help="Batch size (default: 64).")
    parser.add_argument("--importance_weight_samples", required=False, type=int, default=1000,
                        help="How many samples to use in importance sampling (default: 1000).")
    parser.add_argument("--latent_chunk_size", required=False, type=int, default=200,
                        help="How big the chunks of the samples should be. Now should fit exactly N times "
                             "into n_samples (default: 250, means 4 chunks per importance sampling.")
    parser.add_argument("--max_batches", required=False, type=int, default=-1,
                        help="Maximum validation batches to process (default: -1, means all).")
    parser.add_argument("--mode", required=True, type=str,
                        help="Which mode to run in: 'autoregressive' or 'teacherforced'. ")

    config = parser.parse_args()

    return config


def main(config):
    # INIT DDP
    print(f"*** Using DDP, spawing 4 processes")
    set_ddp_environment_vars(port_nr=1235)
    seed_everything(0)
    mp.spawn(iw_ppl, nprocs=4, args=(config.path, config.batch_size,
                                     config.importance_weight_samples, config.latent_chunk_size,
                                     config.max_batches, config.mode))


if __name__ == "__main__":
    args = get_config()
    main(args)
