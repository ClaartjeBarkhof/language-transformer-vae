
from utils_train import transfer_batch_to_device, load_from_checkpoint
from dataset_wrappper import NewsData
import os
import torch
import pathlib
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

batch_size = 32
num_workers = 2
device_name = "cuda:0"

dataset = NewsData(dataset_name="ptb_text_only", tokenizer_name="roberta",
                   batch_size=batch_size, num_workers=num_workers,
                   pin_memory=True, max_seq_len=64,
                   device=device_name)

valid_loader = dataset.val_dataloader(batch_size=batch_size)


d = pathlib.Path("/home/cbarkhof/code-thesis/NewsVAE/Runs")


runs = {
    "2021-03-24-24MAR_BetaVAE_FB_1.0-run-12:32:07": {"FB": True, "TargetRatePD": 1.0, "NZ": 32},
    "2021-03-24-24MAR_AE-run-15:54:46": {"AE": True, "NZ": 32},
    "2021-02-03-PTB-latent64-FB-1.0-run-13:06:00": {"NZ": 64, "FB": True, "TargetRatePD": 1.0},
    "2021-03-24-24MAR_BetaVAE_MDR_1.0-run-12:32:38": {"NZ": 32, "MDR": True, "TargetRatePD": 1.0},
    "2021-03-24-24MAR_BetaVAE_FB_0.5-run-12:32:07": {"NZ": 32, "FB": True, "TargetRatePD": 0.5},
    "2021-03-24-24MAR_BetaVAE_MDR_0.5-run-12:32:38": {"NZ": 32, "MDR": True, "TargetRatePD": 0.5},
    "2021-02-03-PTB-latent64-FB-0.50-run-12:29:58": {"NZ": 64, "FB": True, "TargetRatePD": 0.5},
    "2021-02-03-PTB-latent64-autoencoder-run-18:25:57": {"AE": True, "NZ": 64},
    "2021-03-24-24MAR_HoffmanTest-run-14:34:49": {"VAE": True, "NZ": 32},
    "2021-02-03-PTB-latent64-FB-0.00-run-17:14:10": {"VAE": True, "NZ": 64},
    "2021-03-24-24MAR_Hoffman_Test_ConstraintOptim-run-15:08:49": {"HoffmanCO": True, "NZ": 32, "TargetMI": 16,
                                                                   "TargetMargKL": 3.2},
    "2021-03-24-24MAR_Hoffman_Test_ConstraintOptim32-run-20:53:41": {"HoffmanCO": True, "NZ": 32, "TargetMI": 32,
                                                                     "TargetMargKL": 3.2}
}

for r, feat in runs.items():
    p = d / r / "checkpoint-best.pth"
    runs[r]["path"] = str(p)

df = pd.DataFrame(runs).fillna(False).transpose().reset_index().rename(columns={"index": "run_name"})

# Generation via ancestral sampling

def pad_autoregressive_preds(predictions, eos_token_id=2, pad_token_id=1, bos_token_id=0):
    # Make a tensor with position indices per row
    ind = torch.arange(predictions.shape[1]).repeat(predictions.shape[0], 1)

    # Check where the eos_token is in the predictions, if not there set to max_len
    lens = torch.tensor(
        [a.index(eos_token_id) if eos_token_id in a else len(a) for a in predictions.tolist()]).unsqueeze(1)

    # Mask everything after the eos_token_id (set to 0.0)
    mask = (ind >= lens)

    predictions[mask] = pad_token_id

    bos = torch.zeros_like(predictions)[:, 0].unsqueeze(1)
    predictions = torch.cat([bos, predictions], dim=1)

    return predictions


def get_sampled_x(n_samples, model):
    n_samples_per_decode = 400
    n = int(np.ceil(n_samples / n_samples_per_decode))

    predictions = []
    with torch.no_grad():
        for i in range(n):
            print(f"{i:2d}", end='\r')
            out = model(input_ids=None,
                        attention_mask=None,

                        auto_regressive=True,
                        max_seq_len=64,

                        return_reconstruction_loss=False,

                        return_embedding_distance=False,

                        return_predictions=True,

                        return_posterior_stats=True,

                        nucleus_sampling=True,
                        top_k=0,  # makes it normal sampling
                        top_p=1.0,  # makes it normal sampling

                        decode_sample_from_prior=True,
                        n_prior_samples=n_samples_per_decode,

                        device_name=device_name)

            pred = out["predictions"]
            eos_maks = pad_autoregressive_preds(pred)
            predictions.append(pred)
        predictions = torch.cat(predictions, dim=0)
    return predictions



# save_dir = "x_gen"
# os.makedirs(save_dir, exist_ok=True)
# for i in range(len(df)):
#     name = df.iloc[i]["run_name"]
#     path = df.iloc[i]["path"]
#     nz = df.iloc[i]["NZ"]
#     f = save_dir + "/" + name + ".pth"
#     if os.path.isfile(f):
#         print("done this one already! skipping it.", f)
#         continue
#     if os.path.isfile(path):
#         model = load_from_checkpoint(path, objective="evaluation", dataset_size=len(dataset.datasets["validation"]),
#                                      add_latent_via_embeddings=False, latent_size=nz,
#                                      add_decoder_output_embedding_bias=False, do_tie_embedding_spaces=True)
#         preds = get_sampled_x(3370, model)
#
#         torch.save(preds.cpu(), f)
#     else:
#         print(path)


import copy


def iw_log_p_x(batch_inputs, batch_masks, vae_model, n_samples=600, n_chunks=3):
    # Encode these input ids and sample <n_samples> for each x
    enc_out = vae_model.encoder.encode(batch_inputs, batch_masks,
                                       n_samples=n_samples,
                                       return_log_q_z_x=True,
                                       return_log_q_z=False,
                                       return_log_p_z=True,
                                       return_embeddings=False)

    # Unpack the tensors we need
    # [batch, n_samples, latent_dim], [batch, n_samples], [batch, n_samples]
    post_samples, post_log_p_z, post_log_q_z_x = enc_out["latent_z"], enc_out["log_p_z"], enc_out["log_q_z_x"]

    # Now we need to loop again because our batch size was multiplied by n_samples
    post_log_p_x_z = []

    # For all samples x in batch
    for sample_i in range(batch_size):
        print(f"sample i: {sample_i:3d}", end="\r")

        # [n_samples, latent_dim]
        post_samples_i = post_samples[sample_i, :, :]

        # list of [samples_per_chunk, latent_dim]
        post_samples_i_chunked = list(torch.chunk(post_samples_i, n_chunks, dim=0))

        for post_z in post_samples_i_chunked:
            chunk_size = post_z.shape[0]
            inputs_i = batch_inputs[sample_i, :].unsqueeze(0).repeat(chunk_size, 1)
            att_m_i = batch_masks[sample_i, :].unsqueeze(0).repeat(chunk_size, 1)

            dec_out = vae_model.decoder(post_z, inputs_i, att_m_i,
                                        labels=copy.deepcopy(inputs_i),
                                        return_reconstruction_loss=False,
                                        reduce_seq_dim_ce="sum",
                                        return_cross_entropy=True,
                                        reduce_batch_dim_ce="None")
            ll = - dec_out["cross_entropy"]
            post_log_p_x_z.append(ll)

    # From one big tensor of shape [n_samples * batch_size] to [batch_size, n_samples]
    post_log_p_x_z = torch.cat(post_log_p_x_z).reshape(batch_size, n_samples)
    post_frac = post_log_p_x_z.cpu() + post_log_p_z.cpu() - post_log_q_z_x.cpu()

    # Reduce the sample dimension with logsumexp, leaves shape [batch_size]
    post_likelihood = torch.logsumexp(post_frac, dim=-1) - np.log(n_samples)

    return post_likelihood


# # Get log p(x_obs)

# In[ ]:


# save_dir = "log_p_x_obs"
# os.makedirs(save_dir, exist_ok=True)
# max_batches = 8
# batch_size = 32

# for i in range(len(df)):
#     name = df.iloc[i]["run_name"]
#     path = df.iloc[i]["path"]
#     nz = df.iloc[i]["NZ"]

#     f = save_dir + "/" + name + ".pth"

#     if os.path.isfile(f):
#         print("done this one already! skipping it.", f)
#         continue

#     if os.path.isfile(path):
#         model = load_from_checkpoint(path, objective="evaluation", dataset_size=len(dataset.datasets["validation"]),
#                                      add_latent_via_embeddings=False, latent_size=nz,
#                                      add_decoder_output_embedding_bias=False, do_tie_embedding_spaces=True)

#         log_p_xs = []
#         for batch_i, batch in enumerate(valid_loader):
#             print("*"*40)
#             print(f"{batch_i+1:3d}/{len(valid_loader)}")
#             print("*"*40)
#             batch = transfer_batch_to_device(batch, device_name=device_name)

#             with torch.no_grad():
#                 log_p_x = iw_log_p_x(batch["input_ids"], batch["attention_mask"], model)
#                 log_p_xs.append(log_p_x.cpu())

#             if batch_i + 1 == max_batches:
#                 break
#         log_p_xs = torch.cat(log_p_xs, dim=0)
#         torch.save(log_p_xs, f)


# In[ ]:

from pathlib import Path
save_dir = "log_p_x_gen"
d_news = Path("/home/cbarkhof/code-thesis/NewsVAE")
print(str(d_news / save_dir))
os.makedirs(d_news / save_dir, exist_ok=True)
pad_token_id = 1

max_batches = 8
batch_size = 32

for i in range(len(df)):
    name = df.iloc[i]["run_name"]
    path = df.iloc[i]["path"]
    nz = df.iloc[i]["NZ"]

    f = str(d_news / save_dir)
    f += "/" + name + ".pth"

    if os.path.isfile(f):
        print("done this one already! skipping it.", f)
        continue

    if os.path.isfile(path):

        with torch.no_grad():
            model = load_from_checkpoint(path, objective="evaluation", dataset_size=len(dataset.datasets["validation"]),
                                         add_latent_via_embeddings=False, latent_size=nz,
                                         add_decoder_output_embedding_bias=False, do_tie_embedding_spaces=True)


            f1 = str(d_news / "x_gen")
            print("opening", f1)
            f1 += "/" + name + ".pth"
            x_gen = torch.load(f1)

            n_batches = int(np.ceil(len(x_gen) / batch_size))

            log_p_xs = []
            for batch_i in range(n_batches):
                begin = batch_i * batch_size
                end = (batch_i + 1) * batch_size

                input_batch = x_gen[begin:end, :].to(device_name)
                att_m_batch = (input_batch != pad_token_id).float().to(device_name)

                log_p_x = iw_log_p_x(input_batch, att_m_batch, model)
                log_p_xs.append(log_p_x)

            if batch_i + 1 == max_batches:
                break

            log_p_xs = torch.cat(log_p_xs, dim=0)
            torch.save(log_p_xs, f)

# In[ ]:




