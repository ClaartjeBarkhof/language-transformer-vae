# Sys
import os
import sys; sys.path.append("../..")

# Torch, datasets, transformers, spacy
from datasets import load_from_disk
from transformers import RobertaTokenizerFast
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from utils_evaluation import valid_dataset_loader_tokenizer
import spacy

# My utils
from utils_train import load_from_checkpoint, transfer_batch_to_device
from utils_evaluation import tokenizer_batch_decode, reconstruct_autoregressive
from train import get_model_on_device

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from pylab import rcParams

# DS utils
import numpy as np
import pandas as pd

# Standard utils
from functools import partial
import copy


def load_model(path):
    vae_model = get_model_on_device(device_name="cuda:0", latent_size=768, gradient_checkpointing=False,
                                    add_latent_via_memory=True, add_latent_via_embeddings=True,
                                    do_tie_weights=True, world_master=True)

    # TODO: remove and use actual function argument
    path = "/home/cbarkhof/code-thesis/NewsVAE/Runs/29DEC-exp1-CYCLICAL-2-GRADSTEPS-135000-run-2020-12-28-21:40:07/checkpoint-best.pth"
    _, _, vae_model, _, global_step, epoch, best_valid_loss = load_from_checkpoint(vae_model, path, world_master=True,
                                                                                   ddp=False, use_amp=False)

    return vae_model


def teacher_forced_forward(vae_model, batch, tokenizer, hinge_kl_loss_lambda=0.0):
    output = {
        "text_prediction": None,
        "token_id_prediction": None,
        "cross_entropy_over_sequence": None,
        "cross_entropy_sum": None,
        "exact_match_acc_over_sequence": None,
        "exact_match_acc_mean": None,
        "kl_loss": None,
        "hinge_kl_loss": None,
        "mmd_loss": None
    }

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = copy.deepcopy(input_ids)

    # ENCODER
    mu, logvar, latent_z, kl_loss, hinge_kl_loss, mmd_loss = vae_model.encoder.encode(input_ids=input_ids,
                                                                                      attention_mask=attention_mask,
                                                                                      n_samples=1,
                                                                                      hinge_kl_loss_lambda=
                                                                                      hinge_kl_loss_lambda)

    output["kl_loss"] = kl_loss
    output["hinge_kl_loss"] = hinge_kl_loss
    output["mmd_loss"] = mmd_loss

    # DECODER
    latent_to_decoder_output = vae_model.latent_to_decoder(latent_z)
    decoder_outs = vae_model.decoder.model(input_ids=input_ids, attention_mask=attention_mask,
                                           latent_to_decoder_output=latent_to_decoder_output, labels=labels,
                                           return_attention_probs=True, return_dict=True,
                                           return_predictions=True, return_correct=True,
                                           return_cross_entropy=True, reduce_seq_dim="none",
                                           return_last_hidden_state=False)

    # PREDICTION IDS & TEXT
    # Teacher forced returns batch x 63 (sequence of 62 + end symbol)
    token_id_prediction = decoder_outs["predictions"][:, :-1]
    text_prediction = tokenizer_batch_decode(token_id_prediction, tokenizer)
    output["text_prediction"] = text_prediction
    output["token_id_prediction"] = token_id_prediction

    # CROSS ENTROPY
    # It returns only reduced over the sequence (63)
    output["cross_entropy_over_sequence"] = decoder_outs["cross_entropy"]
    output["cross_entropy_sum"] = decoder_outs["cross_entropy"].sum()

    # TODO: PERPLEXITY

    # EXACT MATCH ACC.
    output["exact_acc_match_over_sequence"] = decoder_outs["correct"].mean(dim=0)
    output["exact_match_acc_mean"] = decoder_outs["correct"].mean()

    # TODO: ATTENTION TO LATENTS?

    return output

def auto_regressive_forward(vae_model, batch, tokenizer):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = copy.deepcopy(input_ids)

    # return mu, logvar, latent_z, kl_loss, hinge_kl_loss, mmd_loss
    _, _, latent_z, kl_loss, _, _ = vae_model.encoder.encode(input_ids, input_ids)

    latent_to_decoder = vae_model.latent_to_decoder(latent_z)

    autoreg_decode = vae_model.decoder.autoregressive_decode(latent_to_decoder, tokenizer,
                                                             max_seq_len=max_seq_len,
                                                             nucleus_sampling=nucleus_sampling,
                                                             temperature=temperature, top_k=top_k, top_p=top_p,
                                                             device_name=device_name,
                                                             return_attention_to_latent=return_attention_to_latent)

    if return_attention_to_latent:
        generated, atts = autoreg_decode
    else:
        generated = autoreg_decode

    generated_text = tokenizer_batch_decode(generated, tokenizer)

    if return_attention_to_latent:
        return generated_text, generated, atts
    else:
        return generated_text, generated


def main(names_paths):
    valid_dataset, tokenizer, valid_loader = valid_dataset_loader_tokenizer(batch_size=32, num_workers=4)

    for name, path in names_paths:
        print("Evaluating:", name)

        hinge_kl_loss_lambda = 0.0 if "FREEBITS" not in name else float(name.split("-")[3])
        print("hinge_kl_loss_lambda", hinge_kl_loss_lambda)

        vae_model = load_model(path)
        vae_model.eval()

        teacher_output_all = []

        input_ids_all = []
        input_text_all = []

        autoreg_output_all = []

        with torch.no_grad():
            for batch_i, batch in enumerate(valid_loader):
                print(f"{batch_i + 1:5d} / {len(valid_loader):5d}", end='\r')

                input_text = tokenizer_batch_decode(batch["input_ids"], tokenizer)
                batch = transfer_batch_to_device(batch)

                ####################
                # TEACHER-FORCED   #
                ####################

                # TODO: fix hinge_kl_loss_lambda
                teacher_output = teacher_forced_forward(vae_model, batch, tokenizer,
                                                        hinge_kl_loss_lambda=hinge_kl_loss_lambda)
                teacher_output_all.append(teacher_output)

                ####################
                # AUTO-REGRESSIVE  #
                ####################

                autoreg_output = auto_regressive_forward(vae_model, batch, tokenizer)
                autoreg_output_all.append(autoreg_output)

            # TODO: remove after debugging
            if batch_i == 2:
                break

        print("Done evaluating:", name)



if __name__=="__main__":
    run_dir = '/home/cbarkhof/code-thesis/NewsVAE/Runs'

    runs_29DEC_names = ["-".join(run_name.split('-')[2:-5]) for run_name in os.listdir(run_dir) if "29DEC" in run_name]
    runs_29DEC_paths = [run_dir + '/' + run_name + "/checkpoint-best.pth" for run_name in os.listdir(run_dir) if "29DEC" in run_name]

    run_names_paths_to_evaluate = list(zip(runs_29DEC_names, runs_29DEC_paths))

    main(run_names_paths_to_evaluate)