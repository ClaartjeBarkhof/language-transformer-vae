import torch
import copy
import utils_evaluation
import numpy as np
import utils_train
from datasets import load_metric
from mutual_information import calc_all_mi_bounds, calc_mi_estimate
import math
from train import get_model_on_device
from utils_evaluation import valid_dataset_loader_tokenizer
from run_validation import load_model_for_eval





def validation_loop(vae_model, valid_loader, device_name="cuda:0"):

    vae_model.eval()

    result_dict = {
        "exact_match": [],
        "cross_entropy": [],
        "kl_loss": [],
        "mmd_loss": [],
        "recon_loss": [],
        "hinge_kl_loss": [],
        "total_loss": [],
        "latents": [],
        "log_p_z": [],
        "log_q_z_x": [],
        "mu": [],
        "logvar": []
    }

    for batch_i, batch in enumerate(valid_loader):
        print(f"Batch {batch_i + 1:4d} / {len(valid_loader):4d}")
        batch = utils_train.transfer_batch_to_device(batch, device_name)

        with torch.no_grad():
            outputs = vae_model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],

                                auto_regressive=False,
                                beta=1.0,

                                return_hidden_states=False,
                                return_attention_to_latent=False,
                                return_attention_probs=False,
                                return_logits=False,
                                return_probabilities=False,

                                return_latents=True,
                                return_mu_logvar=True,
                                return_log_p_z=True,
                                return_log_q_z_x=True,

                                return_predictions=False,
                                return_text_predictions=False,

                                return_exact_match=True,
                                return_cross_entropy=True,

                                reduce_seq_dim_exact_match="mean",
                                reduce_batch_dim_exact_match="mean",
                                reduce_seq_dim_ce="sum",
                                reduce_batch_dim_ce="mean",

                                nucleus_sampling=False)

            # Add (teacher forced) results
            for k, v in outputs.items():
                if k in result_dict:
                    if torch.is_tensor(v):
                        result_dict[k].append(v.detach())
                    else:
                        result_dict[k].append(v)

            if batch_i == 10:
                break

    stacked_result_dict = {}
    for k, v in result_dict.items():
        if torch.is_tensor(v[0]):
            if len(v[0].shape) == 0:
                stacked_result_dict[k] = torch.stack(v)
            else:
                stacked_result_dict[k] = torch.cat(v, dim=0)
        else:
            stacked_result_dict[k] = v

    result_dict = stacked_result_dict
    del stacked_result_dict

    ###############################
    #       MUTUAL INFORMATION    #
    ###############################

    # Aggragate posterior log_q_z
    # Taken from: https://github.com/bohanli/vae-pretraining-encoder/blob/master/modules/encoders/gaussian_encoder.py

    for mi_method in ["zhao", "hoffman"]:
        for log_q_z_method in ["aggregate_posterior", "kernel"]:
            print(mi_method, log_q_z_method)

            mi_est, marg_kl = calc_mi_estimate(log_q_z_x=result_dict["log_q_z_x"],
                                               latents=result_dict["latents"],
                                               mu=result_dict["mu"],
                                               logvar=result_dict["logvar"],
                                               log_p_z=result_dict["log_p_z"],
                                               avg_KL=np.mean(result_dict["kl_loss"]),
                                               log_q_z_method=log_q_z_method,
                                               mi_method=mi_method)
            print("mi_est", mi_est)
            print("marg kl", marg_kl)

            # result_dict["mi_est_" + method] = mi_est
            # result_dict["marg_kl_" + method] = marg_kl

    return result_dict

if __name__ == "__main__":
    device = "cuda:0"

    # vae_model = get_model_on_device(device_name=device)

    path = "/home/cbarkhof/code-thesis/NewsVAE/Runs/18NOV-BETA-VAE-run-2020-11-18-12:36:55/checkpoint-50000.pth"
    vae_model = load_model_for_eval(path, device_name="cpu")
    vae_model.to("cuda:0")

    dataset_path = "/home/cbarkhof/code-thesis/NewsVAE/NewsData/22DEC-cnn_dailymail-roberta-seqlen64/validation"
    _, _, valid_loader = valid_dataset_loader_tokenizer(batch_size=64, num_workers=4, dataset_path=dataset_path)
    validation_loop(vae_model, valid_loader, device_name=device)