import torch
import copy
import utils_evaluation
import numpy as np
import utils_train
from datasets import load_metric
from mutual_information import calc_all_mi_bounds


def validation_set_results(vae_model, valid_loader, tokenizer, device_name="cuda:0",
                           max_batches=-1, batch_size_mi_calc=128, n_batches_mi_calc=20):

    vae_model.eval()

    result_dict = {
        "text_predictions": [],
        "predictions": [],
        "attention_to_latent": [],
        "exact_match": [],
        "cross_entropy": [],
        "kl_loss": [],
        "mmd_loss": [],
        "recon_loss": [],
        "hinge_kl_loss": [],
        "total_loss": [],
        "latents": []
    }

    results = {
        "teacher_forced": copy.deepcopy(result_dict),
        "auto_regressive": copy.deepcopy(result_dict),
        "labels": {
            "labels_ids": [],
            "labels_text": []
        }
    }

    if max_batches == -1:
        max_batches = len(valid_loader)

    for batch_i, batch in enumerate(valid_loader):

        print(f"Batch {batch_i + 1:4d} / {len(valid_loader):4d}")

        batch = utils_train.transfer_batch_to_device(batch, device_name)
        with torch.no_grad():

            labels_ids = batch["input_ids"][:, 1:]
            labels_text = utils_evaluation.tokenizer_batch_decode(labels_ids, tokenizer)

            outputs_t_f = vae_model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],

                                    auto_regressive=False,
                                    beta=1.0,

                                    return_hidden_states=False,
                                    return_attention_to_latent=True,
                                    return_attention_probs=False,
                                    return_logits=False,
                                    return_probabilities=False,
                                    return_latents=True,
                                    return_mu_logvar=False,

                                    return_predictions=True,
                                    return_text_predictions=True,

                                    return_exact_match=True,
                                    return_cross_entropy=True,

                                    reduce_seq_dim_exact_match="none",
                                    reduce_batch_dim_exact_match="none",
                                    reduce_seq_dim_ce="none",
                                    reduce_batch_dim_ce="none",

                                    tokenizer=tokenizer,

                                    nucleus_sampling=False)

            outputs_a_r = vae_model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],

                                    auto_regressive=True,
                                    beta=1.0,

                                    return_hidden_states=False,
                                    return_attention_to_latent=True,
                                    return_attention_probs=False,
                                    return_logits=False,
                                    return_probabilities=False,
                                    return_latents=True,
                                    return_mu_logvar=False,

                                    return_predictions=True,
                                    return_text_predictions=True,

                                    return_exact_match=True,
                                    return_cross_entropy=True,

                                    reduce_seq_dim_exact_match="none",
                                    reduce_batch_dim_exact_match="none",
                                    reduce_seq_dim_ce="none",
                                    reduce_batch_dim_ce="none",

                                    tokenizer=tokenizer,

                                    nucleus_sampling=False)

            # Add labels
            results["labels"]["labels_ids"].append(labels_ids.detach().cpu().numpy())
            results["labels"]["labels_text"].extend(labels_text)

            # Add teacher forced results
            for k, v in outputs_t_f.items():
                if k in results["teacher_forced"]:
                    if torch.is_tensor(v):
                        results["teacher_forced"][k].append(v.detach().cpu().numpy())
                    else:
                        results["teacher_forced"][k].append(v)

            # Add auto-regressive results
            for k, v in outputs_a_r.items():
                if k in results["auto_regressive"]:
                    if torch.is_tensor(v):
                        results["auto_regressive"][k].append(v.detach().cpu().numpy())
                    else:
                        results["auto_regressive"][k].append(v)

            if batch_i + 1 == max_batches:
                break

    stacked_results = {}
    for group_name, group_results in results.items():

        stacked_results[group_name] = {}

        for k, v in group_results.items():
            if isinstance(v[0], np.ndarray) and k != "recon_loss" and k != "total_loss":
                stacked_results[group_name][k] = np.concatenate(v, axis=0)
            elif k == 'text_predictions':
                stacked_results[group_name][k] = [item for sublist in v for item in sublist]
            else:
                stacked_results[group_name][k] = v

    results = stacked_results
    del stacked_results

    ###############################
    #       MUTUAL INFORMATION    #
    ###############################

    for mode, name in zip([False, True], ["teacher_forced", "auto_regressive"]):
        print(f"Calculate MI {name}")
        mi_results = calc_all_mi_bounds(vae_model, valid_loader, max_batches=n_batches_mi_calc,
                                        batch_size=batch_size_mi_calc, auto_regressive=mode)
        results[name]["mi_results"] = mi_results

    ###############################
    #    N-GRAM MATCH STATS       #
    ###############################

    for name in ["teacher_forced", "auto_regressive"]:
        labels = results["labels"]["labels_ids"].tolist()
        preds = results[name]["predictions"].tolist()
        n_gram_results = {}
        for n in range(1, 10):
            matching_pos = []
            for i in range(len(preds)):
                matching_pos.extend(utils_evaluation.get_matching_ngram_stats(preds[i], labels[i], n))
            n_gram_results[n] = matching_pos
        results[name]["n_gram_results"] = n_gram_results

    ###############################
    #       BLEU                  #
    ###############################

    bleu_metric = load_metric("bleu")

    for name in ["teacher_forced", "auto_regressive"]:
        preds = results[name]["predictions"].tolist()
        # needs to be a list of lists
        refs = [[l] for l in results["labels"]["labels_ids"].tolist()]
        bleu = bleu_metric._compute(preds, refs)
        results[name]["bleu"] = bleu

    return results