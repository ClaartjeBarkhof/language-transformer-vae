#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Utils analysis
from utils_analysis import *

# Standard
import torch
import numpy as np
import pandas as pd
import os
import sys; sys.path.append("/home/ec2-user/code-thesis/NewsVAE")

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# In[2]:


def JSD(p, q, log_p, log_q):
    """
    Jenson Shannon Divergence(P || Q)
    
    p, q: n-dimensional tensors both expected to contain log-probabilities
    Args:
        
    
    
    they should be batch x seq_len x vocab_size
    
    """
    
    # Mean distribution 
    m = 0.5 * (p + q)
    
    
    # JSD: symmtretical KL divergence
    # unreduced KL: p(x) * log ( p(x) / q(x) ) -> sum over probability dimension (last dim.)
    # kl div: input given is expected to contain log-probabilities and is not restricted to a 2D Tensor. 
    # The targets are interpreted as probabilities by default
    jsd = 0.5 * F.kl_div(log_p, m, reduction="none").sum(dim=-1) +           0.5 * F.kl_div(log_q, m, reduction="none").sum(dim=-1)
        
    return jsd


# In[3]:


from utils_train import load_from_checkpoint, transfer_batch_to_device
from dataset_wrappper import NewsData
import torch.nn.functional as F


def calc_JSD_over_seq(run_name, exp_name, max_batches, batch_size, dataset, n_exp, device="cuda:0"):
    # collect the mean JSD between prior decoded and posterior decoded samples over 10 samples 
    # for the whole validation set
    jsd_over_seq = []
    jsd_post_post = []
    jsd_prior_post = []
    label_masks = []

    path, best_epoch = get_best_checkpoint(run_name, exp_name=exp_name)
    
    if path is None:
        return None

    #--------------------------------------------------
    # Get data that this model was trained on
    if "optimus" in dataset.lower():
        dataset_name = "optimus_yelp"
    elif "yelp" in dataset.lower():
        dataset_name = "yelp"
    else:
        dataset_name = "ptb_text_only"

    # Load relevant data
    data = NewsData(dataset_name=dataset_name, tokenizer_name="roberta", batch_size=batch_size, 
                    num_workers=3, pin_memory=True, max_seq_len=64, device=device)
    val_loader = data.val_dataloader(shuffle=False, batch_size=batch_size)

    #--------------------------------------------------
    # Get model
    vae_model = load_from_checkpoint(path, world_master=True, ddp=False, device_name=device, 
                                     evaluation=True, return_loss_term_manager=False)



    with torch.no_grad():
        for batch_i, batch in enumerate(val_loader):

            batch = transfer_batch_to_device(batch, device)

            logits_prior_batch = []
            logits_posterior_batch = []

            for exp_i in range(n_exp):
                print(f"Batch: {batch_i+1:3d}/{max_batches} - Exp: {exp_i+1:3d}/{n_exp}", end="\r")

                for decode_sample_from_prior_mode in [False, True]:

                    vae_output = vae_model(input_ids=batch["input_ids"],
                                           attention_mask=batch["attention_mask"],
                                           auto_regressive=False,
                                           max_seq_len=64,
                                           return_reconstruction_loss=True,
                                           return_posterior_stats=False,
                                           nucleus_sampling=False,
                                           top_k=0,
                                           top_p=1.0,

                                           return_logits=True,

                                           # these two are the relevant ones
                                           decode_sample_from_prior=decode_sample_from_prior_mode,
                                           n_prior_samples=batch_size, # as many as the batch reconstruct, so batch size

                                           device_name=device)

                    if decode_sample_from_prior_mode is True:
                        logits_prior_batch.append(vae_output["logits"].cpu())
                    else:
                        logits_posterior_batch.append(vae_output["logits"].cpu())

            # make a reordered posterior list, to compare posterior 
            # to posterior (and account for internal variability)
            re_order = list(np.arange(1, n_exp)) + [0]          
            logits_posterior_reordered = [logits_posterior_batch[i] for i in re_order]            

            # After cat operation, they matrices are [n_exp * batchsize x seq_len x vocab]
            # The log soft_max should operate on the last dimension
            log_probs_prior_batch = F.log_softmax(torch.cat(logits_prior_batch, dim=0), dim=-1)
            log_probs_posterior_batch = F.log_softmax(torch.cat(logits_posterior_batch, dim=0), dim=-1)
            log_probs_posterior_reordered_batch = F.log_softmax(torch.cat(logits_posterior_reordered, dim=0), dim=-1)

            probs_prior_batch = F.softmax(torch.cat(logits_prior_batch, dim=0), dim=-1)
            probs_posterior_batch = F.softmax(torch.cat(logits_posterior_batch, dim=0), dim=-1)
            probs_posterior_reordered_batch = F.softmax(torch.cat(logits_posterior_reordered, dim=0), dim=-1)
            
            print("log_probs_prior_batch.shape, probs_prior_batch.shape")
            print(log_probs_prior_batch.shape, probs_prior_batch.shape)

            # JSD returns [n_exp * batchsize x seq_len]
            jsd_prior_posterior = JSD(probs_posterior_batch, probs_prior_batch, log_probs_posterior_batch, log_probs_prior_batch)
            jsd_posterior_posterior = JSD(probs_posterior_batch, probs_posterior_reordered_batch, log_probs_posterior_batch, log_probs_posterior_reordered_batch)
            jsd_dif = jsd_prior_posterior - jsd_posterior_posterior
            
            print("jsd_dif.shape", jsd_dif.shape)

            # Take into account the different sequence lengths, correct for that when averaging
            labels = batch["input_ids"].cpu()[:, 1:].contiguous()  # skip <s> token
            label_mask = (labels != 1).float().repeat(n_exp, 1) # pad token is int 1
            #label_mask_sum_batch_exp = label_mask.sum(dim=0) # sum over the batch, n_exp dim
            # mean_jsd_dif = jsd_dif.sum(dim=0) / label_mask_sum_batch_exp

            print("label_mask.shape", label_mask.shape)
            
            jsd_over_seq.append(jsd_dif)
            jsd_post_post.append(jsd_posterior_posterior)
            jsd_prior_post.append(jsd_prior_posterior)
            label_masks.append(label_mask)

            if batch_i == max_batches - 1:
                break

            # ------- END BATCH EXP ---------

        # -------- END ALL BATCHES ----------

    # Maximum sequence length, needed for padding
    max_len = max([t.shape[1] for t in jsd_over_seq])

    # pad all to have the same length 
    jsd_over_seq = [F.pad(x, (0, max_len-x.shape[1])) for x in jsd_over_seq]
    jsd_over_seq = torch.cat(jsd_over_seq, dim=0)

    jsd_post_post = [F.pad(x, (0, max_len-x.shape[1])) for x in jsd_post_post]
    jsd_post_post = torch.cat(jsd_post_post, dim=0)

    jsd_prior_post = [F.pad(x, (0, max_len-x.shape[1])) for x in jsd_prior_post]
    jsd_prior_post = torch.cat(jsd_prior_post, dim=0)
    
    

    label_masks = [F.pad(x, (0, max_len-x.shape[1])) for x in label_masks]
    label_masks = torch.cat(label_masks, dim=0)
    
    print(jsd_over_seq.shape, jsd_post_post.shape, jsd_prior_post.shape, label_masks.shape)

    results_jsd ={
        "jsd_over_seq": jsd_over_seq,
        "jsd_post_post": jsd_post_post,
        "jsd_prior_post": jsd_prior_post,
        "label_masks": label_masks
    }
    
    return results_jsd


# In[4]:

if __name__=="__main__":
    print("*"*80)
    print("*" * 80)
    print("*" * 80)
    print("TEST")

    N_EXP = 3
    MAX_BATCHES = 40
    BS = 20
    DEVICE = "cuda:0"
    exp_name = "Runs-target-rate"

    failed_runs = []
    # for exp_name, run_dir in RUN_DIRS.items():
    run_overview = read_overview_csv(exp_name=exp_name)

    for row_i, row in run_overview.iterrows():
        run_name, clean_name = row['run_name'], row['clean_name']

        if check_if_running(run_name, exp_name):
            continue

        print("*" * 50)
        print(clean_name)
        print(run_name)
        print("*" * 50)

        d = f"{RES_FILE_DIR}/{exp_name}/{run_name}"
        os.makedirs(d, exist_ok=True)
        RESULT_FILE = f"{d}/result_JSD_over_seq_N_EXP_{N_EXP}_MAX_BATCHES_{MAX_BATCHES}_BS_{BS}.pickle"
        print(RESULT_FILE)

        # If already ran, do not run again
        if os.path.exists(RESULT_FILE):
            print(f"Loading file {RESULT_FILE}, it existed.")
            results_jsd = pickle.load( open( RESULT_FILE, "rb" ) )

        else:
            # run_name, exp_name, max_batches, batch_size, dataset, n_exp, device="cuda:0"
            print(N_EXP, MAX_BATCHES, BS)

            results_jsd = calc_JSD_over_seq(run_name=run_name, exp_name=exp_name, max_batches=MAX_BATCHES,
                                            batch_size=BS, dataset=row["dataset"], n_exp=N_EXP, device=DEVICE)

        if results_jsd is not None:
            pickle.dump( results_jsd, open( RESULT_FILE, "wb" ))

        else:
            failed_runs.append(run_name)

    print("Failed runs:")
    for i, f in enumerate(failed_runs):
        print(f)




