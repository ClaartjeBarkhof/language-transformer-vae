import torch
from utils_train import transfer_batch_to_device, load_from_checkpoint, cat_pad_uneven
import os
import copy
import numpy as np
from scipy import stats
import sys
import pickle

# ----------------------------------------------------------------------------------------------------
# PRIOR POSTERIOR PERFORMANCE DROP
# ----------------------------------------------------------------------------------------------------

def acc_drop_over_relative_seq_len(data_loader, model=None, path=None, device="cuda:0",
                                   max_batches=-1, N_bins=30):
    N = max_batches if max_batches > 0 else len(data_loader)
    assert not (model is None and path is None), "Either supply model or a path. Aborting."

    if path is not None and model is None:
        model = load_from_checkpoint(path, world_master=True, ddp=False, device_name=device, evaluation=True)

    prior_accs = []
    post_accs = []
    masks = []

    for batch_i, batch in enumerate(data_loader):
        print("Batch {:3d}/{:3d}".format(batch_i + 1, N), end="\r")

        # save mask
        labels = batch["input_ids"][:, 1:].contiguous()  # skip <s> token
        label_mask = (labels != 1).float()  # pad token is int 1
        masks.append(label_mask)

        # transfer batch to device
        batch = transfer_batch_to_device(batch, device)

        # save acc stats of experiments for batch
        for decode_prior_samples in [True, False]:
            with torch.no_grad():
                preds = model(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"],

                              auto_regressive=False,
                              max_seq_len=64,

                              return_exact_match=True,
                              return_cross_entropy=False,
                              return_reconstruction_loss=False,

                              return_posterior_stats=False,

                              reduce_seq_dim_ce="mean",
                              reduce_seq_dim_exact_match="none",
                              reduce_batch_dim_exact_match="none",
                              reduce_batch_dim_ce="none",

                              nucleus_sampling=False,
                              top_k=0,
                              top_p=1.0,

                              decode_sample_from_prior=decode_prior_samples,
                              n_prior_samples=batch["input_ids"].shape[0],

                              device_name=device)

            if decode_prior_samples is True:
                prior_accs.append(preds["exact_match"].cpu())
            else:
                post_accs.append(preds["exact_match"].cpu())

        if (batch_i + 1) == max_batches:
            break

    prior_accs = cat_pad_uneven(prior_accs, pad_value=0)
    post_accs = cat_pad_uneven(post_accs, pad_value=0)
    masks = cat_pad_uneven(masks, pad_value=0)
    seq_lens = masks.sum(dim=1)

    n_samples, max_len = prior_accs.shape
    positions = torch.arange(1, max_len + 1).unsqueeze(0).repeat(n_samples, 1)
    relative_positions = positions / seq_lens.unsqueeze(1)

    prior_accs_masked = torch.masked_select(prior_accs, masks == 1.0)
    post_accs_masked = torch.masked_select(post_accs, masks == 1.0)
    acc_drops = post_accs_masked - prior_accs_masked
    relative_positions_masked = torch.masked_select(relative_positions, masks == 1.0)

    bin_means, bin_edges, bin_ids = stats.binned_statistic(relative_positions_masked.tolist(), acc_drops.tolist(),
                                                           statistic='mean', bins=N_bins)

    return bin_means, bin_edges, acc_drops.mean()


# ----------------------------------------------------------------------------------------------------
# IMPORTANCE WEIGHTED LOG LIKELIHOOD log p (x)
# ----------------------------------------------------------------------------------------------------

def make_batch_from_model_samples(predictions, eos_token_id=2, pad_token_id=1, bos_token_id=0):
    # Add a <s> token to the predictions
    bos = torch.zeros_like(predictions)[:, 0].unsqueeze(1)
    predictions = torch.cat([bos, predictions], dim=1)

    # Make a tensor with position indices per row
    ind = torch.arange(predictions.shape[1]).repeat(predictions.shape[0], 1)

    # Check where the eos_token is in the predictions, if not there set to max_len
    lens = torch.tensor(
        [a.index(eos_token_id) if eos_token_id in a else len(a) for a in predictions.tolist()]).unsqueeze(1)

    # Mask everything after the eos_token_id (set to 0.0)
    mask = (ind > lens)

    # Pad the predictions (setting all tokens after </s> to <pad>)
    predictions[mask] = pad_token_id

    return predictions, mask, lens.flatten()


def iw_log_p_x(vae_model, batch, n_samples=600, n_chunks=3, verbose=False):
    batch_size = batch["input_ids"].shape[0]

    # Encode these input ids and sample <n_samples> for each x
    enc_out = vae_model.encoder.encode(batch["input_ids"], batch["attention_mask"],
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
        # [n_samples, latent_dim]
        post_samples_i = post_samples[sample_i, :, :]

        # list of [samples_per_chunk, latent_dim]
        post_samples_i_chunked = list(torch.chunk(post_samples_i, n_chunks, dim=0))

        for chunk_i, post_z in enumerate(post_samples_i_chunked):
            if verbose is True:
                print(f"sample i: {sample_i:3d} chunk i: {chunk_i:3d}", end="\r")
            chunk_size = post_z.shape[0]
            inputs_i = batch["input_ids"][sample_i, :].unsqueeze(0).repeat(chunk_size, 1)
            att_m_i = batch["attention_mask"][sample_i, :].unsqueeze(0).repeat(chunk_size, 1)

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
    iw_frac = post_log_p_x_z + post_log_p_z - post_log_q_z_x

    # Reduce the sample dimension with logsumexp, leaves shape [batch_size]
    likelihood = torch.logsumexp(iw_frac, dim=-1) - np.log(n_samples)

    return likelihood


def iw_log_p_x_generated(model=None, path=None, n_batches=10, batch_size=64, n_samples=600,
                         n_chunks=3, verbose=False, ddp=False, device_name="cuda:0", max_seq_len_gen=64):

    if model is None and path is None:
        print("Either provide a model, or a checkpoint path. Not neither. Aborting.");
        quit()

    if path is not None:
        print("Loading a model because provided a path {}.".format(path))
        model = load_from_checkpoint(path, world_master=True, ddp=ddp, device_name=device_name, evaluation=True,
                                     return_loss_term_manager=False)

    log_p_xs, log_p_x_ws = [], []

    for batch_i in range(n_batches):
        if verbose:
            print(f"Batch {batch_i}/{n_batches}")

        with torch.no_grad():
            # Sample from the model by decoding from prior auto-regressively with sampling
            out = model(return_reconstruction_loss=False,
                        return_posterior_stats=False,
                        auto_regressive=True,
                        max_seq_len=max_seq_len_gen,
                        return_predictions=True,
                        nucleus_sampling=True,
                        top_k=0,  # no filtering
                        top_p=1.0,  # no filtering
                        decode_sample_from_prior=True,
                        n_prior_samples=batch_size,
                        device_name=device_name)

            padded_predictions, mask, lens = make_batch_from_model_samples(out["predictions"])

            batch = dict(input_ids=padded_predictions.to(device_name), attention_mask=mask.to(device_name))

            log_p_x = iw_log_p_x(model, batch, n_samples=n_samples, n_chunks=n_chunks, verbose=True).cpu()
            log_p_x_w = log_p_x / lens

            log_p_xs.append(log_p_x)
            log_p_x_ws.append(log_p_x_w)

    log_p_xs = torch.cat(log_p_xs)
    log_p_x_ws = torch.cat(log_p_x_ws)

    return log_p_xs, log_p_x_ws, lens

def iw_log_p_x_dataset(data_loader, model=None, path=None, n_samples=600, n_chunks=3,
                       verbose=False, ddp=False, device_name="cuda:0", max_batches=-1):
    if model is None and path is None:
        print("Either provide a model, or a checkpoint path. Not neither. Aborting.");
        quit()

    if path is not None:
        print("Loading a model because provided a path {}.".format(path))
        model = load_from_checkpoint(path, world_master=True, ddp=ddp, device_name=device_name, evaluation=True,
                                     return_loss_term_manager=False)

    N = len(data_loader) if max_batches < 0 else max_batches
    print("N", N)

    log_p_xs = []
    sent_lens = []  # handy for perplexity
    for batch_i, batch in enumerate(data_loader):
        if verbose is True:
            print("*" * 40)
            print(f"{batch_i + 1:3d}/{N}")
            print("*" * 40)
        batch = transfer_batch_to_device(batch, device_name=device_name)

        with torch.no_grad():
            log_p_x = iw_log_p_x(model, batch, verbose=verbose,
                                 n_chunks=n_chunks, n_samples=n_samples)
            sent_lens.append(batch["attention_mask"].sum(dim=1))
            log_p_xs.append(log_p_x)

        if batch_i + 1 == N:
            break

    log_likelihood = torch.cat(log_p_xs, dim=0).cpu()
    sent_lens = torch.cat(sent_lens, dim=0).cpu()
    log_likelihood_p_w = log_likelihood / sent_lens

    return log_likelihood, log_likelihood_p_w, sent_lens


def iw_perplexity(data_loader, model=None, path=None, n_samples=600, n_chunks=3,
                  verbose=False, ddp=False, device_name="cuda:0", max_batches=-1):
    log_likelihood, sent_lens = iw_log_p_x_dataset(data_loader, model=model, path=path, n_samples=n_samples,
                                                   n_chunks=n_chunks, verbose=verbose, ddp=ddp,
                                                   device_name=device_name, max_batches=max_batches)
    iw_ppl = torch.exp((-log_likelihood / sent_lens).mean())

    return iw_ppl.item()


# ----------------------------------------------------------------------------------------------------
# SUMMARY STATS
# ----------------------------------------------------------------------------------------------------


def summary_statistics(path, run_name, data_loader, max_batches=-1,
                       device="cuda:0", result_folder="result-files"):
    os.makedirs(result_folder, exist_ok=True)

    # Make a loss term manager from checkpoint (includes the model)
    loss_term_manager = load_from_checkpoint(path, world_master=True, ddp=False, dataset_size=len(data_loader),
                                             device_name=device, evaluation=True, return_loss_term_manager=True)

    # Set to VAE standard objective
    loss_term_manager.objective = "vae"

    results = {}
    N = max_batches if max_batches > 0 else len(data_loader)

    for batch_i, batch in enumerate(data_loader):
        print("Batch {:3d}/{:3d}".format(batch_i + 1, N), end="\r")

        with torch.no_grad():
            batch = transfer_batch_to_device(batch, device)

            if "decoderOnly" in path:
                decoder_only = True
            else:
                decoder_only = False
            out = loss_term_manager(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    return_exact_match=True,
                                    return_reconstruction_loss=True,
                                    decoder_only=decoder_only,
                                    return_posterior_stats=True,
                                    device_name=device,
                                    return_cross_entropy=False,
                                    reduce_seq_dim_ce="mean",
                                    reduce_batch_dim_ce="mean",
                                    reduce_seq_dim_exact_match="mean",
                                    reduce_batch_dim_exact_match="mean")

            for k, v in out.items():
                if torch.is_tensor(v) and v.dim() == 0:
                    x = v.item()
                else:
                    x = v

                if k in results:
                    results[k].append(x)
                else:
                    results[k] = [x]

            if batch_i + 1 == max_batches:
                break

    results_cat = {}
    for k, v in results.items():
        if torch.is_tensor(v[0]):
            results_cat[k] = torch.cat(v, dim=0)
        else:
            results_cat[k] = v

    dump_pickle(results_cat, "{}/{}.pth".format(result_folder, run_name))


# ----------------------------------------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------------------------------------



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def load_pickle(f):
    return pickle.load(open(f, "rb"))


def dump_pickle(o, f):
    pickle.dump(o, open(f, "wb"))
