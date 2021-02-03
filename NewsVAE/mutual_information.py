import os
from train import get_model_on_device
from utils_evaluation import valid_dataset_loader_tokenizer
import torch
from utils_train import load_from_checkpoint, transfer_batch_to_device
import numpy as np
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.categorical import Categorical
import pickle
from modules.encoder import gaussian_kernel
import math
from scipy.stats import gaussian_kde

"""
Functions in this file:
    - log_prob_pairs(dists, sample_batch)
    - mi_lower_bound(samples, dists)
    - mi_upper_bound(samples, dists)
    - calc_upper_and_lower_bound_representational_mi(batch_latent_samples, batch_mu_logvars, gauss_dists=None)
    - calc_upper_and_lower_bound_generative_mi(batch_probs, cat_dists=None)
    
    - main(model_path) <- executes calculation of MI bounds for whole validation set
"""


def _gaussian_log_likelihood(sample, mask, dim, mu=None, var=None):
    """Computes the log likelihood of a given sample under a gaussian with given parameters.
    If mu or var is not given they are assumed to be standard.
    """


    if mu is None and var is None:
        # log_q_z_x = (-0.5 * ((dev ** 2) / var) - 0.5 * (math.log(2 * math.pi) - logvar)).sum(dim=-1)
        return -0.5 * torch.sum((torch.log(sample.new_tensor(2 * np.pi)) + sample ** 2) * mask.unsqueeze(dim), dim=dim)
    elif mu is None:
        return -0.5 * torch.sum((torch.log(2 * np.pi * var) + sample ** 2 / var) * mask.unsqueeze(dim), dim=dim)
    elif var is None:
        return -0.5 * torch.sum((torch.log(sample.new_tensor(2 * np.pi)) + (sample - mu) ** 2)
                                * mask.unsqueeze(dim), dim=dim)
    else:
        return -0.5 * torch.sum((torch.log(2 * np.pi * var) + (sample - mu) ** 2 / var) * mask.unsqueeze(dim), dim=dim)


def q_z_estimate(z, mu, var, device="cuda:0"):
    """Computes an estimate of q(z), the marginal posterior."""
    # z = [S, z_dim], mu = [N, z_dim], var = [N, z_dim], log_q_z = [S, N]
    log_q_z_x = _gaussian_log_likelihood(sample=z.unsqueeze(1), mask=torch.tensor([[1.]], device=device), dim=2, mu=mu, var=var)
    # [S,]
    print("log_q_z_x.shape", log_q_z_x.shape)
    print('first 10 vals of log_q_z_x in q_z_estimate', log_q_z_x[:10])
    log_q_z = torch.logsumexp(log_q_z_x.unsqueeze(0), dim=1) - torch.log(torch.tensor(log_q_z_x.shape[1], device=device, dtype=torch.float))
    return log_q_z


# Altered from: https://github.com/tom-pelsmaeker/deep-generative-lm/blob/master/util/evaluation.py
def calc_mi_estimate(log_q_z_x, latents, mu, logvar, log_p_z, avg_KL, log_q_z_method="aggregate_posterior", mi_method="zhao"):
    """Computes the mutual information given a batch of samples and their LL under the sampling distribution.
    If log_q_z is not provided, this method uses kernel density estimation on the provided samples to compute this
    quantity. Otherwise, we will use the provided log_q_z to estimate the MI, either through Hoffman's or Zhao's method.
    Args:
        samples(torch.FloatTensor): [N, z_dim] dimensional tensor of samples from q(z|x).
        log_p_z(torch.FloatTensor): [N] dimensional tensor of log-probabilities of the samples under p(z).
        avg_H(torch.FloatTensor): [] dimensional tensor containing the average entropy of q(z|x).
        avg_KL(torch.FloatTensor): [] dimensional tensor containing the average KL[q(z|x)||p(z)].
        method: method for estimating the mutual information.
        kde_method: method for obtaining the kde likelihood estimates of the samples under q(z).
        log_q_z: [N] dimensional tensor of log-probabilities of the samples under q(z), or None.
    Returns:
        mi_estimate(float): estimated mutual information between X and Z given N samples from q(z|x).
        marg_KL(float): estimated KL[q(z)||p(z)] given N samples from q(z|x).
    """

    if log_q_z_method is "kernel":
        print("kernel method for calculating log_q_z of latents")
        # log_q_z = kde_gauss(samples, kde_method)
        samples_cpu = latents.cpu().numpy().transpose()
        print("samples_cpu.shape", samples_cpu.shape)
        kde = gaussian_kde(samples_cpu)
        # [num] with p(samples) under kernels
        print("first 10 vals of scikit gauss kde", kde.logpdf(samples_cpu)[:10])

        log_q_z = torch.log(gaussian_kernel(latents, latents).mean(dim=1) + 1e-80)
        print("first 10 vals of pytorch gauss kde", log_q_z[:10])
    elif log_q_z_method is "aggregate_posterior":
        log_q_z = torch.logsumexp(log_q_z_x, dim=0) - math.log(len(log_q_z_x))
    else:
        raise NotImplementedError

    print(log_q_z.shape, log_p_z.shape)
    print(log_q_z, log_p_z.mean())


    # KL(q||p) = E_q[log q / log p] = E_q[log_q] - E_q[log_p]
    marg_KL = (log_q_z - log_p_z.mean())

    if mi_method == 'zhao':  # https://arxiv.org/pdf/1806.06514.pdf (Lagrangian VAE)
        # recall that avg_h = log_q_z_x.mean()
        mi_estimate = (log_q_z_x.mean() - log_q_z.mean()).item()

    elif mi_method == 'hoffman':  # http://approximateinference.org/accepted/HoffmanJohnson2016.pdf (ELBO surgery)
        mi_estimate = (avg_KL - marg_KL).item()
    else:
        print('MI method {} is unknown. Please choose [zhao, hoffman], quitting...'.format(mi_method)); quit()

    return mi_estimate, marg_KL.item()


def log_prob_pairs(dists, sample_batch):
    """
    Calculates the log probability between all pairs (distribution, sample),
    resulting in a matrix with positive samples on the diagonal and
    negative samples off-diagonal.

    Args:
        dists: List[torch.distributions]
            List of Torch distribution objects that have a log_prob method

        batch_latent_samples: Tensor [batch, latent_size]
            A batch of samples resulting from said distributions

    Returns:
        log_probs_pairs: Tensor [batch, batch]
            A matrix with log probabilities of all samples under all distributions
            with the row dimension is x dimension, col dimension is z dimension
    """

    log_probs_pairs = [d.log_prob(sample_batch.squeeze()) for d in dists]
    log_probs_pairs = torch.stack(log_probs_pairs, dim=0)

    # returns [batch x batch] matrix, where column dimension is z dim
    # and the row dimension is the x dimension
    return log_probs_pairs


def mi_lower_bound(samples, dists):
    """
    Calculates the InfoTNCE lower bound of MI (Poole et al., 2019)

    I(X; Z) >= E[1/K sum^K_{i=1} (log p(z_i|x_i) - log 1/K sum^K_{j=1} p(y_i|x_j))]
             = E[1/K sum^K_{i=1} (log p(z_i|x_i) - log sum^K_{j=1} p(y_i|x_j) - log (K))]
             = E[1/K sum^K_{i=1} (log p(z_i|x_i) - log sum^K_{j=1} exp log p(y_i|x_j) - log (K))]
          --> For the second term the Log Sum Exp trick must be used to ensure numerical stability

    Args:
        batch_latent_samples: Tensor[batch, latent_size]
          a tensor of samples resulting from encoding and sampling

        batch_mu_logvars: Tensor[batch, 2*latent_size]
          a list of distributions that resulted from encoding

    Returns:
        mi_lower_tnce: float:
          InfoTNCE lower bound of MI
    """

    K = samples.shape[0]

    # Log probabilities between all latents and distributions
    # so for every distribution there is 1 positive sample where the latent was actually sampled
    # from that distribution and K-1 negative samples
    # -> diagonals are positive, off-diagonal are negative samples
    log_probs_pairs = log_prob_pairs(dists, samples)

    # Get the diagonal elements of log_probs_pairs for numerator
    log_prob_for_matching_x_z = torch.diag(log_probs_pairs)

    # Get the log mean probability for every z, use the logsumexp trick for this
    log_mean_prob_for_every_x_z = torch.logsumexp(log_probs_pairs, 0) - np.log(K)  # - log K comes from log (1/K)

    # Calculate the lower bound and mean over the batch (z dim)
    mi_lower_tnce = (log_prob_for_matching_x_z - log_mean_prob_for_every_x_z).mean().item()

    return mi_lower_tnce


def mi_upper_bound(samples, dists):
    """
    Calculates the MBU upper bound of MI (Poole et al., 2019)

    I(X; Z) >=  E[1/K sum^K_{i=1} (log p(z_i|x_i) - log 1/(K-1) sum^{K-1}_{j=/=i} p(y_i|x_j))]
             =  E[1/K sum^K_{i=1} (log p(z_i|x_i) - (log sum^{K-1}_{j=/=i} p(y_i|x_j) - log (K-1)))]
             =  E[1/K sum^K_{i=1} (log p(z_i|x_i) - log sum^{K-1}_{j=/=i} exp log p(y_i|x_j) + log (K-1))]
             --> For the second term the Log Sum Exp trick must be used to ensure numerical stability

    Args:
        batch_latent_samples: Tensor[batch, latent_size]
          a tensor of samples resulting from encoding and sampling

        batch_mu_logvars: Tensor[batch, 2*latent_size]
          a list of distributions that resulted from encoding

    Returns:
        mi_upper_mbu: float:
          MBU upper bound of MI
    """
    K = samples.shape[0]

    # Log probability of samples under all distributions of the batch
    # diagonal elements are matching distributions and samples
    # off diagonal elements are 'negative' samples
    log_probs_pairs = log_prob_pairs(dists, samples)

    # Get the diagonal elements of log_probs_pairs for numerator
    log_prob_for_matching_x_z = torch.diag(log_probs_pairs)

    # Get the log mean probability for non matching pairs x and z, use the logsumexp trick for this
    select_offdiagonal = log_probs_pairs - torch.diag(np.inf * torch.ones(K))  # exp(-inf) -> 0
    logsumexp_offdiagonal = torch.logsumexp(select_offdiagonal, 0)  # the exp for log prob to prob

    marginal = logsumexp_offdiagonal - np.log(K - 1)

    # Calculate the uppper bound
    mi_upper_mbu = (log_prob_for_matching_x_z - marginal).mean().item()

    return mi_upper_mbu


def calc_upper_and_lower_bound_representational_mi(batch_latent_samples, batch_mu_logvars, gauss_dists=None):
    """
    This function calculates the Info TNCE lower bound and MBU upper bound
    on mutual information between inputs x and sampled latents with
    tractable condiational (encoder) as a critic (Poole et al., 2019).

    Args:
        batch_latent_samples: Tensor [batch, latent_size]
        batch_mu_logvars: Tensor [batch, 2*latent_size]
        gauss_dists: List[torch.distributions.MultivariateNormal] [batch]

    Returns:
        lower: float
        upper: float

    """

    K = batch_latent_samples.shape[0]

    if not gauss_dists:
        # Chunk the parameters in two and transform logvar into std
        mu, logvar = torch.chunk(batch_mu_logvars, 2, dim=1)
        std = torch.exp(0.5 * logvar)

        # Make distribution objects for the batch of mu, std
        gauss_dists = [MultivariateNormal(mu[i, :], torch.diag(std[i, :])) for i in range(K)]

    lower = mi_lower_bound(batch_latent_samples, gauss_dists)
    upper = mi_upper_bound(batch_latent_samples, gauss_dists)

    return lower, upper


def calc_upper_and_lower_bound_generative_mi(batch_probs, cat_dists=None):
    """
    This function calculates the Info TNCE lower bound and MBU upper bound
    on mutual information latents x and sampled predictions with
    tractable conditional (decoder) as a critic (Poole et al., 2019).

    Args:
        batch_probs: Tensor [batch, seq_len, vocab_size]
        cat_dists: List[torch.distributions.Categorical] [batch]
    Returns:
        lower: float
        upper: float

    """

    K = batch_probs.shape[0]

    batch_probs = batch_probs.reshape(-1, batch_probs.shape[-1])
    sampled_predictions = torch.multinomial(batch_probs, num_samples=1).squeeze(1)

    if not cat_dists:
        # Make distribution objects for the batch probabilities
        cat_dists = [Categorical(probs=batch_probs[i, :]) for i in range(K)]

    lower = mi_lower_bound(sampled_predictions, cat_dists)
    upper = mi_upper_bound(sampled_predictions, cat_dists)

    return lower, upper


def calc_all_mi_bounds(vae_model, valid_loader, device_name="cuda:0", max_batches=10, batch_size=128,
                       auto_regressive=False):

    print(f"Evaluating mutual information bounds for over "
          f"{max_batches} batches of size {batch_size}.")


    lower_rep_mi_all, upper_rep_mi_all = [], []
    lower_gen_mi_all, upper_gen_mi_all = [], []

    with torch.no_grad():
        for batch_i, batch in enumerate(valid_loader):

            batch = transfer_batch_to_device(batch, device_name)

            vae_output = vae_model.forward(input_ids=batch["input_ids"],
                                           attention_mask=batch["attention_mask"],
                                           beta=1.0,

                                           auto_regressive=auto_regressive,

                                           return_probabilities=True,
                                           return_latents=True,
                                           return_mu_logvar=True,

                                           return_exact_match=False,
                                           return_cross_entropy=True,
                                           return_predictions=False,
                                           return_logits=False,
                                           return_hidden_states=False,
                                           return_last_hidden_state=False,
                                           return_attention_to_latent=False,
                                           return_attention_probs=False,
                                           return_text_predictions=False,
                                           tokenizer=None,

                                           device_name=device_name)

            # -------------------------------------------#
            # REPRESENTATIONAL MUTUAL INFORMATION BOUNDS #
            # -------------------------------------------#

            lower_rep_mi, upper_rep_mi = calc_upper_and_lower_bound_representational_mi(vae_output["latents"].cpu(),
                                                                                        vae_output["mu_logvar"].cpu())
            lower_rep_mi_all.append(lower_rep_mi)
            upper_rep_mi_all.append(upper_rep_mi)

            # -------------------------------------#
            # GENERATIVE MUTUAL INFORMATION BOUNDS #
            # -------------------------------------#

            # Merge sequence dimension in the batch dimension
            batch_size, seq_len, vocab_size = vae_output["probabilities"].shape
            probs = vae_output["probabilities"].reshape(-1, vocab_size).cpu()

            # Calculate lower and upper bounds
            lower_gen_mi, upper_gen_mi = calc_upper_and_lower_bound_generative_mi(probs, cat_dists=None)

            lower_gen_mi_all.append(lower_gen_mi)
            upper_gen_mi_all.append(upper_gen_mi)

            print(f"{batch_i:3d} Representational MI -> lower: {lower_rep_mi:.2f}, upper: {upper_rep_mi:.2f} "
                  f"| Generative MI -> lower: {lower_gen_mi:.2f}, upper: {upper_gen_mi:.2f}")

            if max_batches != -1:
                if batch_i == max_batches:
                    break

    mi_results = {
            "lower_rep_mi_all": lower_rep_mi_all,
            "upper_rep_mi_all": upper_rep_mi_all,
            "lower_gen_mi_all": lower_gen_mi_all,
            "upper_gen_mi_all": upper_gen_mi_all
        }

    return mi_results


if __name__ == "__main__":
    # TODO: change this
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    MAX_BATCHES = 30

    # Device
    DEVICE_NAME = "cuda:0"

    # Get the runs of 29DEC
    run_dir = '/home/cbarkhof/code-thesis/NewsVAE/Runs'
    runs_29DEC_names = ["-".join(run_name.split('-')[2:-5]) for run_name in os.listdir(run_dir) if "29DEC" in run_name]
    runs_29DEC_paths = [run_dir + '/' + run_name + "/checkpoint-best.pth" for run_name in os.listdir(run_dir) if
                        "29DEC" in run_name]
    run_names_paths_to_evaluate = list(zip(runs_29DEC_names, runs_29DEC_paths))

    # Get validation data
    _, _, VALID_LOADER = valid_dataset_loader_tokenizer(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Calculate MI bounds for these models
    mutual_information_results = {}
    for name, path in run_names_paths_to_evaluate:

        vae_model = get_model_on_device(device_name=DEVICE_NAME, latent_size=768, gradient_checkpointing=False,
                                        add_latent_via_memory=True, add_latent_via_embeddings=True,
                                        do_tie_weights=True, world_master=True)

        _, _, vae_model, _, _, _, _ = load_from_checkpoint(vae_model, path, world_master=True, ddp=False,
                                                           use_amp=False)

        mi_results = calc_all_mi_bounds(vae_model, VALID_LOADER, device_name=DEVICE_NAME, max_batches=MAX_BATCHES, batch_size=BATCH_SIZE)

        mutual_information_results[name] = mi_results

    prefix = "/home/cbarkhof/code-thesis/NewsVAE/evaluation/29DEC/"
    pickle_filename = "29DEC-mutual-information-results.p"
    pickle_path = prefix + pickle_filename

    pickle.dump(mutual_information_results, open(pickle_path, "wb"))
    # mutual_information_results = pickle.load(open(pickle_path, "rb")) # TODO: remove this
    # print(mutual_information_results)
