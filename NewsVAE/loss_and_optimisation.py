from constraintoptim.constraint import Constraint, ConstraintOptimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler
import torch
import numpy as np
import math
import copy
import torch_two_sample
import torch
from utils_latent_analysis_optimisation import GaussianKDE
from torch.distributions.distribution import Distribution


def kl_divergence(mu, logvar, hinge_kl_loss_lambda=0.5, average_batch=True, sum_latent=True):
    """
    Calculates the KL-divergence between the posterior and the prior analytically.
    """
    kl_loss = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)

    if hinge_kl_loss_lambda > 0.0:

        # Ignore the dimensions of which the KL-div is already under the
        # threshold, avoiding driving it down even further. Those values do
        # not have to be replaced by the threshold because that would not mean
        # anything to the gradient. That's why they are simply removed. This
        # confused me at first.
        kl_mask = (kl_loss > hinge_kl_loss_lambda).float()

        # Sum over the latent dimensions and average over the batch dimension
        hinge_kl_loss = (kl_mask * kl_loss)

    else:
        hinge_kl_loss = kl_loss

    if sum_latent:
        hinge_kl_loss = hinge_kl_loss.sum(dim=1)
        kl_loss = kl_loss.sum(dim=1)

    if average_batch:
        hinge_kl_loss = hinge_kl_loss.mean(dim=0)
        kl_loss = kl_loss.mean(dim=0)

    return kl_loss, hinge_kl_loss


def maximum_mean_discrepancy(latent_z):
    """
    Taken from: https://github.com/aktersnurra/information-maximizing-variational-autoencoders/blob/master/model/loss_functions/mmd_loss.py

    :param latent_z:
    :return:
    """
    x = torch.randn_like(latent_z).to(latent_z.device)
    y = latent_z
    x_kernel = gaussian_kernel(x, x)
    y_kernel = gaussian_kernel(y, y)
    xy_kernel = gaussian_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

    return mmd


def gaussian_kernel(x, y):
    """
    Gaussian kernel
    Taken from: https://github.com/aktersnurra/information-maximizing-variational-autoencoders/blob/master/model/loss_functions/mmd_loss.py
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-torch.tensor(kernel_input))


def sample_log_likelihood(latent_z, mu=None, logvar=None, reduce_latent_dim=True, reduce_batch_dim=True):
    """
    This function calculates the log likelihood of samples under the Normal
    Distribution, either parameterised by mu, logvar (posterior), else under the standard Normal (prior).
    """

    # Multi sample posterior case
    if latent_z.dim() == 3 and mu is not None and logvar is not None:
        if mu.dim() != 3:
            mu = mu.unsqueeze(1)
        if logvar.dim() != 3:
            logvar = logvar.unsqueeze(1)

    # Under a posterior z under q(z|x)
    if logvar is not None and mu is not None:
        # N(z| mu, sigma) = [-1 / 2(log var + log 2pi + (x - mu) ^ 2 / var)]
        var = logvar.exp()
        likelihood = (- 0.5 * (logvar + math.log(2 * math.pi) + ((latent_z - mu) ** 2 / var)))

    # Under a prior
    else:
        # N(z | 0, 1) = [-1/2 ( log 2pi + x^2)]
        likelihood = - 0.5 * (math.log(2 * math.pi) + (latent_z ** 2))

    # Reduce the latent dimension (log sum)
    if reduce_latent_dim:
        likelihood = likelihood.sum(dim=-1)

    if reduce_batch_dim:
        likelihood = likelihood.mean(dim=0)

    return likelihood


def approximate_marginal_KL(mu, logvar, latent_z, method="chen", dataset_size=None):
    log_q_z, _ = approximate_log_q_z(mu, logvar, latent_z, method=method,
                                     dataset_size=dataset_size, prod_marginals=False)

    log_p_z = sample_log_likelihood(latent_z, reduce_latent_dim=True, reduce_batch_dim=False)

    marginal_KL = (log_q_z - log_p_z).mean()

    return marginal_KL


def approximate_total_correlation(mu, logvar, latent_z, method="chen", dataset_size=None):
    #
    # From Chen et al. (2019), Isolating Sources of Disentanglement
    # KL(q(z) || prod q(z_i)) <- mutual information, or dependence, between the latent dimensions

    # log q(z), log prod q(z_i)
    log_q_z, log_q_z_prod_marginals = approximate_log_q_z(mu, logvar, latent_z, method=method,
                                                          dataset_size=dataset_size,
                                                          prod_marginals=True)

    total_correlation = (log_q_z - log_q_z_prod_marginals).mean()

    return total_correlation


def approximate_log_q_z(mu, logvar, latent_z, method="chen", dataset_size=42068, prod_marginals=False,
                        reduce_mean=True):
    """
    Approximate E_q(z) [ log q (z) ]. This evaluates all samples x->z under q(z), which on itself
    relies on all data points. The "ideal" estimator would be:

    1/N sum_i^N [log 1/N sum_j^N q(z_i|z_j)],
        where the only thing that makes this an estimate is
        the fact that we use the empirical data distribution,
        instead of the true one (which we will never have).

    So, in the ideal case, latent_z and mu, logvar, would have the same dimensions and be
    of the size of the whole dataset. Otherwise, we could estimate. One method is proposed by
    Chen et al. (2019), specifically designed for a batch of data.

    Shapes:
        - mu, logvar: [x_batch, latent_dim]
        - latent_z: [z_batch, latent_dim]

        --> x_batch might in a lot of cases be the same as z_batch but does not need to be
            in that case the samples may come from (partially) different distributions than the
            ones defined by mu and logvar
    """
    if method == "chen":
        assert dataset_size is not None, "if using method 'chen', you need to provide a dataset size"

    # print("dataset size", dataset_size)

    # Get shapes
    x_batch, n_dim = mu.shape

    """
    # from: https://github.com/rtqichen/beta-tcvae/blob/1a3577dbb14642b9ac27010928d12132d0c0fb91/vae_quant.py#L225
    # minibatch weighted sampling
    logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
    logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
    """

    # Orient it as a row [1, x_batch, latent_dim]
    mu_exp, logvar_exp = mu.unsqueeze(0), logvar.unsqueeze(0)

    # Orient it as a column [z_batch, 1, latent_dim]
    latent_z_exp = latent_z.unsqueeze(1)

    # Evaluate the log probability q(z_i|mu_j, sigma_j) for all z_i and all (mu_j, sigma_j)
    # [z_batch, x_batch, latent_dim]
    log_dens = sample_log_likelihood(latent_z_exp, mu_exp, logvar_exp,
                                     reduce_latent_dim=False, reduce_batch_dim=False)
    # print(log_dens.sum(dim=-1) / 32)
    # print(log_dens.min(), log_dens.max())

    log_q_z_prod_marg = None

    # Log prod q(z_i) => Sum log q(z_j)
    if prod_marginals:
        # Reduce x_batch dim, then latent_dim (happens later in the code at **)
        log_q_z_prod_marg = torch.logsumexp(log_dens, dim=1, keepdim=False)

    # Reduce latent and then x_batch dim
    log_q_z = torch.logsumexp(log_dens.sum(dim=2), dim=1, keepdim=False)

    # We assume to have been given an batch, and use a weighted version as proposed in
    # Isolating Sources of Disentanglement (Chen et al., 2019)
    if method == "chen":
        if prod_marginals:
            # ** sum reduce the latent_dim
            log_q_z_prod_marg = (log_q_z_prod_marg - math.log(x_batch * dataset_size)).sum(dim=1)

        log_q_z = log_q_z - math.log(x_batch * dataset_size)

    # Assuming the data we got was the whole dataset
    else:
        if prod_marginals:
            # ** sum reduce the latent_dim
            log_q_z_prod_marg = (log_q_z_prod_marg - math.log(x_batch)).sum(dim=1)

        log_q_z = log_q_z - math.log(x_batch)

    if reduce_mean:
        log_q_z = log_q_z.mean()
        log_q_z_prod_marg = log_q_z_prod_marg.mean()

    return log_q_z, log_q_z_prod_marg


class LossTermManager(torch.nn.Module):
    """
        This class manages the loss function by keeping track of:
            - optimisers
            - lr schedulers
            - constraints & constraint optimisers
            - parameters and their schedules

        and can assemble the total loss (on which the VAE parameters bases its updates) according
        to the objective that is set.
        """

    def __init__(self, vae_model, config):
        super(LossTermManager, self).__init__()

        self.vae_model = vae_model
        self.objective = config.objective
        self.ddp = config.ddp

        # TOTAL LOSS
        # scheduler will take care of the actual LR
        self.total_loss_optimiser = torch.optim.AdamW(vae_model.parameters(), lr=1.0)
        self.total_loss_scheduler = get_scheduler(self.total_loss_optimiser,
                                                  warmup_updates=config.lr_warmup_updates,
                                                  lr_scheduler=config.lr_scheduler,
                                                  linear_lr_sched_grad_total_steps=config.linear_lr_sched_grad_total_steps,
                                                  lr_scheduler_type=config.lr_scheduler_type,
                                                  lr=config.lr)

        self.manager = dict()
        self.free_bits_pd = config.fb_b_vae_free_bits_pd

        print("*" * 80)
        print("Free bits pd", config.fb_b_vae_free_bits_pd)
        print("*" * 80)

        # Automatic Mixed Precision scaler
        self.scaler = GradScaler(enabled=config.use_amp)

        if self.objective == "beta-vae" or self.objective == "free-bits-beta-vae":
            # beta * KL term
            if config.b_vae_beta_constant_linear_lagrangian in ["constant", "linear", "cyclical"]:
                if config.b_vae_beta_constant_linear_lagrangian == "linear":
                    print(f"Setting parameter schedule for beta-vae KL term, schedule: "
                          f"{config.b_vae_beta_constant_linear_lagrangian}, ramp length: "
                          f"{config.b_vae_beta_ramp_len_grad_steps}, beta value: {config.b_vae_beta}")
                elif config.b_vae_beta_constant_linear_lagrangian == "cyclical":
                    print(f"Cyclical KL annealing: "
                          f"n_cycles: {config.max_epochs if config.max_epochs > 0 else 50}"
                          f"clycle_len: {config.max_train_steps_epoch_per_rank if config.max_train_steps_epoch_per_rank > 0 else 10e3}"
                          f"decrease_increase: increase"
                          f"value: {config.b_vae_beta}")
                self.manager["beta_KL"] = ParameterScheduler(
                    n_cycles=config.max_epochs if config.max_epochs > 0 else 50,
                    cycle_len=config.max_train_steps_epoch_per_rank if config.max_train_steps_epoch_per_rank > 0 else 10e3,
                    cyc_lin_con=config.b_vae_beta_constant_linear_lagrangian,
                    ramp_len_grad_steps=config.b_vae_beta_ramp_len_grad_steps,
                    warmup_len_grad_steps=0,
                    decrease_increase="increase",
                    value=config.b_vae_beta,
                    name="beta")

            elif config.b_vae_beta_constant_linear_lagrangian == "lagrangian":
                target_kl = config.b_vae_kl_lagrangian_target_pd * config.latent_size
                constraint = Constraint(target_kl, "ge", alpha=config.b_vae_kl_lagrangian_alpha)
                optimiser = ConstraintOptimizer(torch.optim.RMSprop, constraint.parameters(),
                                                config.b_vae_kl_lagrangian_lr)
                print(f"Setting Lagrangian for beta-vae KL term, target: {config.b_vae_kl_lagrangian_target_pd} x"
                      f"{config.latent_size} = {target_kl}, with relation 'ge' and alpha = {config.b_vae_kl_lagrangian_alpha}"
                      f"and lr = {config.b_vae_kl_lagrangian_lr}")
                self.manager["beta_KL"] = {
                    "constraint": constraint,
                    "optimiser": optimiser
                }

        elif self.objective == "beta-tc-vae":
            # alpha * MI term
            if config.b_tc_vae_alpha_constant_linear_lagrangian in ["constant", "linear"]:
                print(f"Setting parameter schedule for beta-tc-vae MI term (alpha), schedule: "
                      f"{config.b_tc_vae_alpha_constant_linear_lagrangian}, ramp length: "
                      f"{config.b_tc_vae_alpha_ramp_len_grad_steps}, alpha value: {config.b_tc_vae_alpha}, ramp type:"
                      f"{config.b_tc_vae_alpha_ramp_type}")
                self.manager["alpha_MI"] = ParameterScheduler(
                    cyc_lin_con=config.b_tc_vae_alpha_constant_linear_lagrangian,
                    ramp_len_grad_steps=config.b_tc_vae_alpha_ramp_len_grad_steps,
                    decrease_increase=config.b_tc_vae_alpha_ramp_type,
                    warmup_len_grad_steps=0,
                    value=config.b_tc_vae_alpha,
                    name="alpha")

            elif config.b_tc_vae_alpha_constant_linear_lagrangian == "lagrangian":
                constraint = Constraint(config.b_tc_vae_MI_lagrangian_target,
                                        config.b_tc_vae_MI_lagrangian_relation,
                                        alpha=config.b_tc_vae_MI_lagrangian_alpha)
                optimiser = ConstraintOptimizer(torch.optim.RMSprop, constraint.parameters(),
                                                config.b_tc_vae_MI_lagrangian_lr)
                print(f"Setting Lagrangian for beta-tc-vae MI term (alpha), target: "
                      f"{config.b_tc_vae_MI_lagrangian_target}"
                      f" with relation {config.b_tc_vae_MI_lagrangian_relation} and "
                      f"alpha = {config.b_tc_vae_MI_lagrangian_alpha}"
                      f"and lr = {config.b_tc_vae_MI_lagrangian_lr}")
                self.manager["alpha_MI"] = {
                    "constraint": constraint,
                    "optimiser": optimiser
                }

            # beta * TC term
            if config.b_tc_vae_beta_constant_linear_lagrangian in ["constant", "linear"]:
                print(f"Setting parameter schedule for beta-tc-vae TC term (beta), schedule: "
                      f"{config.b_tc_vae_beta_constant_linear_lagrangian}, ramp length: "
                      f"{config.b_tc_vae_beta_ramp_len_grad_steps}, beta value: {config.b_tc_vae_beta}, ramp type:"
                      f"{config.b_tc_vae_beta_ramp_type}")
                self.manager["beta_TC"] = ParameterScheduler(
                    cyc_lin_con=config.b_tc_vae_beta_constant_linear_lagrangian,
                    ramp_len_grad_steps=config.b_tc_vae_beta_ramp_len_grad_steps,
                    decrease_increase=config.b_tc_vae_beta_ramp_type,
                    value=config.b_tc_vae_beta,
                    warmup_len_grad_steps=0,
                    name="beta")

            # gamma * DimKL
            if config.b_tc_vae_gamma_constant_linear_lagrangian in ["constant", "linear"]:
                print(f"Setting parameter schedule for beta-tc-vae Dim. KL term (gamma), schedule: "
                      f"{config.b_tc_vae_gamma_constant_linear_lagrangian}, ramp length: "
                      f"{config.b_tc_vae_gamma_ramp_len_grad_steps}, gamma value: {config.b_tc_vae_gamma}, ramp type:"
                      f"{config.b_tc_vae_gamma_ramp_type}")
                self.manager["gamma_DimKL"] = ParameterScheduler(
                    cyc_lin_con=config.b_tc_vae_gamma_constant_linear_lagrangian,
                    ramp_len_grad_steps=config.b_tc_vae_gamma_ramp_len_grad_steps,
                    decrease_increase=config.b_tc_vae_gamma_ramp_type,
                    value=config.b_tc_vae_gamma,
                    name="gamma")
            elif config.b_tc_vae_gamma_constant_linear_lagrangian == "lagrangian":
                target = config.b_tc_vae_Dim_KL_lagrangian_target_pd * config.latent_size
                constraint = Constraint(target, 'ge', alpha=config.b_tc_vae_Dim_KL_lagrangian_alpha)
                optimiser = ConstraintOptimizer(torch.optim.RMSprop, constraint.parameters(),
                                                config.b_tc_vae_Dim_KL_lagrangian_lr)
                self.manager["gamma_DimKL"] = {
                    "constraint": constraint,
                    "optimiser": optimiser
                }
                print(
                    f"Setting Lagrangian for beta-tc-vae Dim. KL term, target: {config.b_tc_vae_Dim_KL_lagrangian_target_pd} x"
                    f"{config.latent_size} = {target}, with relation 'ge' and alpha = {config.b_tc_vae_Dim_KL_lagrangian_alpha}"
                    f"and lr = {config.b_tc_vae_Dim_KL_lagrangian_lr}")

        elif self.objective == "hoffman":
            # alpha * MI
            if config.hoffman_vae_alpha_constant_linear_lagrangian in ["constant", "linear"]:
                print(f"Setting parameter schedule for Hoffman MI (alpha), schedule: "
                      f"{config.hoffman_vae_alpha_constant_linear_lagrangian}, ramp length: "
                      f"{config.hoffman_vae_alpha_ramp_len_grad_steps}, alpha value: {config.hoffman_vae_alpha}, ramp type:"
                      f"{config.hoffman_vae_alpha_ramp_type}")
                self.manager["alpha_MI"] = ParameterScheduler(
                    cyc_lin_con=config.hoffman_vae_alpha_constant_linear_lagrangian,
                    ramp_len_grad_steps=config.hoffman_vae_alpha_ramp_len_grad_steps,
                    decrease_increase=config.hoffman_vae_alpha_ramp_type,
                    value=config.hoffman_vae_alpha,
                    name="alpha")
            elif config.hoffman_vae_alpha_constant_linear_lagrangian == "lagrangian":
                constraint = Constraint(config.hoffman_vae_MI_lagrangian_target,
                                        config.hoffman_vae_MI_lagrangian_relation,
                                        alpha=config.hoffman_vae_MI_lagrangian_alpha)
                optimiser = ConstraintOptimizer(torch.optim.RMSprop, constraint.parameters(),
                                                config.hoffman_vae_MI_lagrangian_lr)
                self.manager["alpha_MI"] = {
                    "constraint": constraint,
                    "optimiser": optimiser
                }
                print(
                    f"Setting Lagrangian for alpha MI term in Hoffman VAE, target: {config.hoffman_vae_MI_lagrangian_target},"
                    f" with relation {config.hoffman_vae_MI_lagrangian_relation} and "
                    f"alpha = {config.hoffman_vae_MI_lagrangian_alpha}"
                    f"and lr = {config.hoffman_vae_MI_lagrangian_lr}")

            # beta * marginal KL
            if config.hoffman_vae_beta_constant_linear_lagrangian in ["constant", "linear"]:
                print(f"Setting parameter schedule for Hoffman marginal KL (beta), schedule: "
                      f"{config.hoffman_vae_beta_constant_linear_lagrangian}, ramp length: "
                      f"{config.hoffman_vae_beta_ramp_len_grad_steps}, beta value: {config.hoffman_vae_beta}, ramp type:"
                      f"{config.hoffman_vae_beta_ramp_type}")
                self.manager["beta_marg_KL"] = ParameterScheduler(
                    cyc_lin_con=config.hoffman_vae_beta_constant_linear_lagrangian,
                    ramp_len_grad_steps=config.hoffman_vae_beta_ramp_len_grad_steps,
                    decrease_increase=config.hoffman_vae_beta_ramp_type,
                    value=config.hoffman_vae_beta,
                    name="beta")
            elif config.hoffman_vae_beta_constant_linear_lagrangian == "lagrangian":
                constraint = Constraint(config.hoffman_vae_marg_KL_lagrangian_target,
                                        config.hoffman_vae_marg_KL_lagrangian_relation,
                                        alpha=config.hoffman_vae_marg_KL_lagrangian_alpha)
                optimiser = ConstraintOptimizer(torch.optim.RMSprop, constraint.parameters(),
                                                config.hoffman_vae_marg_KL_lagrangian_lr)
                self.manager["beta_marg_KL"] = {
                    "constraint": constraint,
                    "optimiser": optimiser
                }
                print(
                    f"Setting Lagrangian for beta marginal KL term in Hoffman VAE, target: "
                    f"{config.hoffman_vae_marg_KL_lagrangian_target},"
                    f" with relation {config.hoffman_vae_marg_KL_lagrangian_relation} and "
                    f"alpha = {config.hoffman_vae_marg_KL_lagrangian_alpha}"
                    f"and lr = {config.hoffman_vae_marg_KL_lagrangian_lr}")

        # ELBO minimisation with several options for constraints: MMD, distortion, rate & 1D KDE marginal KL proxy
        elif self.objective == "elbo-constraint-optim":
            # ELBO constraint
            if config.use_elbo_constraint:
                elbo_constraint = Constraint(config.elbo_constraint_value,
                                             config.elbo_constraint_relation,
                                             alpha=config.elbo_constraint_alpha)
                elbo_optimiser = ConstraintOptimizer(torch.optim.RMSprop, elbo_constraint.parameters(),
                                                     config.elbo_constraint_lr)
                self.manager["elbo_constraint"] = {
                    "constraint": elbo_constraint,
                    "optimiser": elbo_optimiser
                }
                print(f"Setting distortion constraint with values:"
                      f"constraint val: {config.elbo_constraint_value}"
                      f"lr {config.elbo_constraint_lr}, alpha {config.elbo_constraint_alpha}, "
                      f"rel: {config.elbo_constraint_relation}")

            # Distortion constraint
            if config.use_distortion_constraint:
                distortion_constraint = Constraint(config.distortion_constraint_value,
                                                   config.distortion_constraint_relation,
                                                   alpha=config.distortion_constraint_alpha)
                distortion_optimiser = ConstraintOptimizer(torch.optim.RMSprop, distortion_constraint.parameters(),
                                                           config.distortion_constraint_lr)
                self.manager["distortion_constraint"] = {
                    "constraint": distortion_constraint,
                    "optimiser": distortion_optimiser
                }
                print(f"Setting distortion constraint with values:"
                      f"constraint val: {config.distortion_constraint_value}"
                      f"lr {config.distortion_constraint_lr}, alpha {config.distortion_constraint_alpha}, "
                      f"rel: {config.distortion_constraint_relation}")

            # Rate constraint
            if config.use_rate_constraint:
                rate_constraint = Constraint(config.rate_constraint_value,
                                             config.rate_constraint_relation,
                                             alpha=config.rate_constraint_alpha)
                rate_optimiser = ConstraintOptimizer(torch.optim.RMSprop, rate_constraint.parameters(),
                                                     config.rate_constraint_lr)
                self.manager["rate_constraint"] = {
                    "constraint": rate_constraint,
                    "optimiser": rate_optimiser
                }

                print(f"Setting rate constraint with values:"
                      f"constraint val: {config.rate_constraint_value}"
                      f"lr {config.rate_constraint_lr}, alpha {config.rate_constraint_alpha}, "
                      f"rel: {config.rate_constraint_relation}")

            # MMD constraint
            if config.use_mmd_constraint:
                mmd_constraint = Constraint(config.mmd_constraint_value,
                                            config.mmd_constraint_relation,
                                            alpha=config.mmd_constraint_alpha)
                mmd_optimiser = ConstraintOptimizer(torch.optim.RMSprop, mmd_constraint.parameters(),
                                                    config.mmd_constraint_lr)
                self.manager["mmd_constraint"] = {
                    "constraint": mmd_constraint,
                    "optimiser": mmd_optimiser
                }

                print(f"Setting MMD constraint with values:"
                      f"constraint val: {config.mmd_constraint_value}"
                      f"lr {config.mmd_constraint_lr}, alpha {config.mmd_constraint_alpha}, "
                      f"rel: {config.mmd_constraint_relation}")

            # KDE 1D dim marginal KL constraint
            if config.use_kde1d_constraint:
                kde1d_constraint = Constraint(config.kde1d_constraint_value,
                                              config.kde1d_constraint_relation,
                                              alpha=config.kde1d_constraint_alpha)
                kde1d_optimiser = ConstraintOptimizer(torch.optim.RMSprop, kde1d_constraint.parameters(),
                                                      config.kde1d_constraint_lr)
                self.manager["kde1d_constraint"] = {
                    "constraint": kde1d_constraint,
                    "optimiser": kde1d_optimiser
                }
                print(f"Setting 1D KDE dim marginal KL constraint with values:"
                      f"constraint val: {config.kde1d_constraint_value}"
                      f"lr {config.kde1d_constraint_lr}, alpha {config.kde1d_constraint_alpha}, "
                      f"rel: {config.kde1d_constraint_relation}")

            # print(f"Setting Lagrangian for ELBO (>= relation), Rate (>= relation):"
            #       f"\nELBO: target val: {config.elbo_constraint_value}, alpha: {config.elbo_constraint_alpha}, lr: {config.elbo_constraint_lr}"
            #       f"\nRate:  target val: {config.rate_constraint_value}, alpha: {config.rate_constraint_alpha}, lr: {config.rate_constraint_lr}")

    def multi_sample_vae_forward(self, input_ids, attention_mask, return_exact_match=False,
                                 n_samples=100, return_attention_to_latent=False):

        # Encode these input ids and sample <n_samples> for each x
        enc_out = self.vae_model.encoder.encode(input_ids, attention_mask,
                                                n_samples=n_samples,
                                                return_log_q_z_x=True,
                                                return_log_q_z=True,
                                                return_log_p_z=True,
                                                return_embeddings=False)

        # Unpack the tensors we need, shapes: [batch, n_samples, latent_dim], [batch, n_samples], [batch, n_samples]
        post_samples, post_log_p_z, post_log_q_z_x = enc_out["latent_z"], enc_out["log_p_z"], enc_out["log_q_z_x"]

        # Multi sample decode
        log_p_x_z, dec_out = [], None
        for i in range(n_samples):
            latent_z = post_samples[:, i, :]

            dec_out = self.vae_model.decoder(latent_z, input_ids, attention_mask,
                                             return_attention_to_latent=return_attention_to_latent,
                                             return_exact_match=return_exact_match,
                                             return_cross_entropy=True,
                                             return_reconstruction_loss=True,
                                             reduce_batch_reconstruction_loss=False,
                                             reduce_seq_dim_ce="mean",
                                             reduce_seq_dim_exact_match="mean",
                                             reduce_batch_dim_exact_match="mean",
                                             reduce_batch_dim_ce="mean",
                                             labels=copy.copy(input_ids))

            # log likelihood = - reconstruction_loss
            log_p_x_z.append(- dec_out["reconstruction_loss"])

        log_p_x_z = torch.stack(log_p_x_z, dim=1)

        iw_ll = self.iw_log_p_x(log_p_x_z, log_p_z=enc_out["log_p_z"], log_q_z_x=enc_out["log_q_z_x"])

        # normalise for length
        lens = torch.sum(attention_mask, dim=-1)
        iw_ll_not_zero = iw_ll[lens > 0.0]
        lens_not_zero = lens[lens > 0.0]

        if len(lens[lens == 0]) > 0:
            print("-> Zero length sequence generation!")

        iw_ll_p_w = iw_ll_not_zero / lens_not_zero

        vae_out = {**enc_out, **dec_out, "log_p_x_z": log_p_x_z,
                   "lens": lens.float(), "iw_ll": iw_ll_not_zero, "iw_ll_mean": iw_ll_not_zero.mean().cpu().item(),
                   "iw_ll_p_w": iw_ll_p_w, "iw_ll_p_w_mean": iw_ll_p_w.mean().cpu().item()}

        return vae_out

    @staticmethod
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
        # so where < lens, should be true, all after should be 0.0
        mask = (ind <= lens)

        # Pad the predictions (setting all tokens after </s> to <pad>)
        predictions[~mask] = pad_token_id

        return predictions, mask, lens.flatten()

    def get_tts_mmmd(self, latent_z):
        latent_z = latent_z.squeeze(1)
        prior_sample = torch.randn(latent_z.shape).to(latent_z.device)
        alphas = [0.1 * i for i in range(15)]

        """
        torch two samples talks about alpha, while this paper talks about estimating kernel width (is that gamma?)
        paper: https://papers.nips.cc/paper/2012/file/dbe272bab69f8e13f14b405e038deb64-Paper.pdf
        kernel width might be

        from torch_two_sample import util

        sample_12 = torch.cat((latents_1, latents_2), 0)
        distances = util.pdist(sample_12, sample_12, norm=2)
        plt.hist(distances.flatten().numpy(), bins=30)
        gamma = median_distance = torch.median(distances).item()
        alpha = 1 / (2 * gamma**2)
        print(alpha)
        """

        # print(prior_sample.shape, latent_z.shape)

        n_1, n_2 = len(latent_z), len(prior_sample)
        MMD_stat = torch_two_sample.statistics_diff.MMDStatistic(n_1, n_2)
        tts_mmd = MMD_stat(latent_z, prior_sample, alphas, ret_matrix=False)

        # print("**** TORCH TWO SAMPLE MMD", tts_mmd)

        return tts_mmd

    @staticmethod
    def get_1d_kde_dim_marg_kl_estimate(latents, log_p_z, bw="scott", device="cuda:0"):
        latents_1D = latents.reshape(-1, 1)
        kde = GaussianKDE(X=latents_1D, bw=bw, device=device)
        log_q_z_kde = kde.score_samples(Y=latents_1D).mean()
        dim_marg_kl = log_q_z_kde - log_p_z
        return dim_marg_kl

    def forward(self, input_ids, attention_mask, return_exact_match=False, decoder_only=False, eval_iw_ll_x_gen=False,
                return_posterior_stats=True, device_name="cuda:0", iw_ll_n_samples=10,
                return_attention_to_latent=False, train=True, max_seq_len_x_gen=64, save_latents=False):
        """
        This forward implements the forward of the VAE in train or validation mode and then assembles
        the loss / stats that are relevant for training. This is separate from the normal VAE forward as it is focused
        on assembling quantities that are relevant for training and optimisation, rather than practical features such as
        returning embeddings etc.

        There are two important settings:
            - decoder_only: if True, the model is a regular language model, everything related to the encoder is skipped

            - n_samples: the number of samples used to approximate the log likelihood of the data
                if > 1: importance weighted ll
                else: normal elbo

        """

        # Decoder only mode
        if decoder_only:
            vae_out = self.vae_model.decoder_only_forward(input_ids=input_ids, attention_mask=attention_mask)

        # Normal VAE mode
        else:
            vae_out = self.multi_sample_vae_forward(input_ids=input_ids, attention_mask=attention_mask,
                                                    return_exact_match=return_exact_match, n_samples=iw_ll_n_samples,
                                                    return_attention_to_latent=return_attention_to_latent)

            if self.objective == "elbo-constraint-optim" or "mmd-constraint-optim":
                if vae_out["latent_z"].dim() == 3:
                    latents = vae_out["latent_z"][:, 0, :]
                else:
                    latents = vae_out["latent_z"]
                tts_mmd = self.get_tts_mmmd(latents)
                kde_1d_marginal_kl = self.get_1d_kde_dim_marg_kl_estimate(latents=latents,
                                                                          log_p_z=vae_out["log_p_z"].mean(),
                                                                          bw="scott", device=device_name)
            else:
                tts_mmd = None
                kde_1d_marginal_kl = None

            if return_posterior_stats:
                post_stats = self.vae_model.calc_posterior_stats(mu=vae_out["mu"], logvar=vae_out["logvar"])
                vae_out = {**vae_out, **post_stats}

            # Shapes at this point
            # vae_out["reconstruction_loss"] (- log p_x_z), log_q_z_x, log_p_z = [batch, n_samples]
            # log_q_z, log_q_z_prod_marg = 0-dim (number)
            loss_dict = self.assemble_loss(reconstruction_loss=vae_out["reconstruction_loss"].mean(),
                                           mu=vae_out["mu"], logvar=vae_out["logvar"],
                                           log_p_z=vae_out["log_p_z"].mean(),
                                           log_q_z_x=vae_out["log_q_z_x"].mean(),
                                           log_q_z=vae_out["log_q_z"],
                                           tts_mmd=tts_mmd,
                                           log_q_z_prod_marg=vae_out["log_q_z_prod_marg"],
                                           mmd=None,
                                           kde_1d_marginal_kl=kde_1d_marginal_kl)

            vae_out = {**vae_out, **loss_dict}

            if train is True:
                # these are not tracked with train
                del vae_out["mu"]
                del vae_out["logvar"]

                if not save_latents:
                    del vae_out["latent_z"]
                else:
                    vae_out["latent_z"] = vae_out["latent_z"].detach()

                del vae_out["cross_entropy"]  # spurious

                for k in list(vae_out.keys()):
                    if torch.is_tensor(vae_out[k]) and k != "total_loss" \
                            and k not in ["iw_ll", "iw_ll_p_w", "lens", "latent_z"]:
                        vae_out[k] = vae_out[k].mean().cpu().item()
                    elif vae_out[k] is None:
                        del vae_out[k]

            # Evaluate the likelihood given to samples that originate from the generative model
            # i.w. log p(x_gen)
            if eval_iw_ll_x_gen:
                # Without grad, because used as 'external' to the model
                with torch.no_grad():
                    # Sample from the model by decoding from prior auto-regressively with sampling
                    out = self.vae_model(return_reconstruction_loss=False,
                                         return_posterior_stats=False,
                                         auto_regressive=True,
                                         max_seq_len=max_seq_len_x_gen,
                                         return_predictions=True,
                                         nucleus_sampling=True,
                                         top_k=0,  # no filtering
                                         top_p=1.0,  # no filtering
                                         decode_sample_from_prior=True,
                                         n_prior_samples=input_ids.shape[0],
                                         device_name=device_name)

                    # this prepares the predictions as input samples
                    padded_predictions, mask, lens = self.make_batch_from_model_samples(out["predictions"])

                # Should use gradients, but for now only used in
                vae_out_x_gen = self.multi_sample_vae_forward(input_ids=padded_predictions.to(device_name),
                                                              attention_mask=mask.to(device_name),
                                                              return_exact_match=False, n_samples=eval_iw_ll_x_gen,
                                                              return_attention_to_latent=False)

                # Log the mean of the batch as well as the values to add to histogram
                vae_out["iw_ll_x_gen_mean"] = vae_out_x_gen["iw_ll_mean"]
                vae_out["iw_ll_x_gen"] = vae_out_x_gen["iw_ll"].detach()
                vae_out["iw_ll_x_gen_p_w_mean"] = vae_out_x_gen["iw_ll_p_w_mean"]
                vae_out["iw_ll_x_gen_p_w"] = vae_out_x_gen["iw_ll_p_w"].detach()
                vae_out["lens_x_gen"] = vae_out_x_gen["lens"].detach()

            # print("-" * 40)
            # print("VAE OUT")
            # for k, v in vae_out.items():
            #     if torch.is_tensor(v):
            #         print(k, ":", v.shape)
            #     else:
            #         print(k, v)
            # print("-" * 40)
            #
            # quit()

        # Delete all that is None
        key_list = list(vae_out.keys())
        for k in key_list:
            if vae_out[k] is None:
                del vae_out[k]

        # TODO: check this
        if return_attention_to_latent:
            if "self_attention_to_latent" in vae_out:
                # avg over heads and layers
                # vae_out["attention_to_latent"] = vae_out["self_attention_to_latent"].mean(dim=1).mean(dim=1)
                del vae_out["self_attention_to_latent"]

            elif "cross_attention_to_latent" in vae_out:
                # avg over heads and layers
                # vae_out["attention_to_latent"] = vae_out["cross_attention_to_latent"].mean(dim=1).mean(dim=1)
                del vae_out["cross_attention_to_latent"]

        return vae_out

    def iw_log_p_x(self, log_p_x_z, log_p_z, log_q_z_x):
        """
        Importance weighted likelihood.

        log_p_x_z, log_p_z, log_q_z_x: [batch, n_samples]
        """
        n_samples = log_p_x_z.shape[1]
        iw_frac = log_p_x_z + log_p_z - log_q_z_x

        # Reduce the sample dimension with logsumexp, leaves shape [batch_size]
        iw_likelihood = torch.logsumexp(iw_frac, dim=-1) - np.log(n_samples)
        return iw_likelihood

    def assemble_loss(self, reconstruction_loss, mu, logvar,
                      log_p_z=None, log_q_z_x=None, tts_mmd=None,
                      log_q_z=None, log_q_z_prod_marg=None, mmd=None, kde_1d_marginal_kl=None):
        """
        # log_p_x_z, log_q_z_x, log_p_z = [batch, n_samples]
        # log_q_z, log_q_z_prod_marg = 0-dim (numbers)

        """

        loss_dict = dict()

        # Calculate the KL divergence analytically
        kl_analytical, fb_kl_analytical = kl_divergence(mu, logvar,
                                                        hinge_kl_loss_lambda=self.free_bits_pd,
                                                        average_batch=True, sum_latent=True)

        # Non-analytical KL computation
        # kl_loss = log_q_z_x - log_p_z
        elbo = - (reconstruction_loss + kl_analytical)

        mi, tc, dim_kl, marginal_kl = None, None, None, None
        if log_q_z_x is not None and log_q_z is not None:
            mi = log_q_z_x - log_q_z
        if log_q_z is not None and log_q_z_prod_marg is not None:
            tc = log_q_z - log_q_z_prod_marg
        if log_q_z_prod_marg is not None and log_p_z is not None:
            dim_kl = log_q_z_prod_marg - log_p_z
        if log_q_z is not None and log_p_z is not None:
            marginal_kl = log_q_z - log_p_z

        # Autoencoder
        if self.objective == "autoencoder":
            total_loss = reconstruction_loss

        # VAE
        elif self.objective == "vae":
            total_loss = -elbo

        # ELBO with constraints
        elif self.objective == "elbo-constraint-optim" or self.objective == "mmd-constraint-optim":
            if self.objective == "elbo-constraint-optim":
                total_loss = -elbo
            else:
                total_loss = tts_mmd

            loss_dict["tts_mmd_loss"] = tts_mmd
            loss_dict["kde1d_loss"] = kde_1d_marginal_kl

            if "elbo_constraint" in self.manager:
                loss_dict["elbo_multiplier"] = self.manager["elbo_constraint"]["constraint"].multiplier
                elbo_lagrange_loss = self.manager["elbo_constraint"]["constraint"](elbo)
                loss_dict["elbo_lagrange_loss"] = elbo_lagrange_loss
                total_loss = total_loss + elbo_lagrange_loss

            if "distortion_constraint" in self.manager:
                loss_dict["distortion_multiplier"] = self.manager["distortion_constraint"]["constraint"].multiplier
                distortion_lagrange_loss = self.manager["distortion_constraint"]["constraint"](reconstruction_loss)
                loss_dict["distortion_lagrange_loss"] = distortion_lagrange_loss
                total_loss = total_loss + distortion_lagrange_loss

            if "mmd_constraint" in self.manager:
                loss_dict["mmd_multiplier"] = self.manager["mmd_constraint"]["constraint"].multiplier
                mmd_lagrange_loss = self.manager["mmd_constraint"]["constraint"](tts_mmd)

                loss_dict["mmd_lagrange_loss"] = mmd_lagrange_loss
                total_loss = total_loss + mmd_lagrange_loss

            if "rate_constraint" in self.manager:
                loss_dict["rate_multiplier"] = self.manager["rate_constraint"]["constraint"].multiplier
                rate_lagrange_loss = self.manager["rate_constraint"]["constraint"](kl_analytical)
                loss_dict["rate_lagrange_loss"] = rate_lagrange_loss
                total_loss = total_loss + rate_lagrange_loss

            if "kde1d_constraint" in self.manager:
                loss_dict["kde1d_multiplier"] = self.manager["kde1d_constraint"]["constraint"].multiplier
                kde1d_lagrange_loss = self.manager["kde1d_constraint"]["constraint"](kde_1d_marginal_kl)
                loss_dict["kde1d_lagrange_loss"] = kde1d_lagrange_loss
                total_loss = total_loss + kde1d_lagrange_loss

        # (Free bits) Beta-VAE
        elif self.objective == "beta-vae" or self.objective == "free-bits-beta-vae":

            if self.objective == "free-bits-beta-vae":
                kl_loss = fb_kl_analytical
            else:
                kl_loss = kl_analytical

            beta_kl, lagrange_loss, multiplier, beta = None, None, None, None
            # Parameter schedule
            if isinstance(self.manager["beta_KL"], ParameterScheduler):
                beta = self.manager["beta_KL"].current_val
                beta_kl = beta * kl_loss
                total_loss = reconstruction_loss + beta_kl

            # Lagrangian
            else:
                multiplier = self.manager["beta_KL"]["constraint"].multiplier
                lagrange_loss = self.manager["beta_KL"]["constraint"](kl_loss)
                total_loss = (reconstruction_loss + kl_loss) + lagrange_loss

            if multiplier is not None and lagrange_loss is not None:
                loss_dict["multiplier"] = multiplier.item() if torch.is_tensor(multiplier) else multiplier
                loss_dict["lagrange_loss"] = lagrange_loss.item() if torch.is_tensor(lagrange_loss) else lagrange_loss

            elif beta_kl is not None and beta is not None:
                loss_dict["KL"] = kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss
                loss_dict["beta_KL"] = beta_kl.item() if torch.is_tensor(beta_kl) else beta_kl
                loss_dict["beta"] = beta.item() if torch.is_tensor(beta) else beta

            """
            from the example notebook:
            https://github.com/EelcovdW/pytorch-constrained-opt/blob/master/VAE%20Example.ipynb
            # Loss
            elbo = ll - kl
            loss = (-elbo + kl_constraint(kl)).mean()
            """

            # print("total_loss", total_loss)
            # print("reconstruction_loss", reconstruction_loss)
            # print("beta_kl", beta_kl)
            # print("total_loss requires_grad", total_loss.requires_grad)
            # print("reconstruction_loss.requires_grad", reconstruction_loss.requires_grad)
            # print("beta_kl.requires_grad", beta_kl.requires_grad)

            # loss_dict["beta"] = beta.item() if torch.is_tensor(beta) else beta
            # loss_dict["beta_KL"] = beta_kl.item() if torch.is_tensor(beta_kl) else beta_kl
            # loss_dict["KL"] = kl_loss.item()

        # Beta-TC-VAE
        elif self.objective == "beta-tc-vae":
            # mi = log_q_z_x - log_q_z
            # Parameter schedule
            if isinstance(self.manager["alpha_MI"], ParameterScheduler):
                alpha = self.manager["alpha_MI"].current_val
                alpha_mi = alpha * mi
            # Lagrangian
            else:
                alpha = self.manager["alpha_MI"]["constraint"].multiplier

                alpha_mi = self.manager["alpha_MI"]["constraint"](mi)

            # Parameter schedule
            # tc = log_q_z - log_q_z_prod_marg
            beta = self.manager["beta_TC"].current_val
            beta_tc = beta * tc

            # dim_kl = log_q_z_prod_marg - log_p_z
            # Parameter schedule (linear or constant)
            if isinstance(self.manager["gamma_DimKL"], ParameterScheduler):
                gamma = self.manager["gamma_DimKL"].current_val
                gamma_dim_kl = gamma * dim_kl
            # Lagrangian
            else:
                gamma = self.manager["gamma_DimKL"]["constraint"].multiplier
                gamma_dim_kl = self.manager["gamma_DimKL"]["constraint"](dim_kl)

            # marginal_kl = log_q_z - log_p_z
            approx_kl = log_q_z_x - log_p_z

            total_loss = reconstruction_loss + alpha_mi + beta_tc + gamma_dim_kl

            loss_dict["alpha_MI"] = alpha_mi.item()
            loss_dict["alpha"] = alpha.item() if torch.is_tensor(alpha) else alpha
            loss_dict["TC"] = tc.item()
            loss_dict["beta_TC"] = beta_tc.item()
            loss_dict["beta"] = beta.item() if torch.is_tensor(beta) else beta
            loss_dict["dim_KL"] = dim_kl.item()
            loss_dict["gamma_dim_KL"] = gamma_dim_kl.item()
            loss_dict["gamma"] = gamma.item() if torch.is_tensor(gamma) else gamma
            loss_dict["marginal_KL"] = marginal_kl.item()
            loss_dict["approx_KL"] = approx_kl.item()

        # MMD-VAE
        elif self.objective == "mmd-vae":
            total_loss = reconstruction_loss + mmd

        # HOFFMAN VAE
        elif self.objective == "hoffman":
            # mi = log_q_z_x - log_q_z

            # loss_dict["MI"] = mi
            # loss_dict["marginal KL"] = marginal_kl

            total_loss = reconstruction_loss
            if isinstance(self.manager["alpha_MI"], ParameterScheduler):
                alpha = self.manager["alpha_MI"].current_val
                alpha_mi = mi * alpha
                total_loss = total_loss + alpha_mi

                loss_dict["alpha"] = alpha.item() if torch.is_tensor(alpha) else alpha
                loss_dict["alpha_MI"] = alpha_mi.item() if torch.is_tensor(alpha_mi) else alpha_mi

            else:
                multiplier = self.manager["alpha_MI"]["constraint"].multiplier
                lagrange_loss_alpha = self.manager["alpha_MI"]["constraint"](mi)
                total_loss = total_loss + mi + lagrange_loss_alpha

                loss_dict["multiplier (alpha)"] = multiplier.item() if torch.is_tensor(multiplier) else multiplier
                loss_dict["lagrange_loss_alpha"] = lagrange_loss_alpha.item() if torch.is_tensor(
                    lagrange_loss_alpha) else lagrange_loss_alpha

            if isinstance(self.manager["beta_marg_KL"], ParameterScheduler):
                beta = self.manager["beta_marg_KL"].current_val
                beta_marg_KL = marginal_kl * beta
                total_loss = total_loss + beta_marg_KL

                loss_dict["beta"] = beta.item() if torch.is_tensor(beta) else beta
                loss_dict["beta_marg_KL"] = beta_marg_KL.item() if torch.is_tensor(beta_marg_KL) else beta_marg_KL

            else:
                multiplier = self.manager["beta_marg_KL"]["constraint"].multiplier
                lagrange_loss_beta = self.manager["beta_marg_KL"]["constraint"](marginal_kl)
                total_loss = total_loss + marginal_kl + lagrange_loss_beta

                loss_dict["multiplier (beta)"] = multiplier.item() if torch.is_tensor(multiplier) else multiplier
                loss_dict["lagrange_loss_beta"] = lagrange_loss_beta.item() if torch.is_tensor(
                    lagrange_loss_beta) else lagrange_loss_beta

            hoffman_elbo = reconstruction_loss + mi + marginal_kl
            loss_dict["hoffman_elbo"] = hoffman_elbo.item()
            # loss_dict["marginal KL"] = marginal_kl.item()
            # loss_dict["MI"] = mi.item()
            loss_dict["KL = (MI + marginal KL)"] = (mi + marginal_kl).item()

        # Evaluation
        else:
            total_loss = torch.tensor(0.0)

        loss_dict["MI"] = mi.item() if mi is not None else None
        loss_dict["TC"] = tc.item() if tc is not None else None
        loss_dict["dim_KL"] = dim_kl.item() if dim_kl is not None else None
        loss_dict["marginal KL"] = marginal_kl.item() if marginal_kl is not None else None
        loss_dict["total_loss"] = total_loss
        loss_dict["elbo"] = elbo.item()
        loss_dict["kl_analytical"] = kl_analytical.item()
        loss_dict["fb_kl_analytical"] = fb_kl_analytical.item()
        loss_dict["reconstruction_loss"] = reconstruction_loss.item()

        return loss_dict


def get_scheduler(optimizer, warmup_updates=4000, lr_scheduler=True,
                  linear_lr_sched_grad_total_steps=5500,
                  lr_scheduler_type="vaswani", lr=2e-5):
    """
    Returns a learning rate scheduler with linear warm-up (if lr_scheduler).
    Otherwise it returns the unscaled learning rate.

    Args:
        optimizer: torch.optim
        warmup_updates: int:
            How many steps to warm-up (linearly) to the initial learning rate.
        lr_scheduler: bool:
            Whether or not to actually apply the scheduler.
    Returns:
        scheduler: LambdaLR scheduler
            This is the learning rate scheduler that may be updated during training by calling .step()
    """

    # Note that my implementation uses step = gradient step (or global step / accumulate_n_steps)
    def get_lr_scale(step):
        step = max(1.0, step)  # to avoid zero division

        # If using a scheduler, calculate the scaling coefficient
        if lr_scheduler:
            # Square root decay as described in Vaswani et al. (2017), section 5.3
            if lr_scheduler_type == "vaswani":
                # performs linear warm-up
                arg1 = 1 / np.sqrt(step)
                arg2 = step * (warmup_updates ** (-1.5))
                d_model = 768
                lr_scale = (1 / np.sqrt(d_model)) * np.min([arg1, arg2])

            # Else: linear warmup and decay
            else:
                if step < warmup_updates:
                    lr_scale = float(step / warmup_updates) * lr
                else:
                    ratio = max(0.0, float(linear_lr_sched_grad_total_steps - step) /
                                float(max(1, linear_lr_sched_grad_total_steps - warmup_updates)))
                    lr_scale = ratio * lr

        # Scale to the set learning rate (fixed)
        else:
            lr_scale = lr

        return lr_scale

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=get_lr_scale
    )

    return scheduler


# ----------------------------------------------------------------------------------------------------
# PARAMETER SCHEDULER
# ----------------------------------------------------------------------------------------------------

class ParameterScheduler:
    """
    This class implements a simple Parameter Scheduder, that follows:
      1. Cyclical: a cyclical schedule with:
         -> half cycle flat, quarter cycle increase/decrease, half cycle flat
      2. Linear: a linear schedule with linear warmup
      3. Constant: a constant schedule

    It's orientation may be decreasing or increasing.

    .step() needs to be called in the part where the gradient step increases
    .curren_value is a property that can be called where the loss is assembled
    """

    def __init__(self,
                 cyc_lin_con="cyclical",  # cyclical, linear, constant
                 n_cycles=5,
                 cycle_len=10,
                 ramp_len_grad_steps=10,
                 warmup_len_grad_steps=100,
                 decrease_increase="increase",
                 value=3.0,
                 name="beta"):

        assert cyc_lin_con in ["linear", "cyclical", "constant"], \
            "the 'cyc_lin' parameter must be set to 'linear', 'cyclical' or 'constant'"

        self.name = name
        self.cyc_lin_con = cyc_lin_con
        self.cycle_len = cycle_len
        self.n_cycles = n_cycles
        self.ramp_len_grad_steps = ramp_len_grad_steps
        self.decrease_increase = decrease_increase

        self.value = value

        self.cycle_i = 0
        self.grad_step_i = 0

        # Cyclical -> no warmup
        if self.cyc_lin_con == "cyclical":
            self.warmup_len_grad_steps = 0
        else:
            self.warmup_len_grad_steps = warmup_len_grad_steps

    @property
    def current_val(self):
        if self.cyc_lin_con == "constant":
            current_val = self.value

        elif self.cyc_lin_con == "linear":
            # Warm-up linear decrease / increase
            if self.grad_step_i < self.warmup_len_grad_steps:
                # For decrease, the warm-up should be increase
                if self.decrease_increase == "decrease":
                    if self.grad_step_i == 0:
                        current_val = 0.0
                    else:
                        current_val = (self.grad_step_i / self.warmup_len_grad_steps) * self.value
                # For increase, the warm-up should be decrease
                else:
                    current_val = ((self.warmup_len_grad_steps - self.grad_step_i)
                                   / self.warmup_len_grad_steps) * self.value

            # Linear decrease / increase
            else:
                # For decrease, the main part should be decrease
                if self.decrease_increase == "decrease":
                    current_val = ((self.ramp_len_grad_steps - (
                            self.grad_step_i - self.warmup_len_grad_steps)) / self.ramp_len_grad_steps) * self.value

                # For increase, the main part should be increase
                else:
                    if self.grad_step_i == 0:
                        current_val = 0.0
                    else:
                        current_val = ((self.grad_step_i - self.warmup_len_grad_steps)
                                       / self.ramp_len_grad_steps) * self.value
        # Cyclical
        else:
            half_cycle = int(self.cycle_len / 2)
            quarter_cycle = int(self.cycle_len / 4)
            where = self.grad_step_i - (self.cycle_i * self.cycle_len - 1)

            # Increase, half cycle at max, then increase for quarter cycle, then at max for quarter
            if self.decrease_increase == "increase":
                if where < half_cycle:
                    current_val = 0.0
                elif where > (half_cycle + quarter_cycle):
                    current_val = self.value
                else:
                    current_val = ((where - half_cycle) / quarter_cycle) * self.value

            # Decrease, half cycle at max, then decrease for quarter cycle, then at min for quarter
            else:
                if where < half_cycle:
                    current_val = self.value
                elif where > (half_cycle + quarter_cycle):
                    current_val = 0.0
                else:
                    current_val = ((quarter_cycle - (where - half_cycle)) / quarter_cycle) * self.value

        return current_val

    def step(self):
        self.grad_step_i += 1

        # Cyclical check if done with cycle and in total
        if self.cyc_lin_con == "cyclical":
            if self.grad_step_i % self.cycle_len == 0:
                self.cycle_i += 1

            # If done with all the cycles, reduces to a constant schedule afterwards
            if self.cycle_i == self.n_cycles:
                self.cyc_lin_con = "constant"
                if self.decrease_increase == "decrease":
                    self.value = 0.0

        # Linear check if done, reduces to a constant schedule afterwards
        elif self.cyc_lin_con == "linear":
            if self.grad_step_i > (self.warmup_len_grad_steps + self.ramp_len_grad_steps):
                self.cyc_lin_con = "constant"
                if self.decrease_increase == "decrease":
                    self.value = 0.0
