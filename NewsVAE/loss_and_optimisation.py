from constraintoptim.constraint import Constraint, ConstraintOptimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler
import torch
import numpy as np
import math


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
    # From Chen et al. (2019), Isolating Sources of Disentanglement
    # KL(q(z) || prod q(z_i)) <- mutual information, or dependence, between the latent dimensions

    # log q(z), log prod q(z_i)
    log_q_z, log_q_z_prod_marginals = approximate_log_q_z(mu, logvar, latent_z, method=method,
                                                          dataset_size=dataset_size,
                                                          prod_marginals=True)

    total_correlation = (log_q_z - log_q_z_prod_marginals).mean()

    return total_correlation


def approximate_log_q_z(mu, logvar, latent_z, method="chen", dataset_size=42068, prod_marginals=False):
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

    #print("dataset size", dataset_size)

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

    return log_q_z.mean(), log_q_z_prod_marg.mean()


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
        self.total_loss_optimiser = torch.optim.AdamW(vae_model.parameters(), lr=config.lr)
        self.total_loss_scheduler = get_scheduler(self.total_loss_optimiser,
                                                  warmup_updates=config.lr_warmup_updates,
                                                  lr_scheduler=config.lr_scheduler,
                                                  linear_lr_sched_grad_total_steps=config.linear_lr_sched_grad_total_steps,
                                                  lr_scheduler_type=config.lr_scheduler_type,
                                                  lr=config.lr)

        self.manager = dict()
        self.free_bits_pd = config.fb_b_vae_free_bits_pd
        # Automatic Mixed Precision scaler
        self.scaler = GradScaler(enabled=config.use_amp)

        if self.objective == "beta-vae":
            # beta * KL term
            if config.b_vae_beta_constant_linear_lagrangian in ["constant", "linear"]:
                print(f"Setting parameter schedule for beta-vae KL term, schedule: "
                      f"{config.b_vae_beta_constant_linear_lagrangian}, ramp length: "
                      f"{config.b_vae_beta_ramp_len_grad_steps}, beta value: {config.b_vae_beta}")
                self.manager["beta_KL"] = ParameterScheduler(
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

    def forward(self, input_ids, attention_mask, return_exact_match=False, return_reconstruction_loss=True,
                return_posterior_stats=True, device_name="cuda:0", return_cross_entropy=False,  reduce_seq_dim_ce="mean",
                reduce_batch_dim_ce="mean", reduce_seq_dim_exact_match="mean", reduce_batch_dim_exact_match="mean"):

        vae_out = self.vae_model(input_ids=input_ids, attention_mask=attention_mask,
                                 return_exact_match=return_exact_match,
                                 reduce_seq_dim_exact_match=reduce_seq_dim_exact_match,
                                 reduce_batch_dim_exact_match=reduce_batch_dim_exact_match,
                                 return_mu_logvar=True,
                                 return_latents=True,
                                 return_reconstruction_loss=return_reconstruction_loss,
                                 return_posterior_stats=return_posterior_stats,
                                 return_cross_entropy=return_cross_entropy,
                                 reduce_seq_dim_ce=reduce_seq_dim_ce,
                                 reduce_batch_dim_ce=reduce_batch_dim_ce,
                                 device_name=device_name)

        loss_dict = self.assemble_loss(vae_out["reconstruction_loss"], vae_out["mu"], vae_out["logvar"],
                                       log_p_z=vae_out["log_p_z"],
                                       log_q_z_x=vae_out["log_q_z_x"],
                                       log_q_z=vae_out["log_q_z"], log_q_z_prod_marg=vae_out["log_q_z_prod_marg"],
                                       mmd=None)

        return loss_dict

    def assemble_loss(self, reconstruction_loss, mu, logvar,
                      log_p_z=None, log_q_z_x=None,
                      log_q_z=None, log_q_z_prod_marg=None, mmd=None):

        # Calculate the KL divergence analytically
        kl_analytical, fb_kl_analytical = kl_divergence(mu, logvar,
                                                        hinge_kl_loss_lambda=self.free_bits_pd,
                                                        average_batch=True, sum_latent=True)

        # Non-analytical KL computation
        # kl_loss = log_q_z_x - log_p_z
        elbo = - (reconstruction_loss + kl_analytical)

        loss_dict = dict()

        # Autoencoder
        if self.objective == "autoencoder":
            total_loss = reconstruction_loss

        # VAE
        elif self.objective == "vae":
            total_loss = -elbo

        # (Free bits) Beta-VAE
        elif self.objective == "beta-vae" or self.objective == "free-bits-beta-vae":

            if self.objective == "free-bits-beta-vae":
                kl_loss = fb_kl_analytical
            else:
                kl_loss = kl_analytical

            # Parameter schedule
            if isinstance(self.manager["beta_KL"], ParameterScheduler):
                beta = self.manager["beta_KL"].current_val
                beta_kl = beta * kl_loss
            # Lagrangian
            else:
                beta = self.manager["beta_KL"]["constraint"].multiplier
                beta_kl = self.manager["beta_KL"]["constraint"](kl_loss)

            total_loss = reconstruction_loss + beta_kl
            loss_dict["beta"] = beta.item() if torch.is_tensor(beta) else beta
            loss_dict["beta_KL"] = beta_kl.item() if torch.is_tensor(beta_kl) else beta_kl
            loss_dict["KL"] = kl_loss.item()

        # Beta-TC-VAE
        elif self.objective == "beta-tc-vae":
            mi = log_q_z_x - log_q_z
            # Parameter schedule
            if isinstance(self.manager["alpha_MI"], ParameterScheduler):
                alpha = self.manager["alpha_MI"].current_val
                alpha_mi = alpha * mi
            # Lagrangian
            else:
                alpha = self.manager["alpha_MI"]["constraint"].multiplier

                alpha_mi = self.manager["alpha_MI"]["constraint"](mi)

            # Parameter schedule
            tc = log_q_z - log_q_z_prod_marg
            beta = self.manager["beta_TC"].current_val
            beta_tc = beta * tc

            dim_kl = log_q_z_prod_marg - log_p_z
            # Parameter schedule (linear or constant)
            if isinstance(self.manager["gamma_DimKL"], ParameterScheduler):
                gamma = self.manager["gamma_DimKL"].current_val
                gamma_dim_kl = gamma * dim_kl
            # Lagrangian
            else:
                gamma = self.manager["gamma_DimKL"]["constraint"].multiplier
                gamma_dim_kl = self.manager["gamma_DimKL"]["constraint"](dim_kl)

            marginal_kl = log_q_z - log_p_z
            approx_kl = log_q_z_x - log_p_z

            total_loss = reconstruction_loss + alpha_mi + beta_tc + gamma_dim_kl
            loss_dict["MI"] = mi.item()
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

        # Evaluation
        else:
            total_loss = torch.tensor(0.0)

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
      1. Cyclical: a (linear) cyclical schedule with a saw tooth behaviour
      2. Linear: a linear schedule with linear warmup
      3. Constant: a constant schedule

    It's orientation may be decreasing or increasing.

    .step() needs to be called in the part where the gradient step increases
    .curren_value is a property that can be called where the loss is assembled
    """

    def __init__(self,
                 cyc_lin_con="cyclical",  # cyclical, linear, constant
                 n_cycles=5,
                 ramp_len_grad_steps=10,
                 warmup_len_grad_steps=100,
                 decrease_increase="increase",
                 value=3.0,
                 name="beta"):

        assert cyc_lin_con in ["linear", "cyclical", "constant"], \
            "the 'cyc_lin' parameter must be set to 'linear', 'cyclical' or 'constant'"

        self.name = name
        self.cyc_lin_con = cyc_lin_con
        self.ramp_len_grad_steps = ramp_len_grad_steps
        self.decrease_increase = decrease_increase

        self.value = value

        self.cycle_i = 0
        self.grad_step_i = 0

        # Linear is a special case of cyclical with 1 cycle
        if self.cyc_lin_con == "linear":
            self.n_cycles = 1
        else:
            self.n_cycles = n_cycles

        # Cyclical is linear with 0 warmup
        if self.cyc_lin_con == "cyclical":
            self.warmup_len_grad_steps = 0
        else:
            self.warmup_len_grad_steps = warmup_len_grad_steps

    @property
    def current_val(self):
        if self.cyc_lin_con == "constant":
            current_val = self.value

        else:
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

        return current_val

    def step(self):
        self.grad_step_i += 1

        # Cyclical check if done with cycle and in total
        if self.cyc_lin_con == "cyclical":
            if self.grad_step_i > self.ramp_len_grad_steps:
                self.grad_step_i = 0
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
