import torch.nn as nn
from transformers import RobertaModel
import torch
import math
from modules.encoder_roberta import VAE_Encoder_RobertaModel


class EncoderNewsVAE(torch.nn.Module):
    def __init__(self, gradient_checkpointing=False, latent_size=768):
        """
        Encoder of VAE based on a roberta-base checkpoint with a custom pooler.
        """
        super(EncoderNewsVAE, self).__init__()

        self.model = VAE_Encoder_RobertaModel.from_pretrained("roberta-base",
                                                              add_pooling_layer=False,
                                                              return_dict=True,
                                                              gradient_checkpointing=gradient_checkpointing)

        # Add a fresh pooling layer
        self.model.pooler = PoolerEncoderNewsVAE(hidden_size=self.model.config.hidden_size,
                                                 latent_size=latent_size)
        self.model.pooler.dense.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)

    def forward(self, input_ids, attention_mask, return_embeddings=True):
        """
        Encode input sequences and sample from the approximate posterior.

        Args:
            input_ids: Tensor [batch, seq_len]
                The input sequence token ids
            attention_mask: Tensor [batch, seq_len]
                The input sequence mask, masking padded tokens with 0
            n_samples: int
                The number of samples per encoded posterior of sample to return

        Returns:
            mu: Tensor [batch, latent_size]
                The mean of the posterior of the input samples
            logvar: Tensor [batch, latent_size]
                The log variance of the posterior of the input samples
            latent_z: Tensor [batch, n_samples, latent_size]
                Samples from encoded posterior:
        """
        # Encoder
        encoder_outs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=return_embeddings)

        # Get pooled features
        pooled_output = encoder_outs.pooler_output

        # Interpret as mean, log variance
        mu, logvar = pooled_output.chunk(2, dim=1)

        if return_embeddings:
            word_embeddings = encoder_outs["hidden_states"][0]
        else:
            word_embeddings = None

        return_dict = {
            "mu": mu,
            "logvar": logvar,
            "word_embeddings": word_embeddings
        }

        return return_dict

    def encode(self, input_ids, attention_mask, n_samples=1, hinge_kl_loss_lambda=0.5,
               return_log_q_z_x=False, return_log_p_z=False, return_embeddings=False):
        """
        This function encodes samples into latents by sampling and returns losses (kl, hinge_kl & mmd).

        Args:
            input_ids: Tensor [batch, seq_len]
                The input sequence token ids
            attention_mask: Tensor [batch, seq_len]
                The input sequence mask, masking padded tokens with 0
            n_samples: int
                The number of samples per encoded posterior of sample to return
            hinge_kl_loss_lambda: float
                At what value to cap the KL-divergence, ie. KL will not be lower
                than this value (not reaching 0).
        Returns:
            mu: Tensor [batch, latent_size]
                The mean of the posterior of the input samples
            logvar: Tensor [batch, latent_size]
                The log variance of the posterior of the input samples
            latent_z: Tensor [batch, n_samples, latent_size]
                Samples from encoded posterior
            kl_loss: float: KL-divergence
            hinge_kl_loss: float: clamped KL-divergence
            mmd_loss: float: MMD-loss
        """

        print(input_ids.device, attention_mask.device)

        # Forward
        encoder_out = self.forward(input_ids=input_ids, attention_mask=attention_mask,
                                   return_embeddings=return_embeddings)

        mu, logvar = encoder_out["mu"], encoder_out["logvar"]

        # Sample latents from posterior
        latent_z = self.reparameterize(encoder_out["mu"], encoder_out["logvar"], n_samples=n_samples)

        if n_samples == 1:
            latent_z = latent_z.squeeze(1)
            logvar_, mu_ = logvar, mu
        else:
            # repeat along the sample dimension (1)
            logvar_, mu_ = logvar.unsqueeze(1).repeat(1, n_samples, 1), mu.unsqueeze(1).repeat(1, n_samples, 1)

        if return_log_q_z_x:
            log_q_z_x = self.sample_log_likelihood(latent_z, mu=mu_, logvar=logvar_,
                                                   reduce_latent_dim=True, reduce_batch_dim=False)
        else:
            log_q_z_x = None

        if return_log_p_z:
            log_p_z = self.sample_log_likelihood(latent_z, mu=None, logvar=None,
                                                 reduce_latent_dim=True, reduce_batch_dim=False)
        else:
            log_p_z = None

        # Calculate the KL divergence
        kl_loss, hinge_kl_loss = self.kl_divergence(mu, logvar, hinge_kl_loss_lambda=hinge_kl_loss_lambda)

        # Calculate the Maximum Mean Discrepancy
        if n_samples != 1:
            # print("Warning, MMD loss not implemented for multiple samples.")
            mmd_loss = None
        else:
            mmd_loss = self.maximum_mean_discrepancy(latent_z)

        return_dict = {
            "mu": mu,
            "logvar": logvar,
            "latent_z": latent_z,
            "kl_loss": kl_loss,
            "hinge_kl_loss": hinge_kl_loss,
            "mmd_loss": mmd_loss,
            "log_q_z_x": log_q_z_x,
            "log_p_z": log_p_z,
            "word_embeddings": encoder_out["word_embeddings"]
        }

        return return_dict

    @staticmethod
    def sample_log_likelihood(latent_z, mu=None, logvar=None, reduce_latent_dim=True, reduce_batch_dim=False):
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

    @staticmethod
    def approximate_marginal_KL(mu, logvar, latent_z, method="chen", dataset_size=None):

        log_q_z = EncoderNewsVAE.approximate_log_q_z(mu, logvar, latent_z, method=method,
                                                     dataset_size=dataset_size, prod_marginals=False)

        log_p_z = EncoderNewsVAE.sample_log_likelihood(latent_z, reduce_latent_dim=True, reduce_batch_dim=False)

        marginal_KL = (log_q_z - log_p_z).mean()

        return marginal_KL

    @staticmethod
    def approximate_total_correlation(mu, logvar, latent_z, method="chen", dataset_size=None):
        # From Chen et al. (2019), Isolating Sources of Disentanglement
        # KL(q(z) || prod q(z_i)) <- mutual information, or dependence, between the latent dimensions

        # log q(z)
        log_q_z = EncoderNewsVAE.approximate_log_q_z(mu, logvar, latent_z, method=method,
                                                     dataset_size=dataset_size, prod_marginals=False)
        # log prod q(z_i)
        log_q_z_prod_marginals = EncoderNewsVAE.approximate_log_q_z(mu, logvar, latent_z, method=method,
                                                                    dataset_size=dataset_size, prod_marginals=True)

        total_correlation = (log_q_z - log_q_z_prod_marginals).mean()

        return total_correlation

    @staticmethod
    def approximate_log_q_z(mu, logvar, latent_z, method="chen", dataset_size=None, prod_marginals=False):
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

        # Get shapes
        x_batch, n_dim = mu.shape

        # Orient it as a row [1, x_batch, latent_dim]
        # mu_1, mu_2, ..., mu_n
        mu_exp, logvar_exp = mu.unsqueeze(0), logvar.unsqueeze(0)

        # Orient it as a column [z_batch, 1, latent_dim]
        latent_z_exp = latent_z.unsqueeze(1)

        # Evaluate the log probability q(z_i|mu_j, sigma_j) for all z_i and all (mu_j, sigma_j)
        # [z_batch, x_batch, latent_dim]
        log_dens = EncoderNewsVAE.sample_log_likelihood(latent_z_exp, mu_exp, logvar_exp,
                                                        reduce_latent_dim=False, reduce_batch_dim=False)

        log_q_z = None
        log_q_z_prod_marg = None

        # Log prod q(z_i) => Sum log q(z_j)
        if prod_marginals:
            # Reduce x_batch dim, then latent_dim (happens later in the code at **)
            log_q_z_prod_marg = torch.logsumexp(log_dens, dim=1, keepdim=False)
        else:
            # Reduce latent and then x_batch dim
            log_q_z = torch.logsumexp(log_dens.sum(2), dim=1, keepdim=False)

        # We assume to have been given an batch, and use a weighted version as proposed in
        # Isolating Sources of Disentanglement (Chen et al., 2019)
        if method == "chen":
            if prod_marginals:
                # ** sum reduce the latent_dim
                log_q_z_prod_marg = (log_q_z_prod_marg - math.log(x_batch * dataset_size)).sum(dim=1)
            else:
                log_q_z -= math.log(x_batch * dataset_size)

        # Assuming the data we got was the whole dataset
        else:
            if prod_marginals:
                # ** sum reduce the latent_dim
                log_q_z_prod_marg = (log_q_z_prod_marg - math.log(x_batch)).sum(dim=1)
            else:
                log_q_z -= math.log(x_batch)

        if prod_marginals:
            return log_q_z_prod_marg
        else:
            return log_q_z

    @staticmethod
    def reparameterize(mu, logvar, n_samples=1):
        """
        Sample from posterior Gaussian family

        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                Log variance of gaussian distibution with shape (batch, nz)
                Log variance to ensure positivity
            n_samples: int
                How many samples z per encoded sample x to return
        Returns:
            latent_z: Tensor
                Sampled z with shape (batch, n_samples, nz)
        """

        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()  # logvar -> std

        # Expand for multiple samples per parameter pair
        mu_expd = mu.unsqueeze(1).expand(batch_size, n_samples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, n_samples, nz)

        # Reparameterize: transform basic noise sample into complex sample
        eps = torch.zeros_like(std_expd).normal_()
        latent_z = mu_expd + torch.mul(eps, std_expd)

        return latent_z

    @staticmethod
    def sample_from_prior(latent_size=768, n_samples=8, device_name="cuda:0"):
        """
        Sampels from prior distribution (factorised standard normal).

        Args:
            latent_size: int
            n_samples: int
            device_name: str

        Returns:
            samples: Tensor [batch, latent_size]
        """
        loc = torch.zeros(latent_size, device=device_name)
        scale = torch.ones(latent_size, device=device_name)
        prior_dist = torch.distributions.normal.Normal(loc, scale)
        samples = prior_dist.sample((n_samples,))

        return samples

    @staticmethod
    def kl_divergence(mu, logvar, hinge_kl_loss_lambda=0.5, average_batch=True):
        """
        Calculates the KL-divergence between the posterior and the prior.

        Arguments:
            mu: Tensor [batch, latent_size]
                Mean of the posteriors that resulted from encoding.
            logvar: Tensor [batch, latent_size]
                Log variance of the posteriors that resulted from encoding.
            hing_kl_loss_lambda: float
                At what value to cap the KL-divergence, ie. KL will not be lower
                than this value (not reaching 0).
        Returns:
            kl_loss: Tensor [batch]
                The KL-divergence for all samples in the batch.
            hing_kl_loss: Tensor [batch]
                The KL-divergence for all samples greater than <hinge_kl_loss_lambda>
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
            hinge_kl_loss = (kl_mask * kl_loss).sum(dim=1)

        else:
            hinge_kl_loss = kl_loss.sum(dim=1)

        kl_loss = kl_loss.sum(dim=1)

        if average_batch:
            hinge_kl_loss = hinge_kl_loss.mean(dim=0)
            kl_loss = kl_loss.mean(dim=0)

        return kl_loss, hinge_kl_loss

    @staticmethod
    def maximum_mean_discrepancy(latent_z):
        """
        Taken from: https://github.com/aktersnurra/information-maximizing-variational-autoencoders/blob/master/model/loss_functions/mmd_loss.py

        :param latent_z:
        :return:
        """
        x = torch.randn_like(latent_z).to(latent_z.device)
        y = latent_z
        x_kernel = EncoderNewsVAE.gaussian_kernel(x, x)
        y_kernel = EncoderNewsVAE.gaussian_kernel(y, y)
        xy_kernel = EncoderNewsVAE.gaussian_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

        return mmd

    @staticmethod
    def gaussian_kernel(x, y):
        """
        Gaussian kernel
        Taken from: https://github.com/aktersnurra/information-maximizing-variational-autoencoders/blob/master/model/loss_functions/mmd_loss.py

        :param x:
        :param y:
        :return:
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


class PoolerEncoderNewsVAE(nn.Module):
    def __init__(self, hidden_size=768, latent_size=768):
        """
        This module acts like a pooler in the EncoderNewsVAE, aggregating the info into one vector.
        """

        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, latent_size * 2)

    def init_weights(self, initializer_range):
        self.dense.weight.data.normal_(mean=0.0, std=initializer_range)

    def forward(self, hidden_states):
        """
        The forward pass concats the last hidden state of the first and last token
        and takes them through a dense layer.

        :param hidden_states:
        :return:
        """

        first_token_tensor = hidden_states[:, 0]
        last_token_tensor = hidden_states[:, -1]
        first_last = torch.cat((first_token_tensor, last_token_tensor), 1)
        pooled_output = self.dense(first_last)
        return pooled_output
