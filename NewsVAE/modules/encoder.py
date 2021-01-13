import torch.nn as nn
from transformers import RobertaModel
import torch
import math


class EncoderNewsVAE(torch.nn.Module):
    def __init__(self, gradient_checkpointing=False, latent_size=768):
        """
        Encoder of VAE based on a roberta-base checkpoint with a custom pooler.
        """
        super(EncoderNewsVAE, self).__init__()

        self.model = RobertaModel.from_pretrained("roberta-base",
                                                  add_pooling_layer=False,
                                                  return_dict=True,
                                                  gradient_checkpointing=gradient_checkpointing)

        # Add a fresh pooling layer
        self.model.pooler = PoolerEncoderNewsVAE(hidden_size=self.model.config.hidden_size,
                                                 latent_size=latent_size)
        self.model.pooler.dense.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)

    def forward(self, input_ids, attention_mask, return_embeddings=False):
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
        # Encode
        encoder_outs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=return_embeddings)

        # Get pooled features
        pooled_output = encoder_outs.pooler_output

        # Interpret as mean, log variance
        mu, logvar = pooled_output.chunk(2, dim=1)

        if return_embeddings:
            return mu, logvar, encoder_outs["hidden_states"][0]

        return mu, logvar

    def encode(self, input_ids, attention_mask, n_samples=1, hinge_kl_loss_lambda=0.5):
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

        # Forward
        mu, logvar = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        # Sample latents from posterior
        latent_z = self.reparameterize(mu, logvar, n_samples=n_samples)

        latent_z = latent_z.squeeze(1)

        # Calculate the KL divergence
        kl_loss, hinge_kl_loss = self.kl_divergence(mu, logvar, hinge_kl_loss_lambda=hinge_kl_loss_lambda)

        # Calculate the Maximum Mean Discrepancy
        mmd_loss = self.maximum_mean_discrepancy(latent_z)

        return mu, logvar, latent_z, kl_loss, hinge_kl_loss, mmd_loss

    # # TODO: NOT TESTED
    # def log_q_z_x(self, latent_z, input_ids=None, attention_mask=None, mu=None, logvar=None):
    #     """
    #     This function computes log q(z|x) for a set of latents z sampled from
    #     posterior that is defined by the encoded input or by the parameters.
    #
    #     Args:
    #         z: Tensor [batch, n_samples, latent_size]
    #             Different latents that will be evaluated under the posterior q
    #
    #     Returns:
    #         log_density: Tensor [batch, n_samples]
    #             Log density, log q(z|x), for all samples
    #     """
    #
    #     batch, n_samples, latent_size = latent_z.shape
    #
    #     # Either provide input or parameters of the distribution q_z_x
    #     params_present = mu is not None and logvar is not None
    #     inputs_present = input_ids is not None and attention_mask is not None
    #
    #     assert params_present or inputs_present, "Either provide inputs or provide parameters of the distributions"
    #
    #     # Do forward if only inputs are given
    #     if not params_present:
    #         mu, logvar = self.model(input_ids=input_ids,
    #                                 attention_mask=attention_mask)
    #
    #     # (batch_size, 1, nz)
    #     mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
    #     var = logvar.exp()
    #
    #     # (batch_size, nsamples, nz)
    #     dev = latent_z - mu
    #
    #     # (batch_size, nsamples)
    #     log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - 0.5 * (
    #                 latent_size * math.log(2 * math.pi) + logvar.sum(-1))
    #
    #     return log_density

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
    def kl_divergence(mu, logvar, hinge_kl_loss_lambda=0.5):
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

        # Ignore the dimensions of which the KL-div is already under the
        # threshold, avoiding driving it down even further. Those values do
        # not have to be replaced by the threshold because that would not mean
        # anything to the gradient. That's why they are simply removed. This
        # confused me at first.
        kl_mask = (kl_loss > hinge_kl_loss_lambda).float()

        # Sum over the latent dimensions and average over the batch dimension
        hinge_kl_loss = (kl_mask * kl_loss).sum(dim=1).mean(dim=0)
        kl_loss = kl_loss.sum(dim=1).mean(dim=0)

        return kl_loss, hinge_kl_loss

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
        return torch.exp(-kernel_input)  # TODO: check this, gives type error

    def maximum_mean_discrepancy(self, latent_z):
        """
        Taken from: https://github.com/aktersnurra/information-maximizing-variational-autoencoders/blob/master/model/loss_functions/mmd_loss.py

        :param latent_z:
        :return:
        """
        x = torch.randn_like(latent_z).to(latent_z.device)
        y = latent_z
        x_kernel = self.gaussian_kernel(x, x)
        y_kernel = self.gaussian_kernel(y, y)
        xy_kernel = self.gaussian_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

        return mmd


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
