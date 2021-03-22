import torch.nn as nn
import torch
import math
from modules.encoder_roberta import VAE_Encoder_RobertaModel
from loss_and_optimisation import *


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

    def encode(self, input_ids, attention_mask, n_samples=1, dataset_size=10000,
               return_log_q_z_x=True, return_log_p_z=True, return_log_q_z=True, return_embeddings=False):
        """
        This function encodes samples into latents by sampling and returns losses (kl, hinge_kl & mmd).
        """

        # Forward
        encoder_out = self.forward(input_ids=input_ids, attention_mask=attention_mask,
                                   return_embeddings=return_embeddings)

        mu, logvar = encoder_out["mu"], encoder_out["logvar"]

        # Sample latents from posterior
        latent_z = self.reparameterize(encoder_out["mu"], encoder_out["logvar"], n_samples=n_samples)

        print("n_samples", n_samples)

        if n_samples == 1:
            latent_z = latent_z.squeeze(1)
            logvar_, mu_ = logvar, mu
        else:
            # repeat along the sample dimension (1)
            logvar_, mu_ = logvar.unsqueeze(1).repeat(1, n_samples, 1), mu.unsqueeze(1).repeat(1, n_samples, 1)

        print("in encode before return_log_q_z_x latent_z.shape", latent_z.shape)
        if return_log_q_z_x:
            log_q_z_x = sample_log_likelihood(latent_z, mu=mu_, logvar=logvar_,
                                              reduce_latent_dim=True, reduce_batch_dim=True)
        else:
            log_q_z_x = None

        print("in encode before return_log_p_z latent_z.shape", latent_z.shape)
        if return_log_p_z:
            log_p_z = sample_log_likelihood(latent_z, mu=None, logvar=None,
                                            reduce_latent_dim=True, reduce_batch_dim=True)
        else:
            log_p_z = None

        if return_log_q_z:
            log_q_z, log_q_z_prod_marg = approximate_log_q_z(mu, logvar, latent_z, method="chen",
                                                             dataset_size=dataset_size,
                                                             prod_marginals=True)
        else:
            log_q_z, log_q_z_prod_marg = None, None

        # Calculate the Maximum Mean Discrepancy
        if n_samples != 1:
            # print("Warning, MMD loss not implemented for multiple samples.")
            mmd_loss = None
        else:
            mmd_loss = maximum_mean_discrepancy(latent_z)

        return_dict = {
            "mu": mu,
            "logvar": logvar,
            "latent_z": latent_z,
            # "kl_loss": kl_loss,
            # "hinge_kl_loss": hinge_kl_loss,
            "mmd_loss": mmd_loss,
            "log_q_z": log_q_z,
            "log_q_z_prod_marg": log_q_z_prod_marg,
            "log_q_z_x": log_q_z_x,
            "log_p_z": log_p_z
        }

        return return_dict

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
