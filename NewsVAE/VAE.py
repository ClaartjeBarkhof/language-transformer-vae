import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AutoModelForCausalLM, AutoModel
from utils_external import tie_weights
import torch
import argparse
from VAE_Decoder_Roberta import VAE_Decoder_RobertaForCausalLM, VAE_Decoder_RobertaPooler
import copy
from NewsVAEArguments import preprare_parser
import utils
import math


class NewsVAE(torch.nn.Module):
    def __init__(self, encoder, decoder,
                 latent_size=768,
                 add_latent_via_memory=True,
                 add_latent_via_embeddings=True,
                 do_tie_weights=True):
        super(NewsVAE, self).__init__()

        # Essentials
        self.encoder = encoder
        self.decoder = decoder

        # Some parameters
        self.latent_size = latent_size
        self.n_layers = self.encoder.config.num_hidden_layers
        self.hidden_size = self.encoder.hidden_size
        self.initializer_range = self.encoder.config.initializer_range

        # To connect the encoder and the decoder through the latent space
        self.latent_to_decoder = LatentToDecoderNewsVAE(add_latent_via_memory=add_latent_via_memory,
                                                        add_latent_via_embeddings=add_latent_via_embeddings,
                                                        latent_size=self.latent_size, hidden_size=self.hidden_size,
                                                        n_layers=self.n_layers,
                                                        initializer_range=self.initializer_range)

        # Weight tying / sharing
        if do_tie_weights:
            base_model_prefix = self.decoder.base_model_prefix
            tie_weights(self.encoder, self.decoder._modules[base_model_prefix], base_model_prefix)

    def forward(self, input_ids, attention_mask, beta,
                return_predictions=False,
                return_attention_probs=False,
                return_exact_match_acc=True, objective='beta-vae'):
        """
        Perform a forward pass through the whole VAE with the sampling operation in between.

        Args:
            input_ids: Tensor [batch, seq_len]
                The input sequence token ids
            beta: float
                What weight to give to the KL-term in the loss for beta-vae objective
            attention_mask: Tensor [batch, seq_len]
                The input sequence mask, masking padded tokens with 0
            objective: str ["beta-vae" or "mmd-vae"]
                According to what objective to calculate the full loss (to perform backward pass on).
            return_predictions: bool
                Whether or not to return predictions and logits in the losses dict
            return_exact_match_acc: bool
                Whether or not to return exact match accuracy in the losses dict
        Returns:
            losses: Dict[str, Union[float, Tensor]
                The result dictionary of the full forward pass with metrics
                and possibly predictions.
        """

        # Forward through encoder and sample
        mu, logvar, latent_z, kl_loss, hinge_kl_loss, mmd_loss = self.encoder.encode(input_ids=input_ids,
                                                                                     attention_mask=attention_mask,
                                                                                     n_samples=1)
        latent_to_decoder_output = self.latent_to_decoder(latent_z)

        decoder_outs = self.decoder(latent_to_decoder_output=latent_to_decoder_output,
                                    input_ids=input_ids, attention_mask=attention_mask,
                                    output_attentions=return_attention_probs,
                                    return_predictions=return_predictions,
                                    return_exact_match_acc=return_exact_match_acc)

        total_loss = None

        if objective == 'beta-vae':
            total_loss = decoder_outs["cross_entropy"] + (beta * hinge_kl_loss)

        elif objective == 'mmd-vae':
            total_loss = decoder_outs["cross_entropy"] + mmd_loss

        else:
            print("Not supported objective. Set valid option: beta-vae or mmd-vae.")

        # Detach all except the total loss on which we need to base our backward pass
        losses = {'kl_loss': kl_loss.item(), 'hinge_kl_loss': hinge_kl_loss.item(),
                  'recon_loss': decoder_outs["cross_entropy"].item(), 'total_loss': total_loss,
                  'mmd_loss': mmd_loss.item()}

        if return_predictions:
            losses['logits'] = decoder_outs["logits"]
            losses["predictions"] = decoder_outs["predictions"]

        if return_attention_probs:
            losses['attention_probs'] = decoder_outs["decoder_outs"]

        if return_exact_match_acc:
            losses["exact_match_acc"] = decoder_outs["exact_match_acc"].item()

        return losses


class PoolerEncoderNewsVAE(nn.Module):
    def __init__(self, config, latent_size):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, latent_size * 2)

    def init_weights(self, initializer_range):
        self.dense.weight.data.normal_(mean=0.0, std=initializer_range)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        last_token_tensor = hidden_states[:, -1]
        first_last = torch.cat((first_token_tensor, last_token_tensor), 1)
        pooled_output = self.dense(first_last)
        return pooled_output


class EncoderNewsVAE(torch.nn.Module):
    def __init__(self, gradient_checkpointing=False, base_checkpoint="roberta-base", latent_size=768):
        super(EncoderNewsVAE, self).__init__()

        self.encoder = RobertaModel.from_pretrained(base_checkpoint, add_pooling_layer=False,
                                                    return_dict=True,
                                                    gradient_checkpointing=gradient_checkpointing)

        # Add a fresh pooling layer
        self.encoder.pooler = PoolerEncoderNewsVAE(self.encoder.config.hidden_size, latent_size)
        self.encoder.pooler.dense.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)

    def forward(self, input_ids, attention_mask):
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
        encoder_outs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Get pooled features
        pooled_output = encoder_outs.pooler_output

        # Interpret as mean, log variance
        mu, logvar = pooled_output.chunk(2, dim=1)

        return mu, logvar

    def encode(self, input_ids, attention_mask, n_samples=1, hinge_kl_loss_lambda=0.5):
        mu, logvar = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Sample latents from posterior
        latent_z = self.reparameterize(mu, logvar, n_samples=n_samples)

        # Calculate the KL divergence
        kl_loss, hinge_kl_loss = self.kl_divergence(mu, logvar, hinge_kl_loss_lambda=hinge_kl_loss_lambda)

        # Calculate the Maximum Mean Discrepancy
        mmd_loss = self.maximum_mean_discrepancy(latent_z)

        return mu, logvar, latent_z, kl_loss, hinge_kl_loss, mmd_loss

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
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, n_samples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, n_samples, nz)

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

        # Sum over the latent dimensions (and average over the batch dimension
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
        return torch.exp(-kernel_input)

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


class LatentToDecoderNewsVAE(nn.Module):
    def __init__(self, add_latent_via_memory=True,
                 add_latent_via_embeddings=True,
                 latent_size=768, hidden_size=768, n_layers=12,
                 initializer_range=0.02):
        super(LatentToDecoderNewsVAE, self).__init__()

        self.add_latent_via_memory = add_latent_via_memory
        self.add_latent_via_embeddings = add_latent_via_embeddings

        # Latent via memory layer
        if self.add_latent_via_memory:
            self.latent_to_memory_projection = nn.Linear(latent_size, hidden_size * n_layers)
            self.latent_to_memory_projection.weight.data.normal_(mean=0.0, std=initializer_range)

        # Latent via embedding layer
        if self.add_latent_via_embeddings:
            self.latent_to_embedding_projection = nn.Linear(latent_size, self.hidden_size)
            self.latent_to_embedding_projection.weight.data.normal_(mean=0.0, std=initializer_range)

    def forward(self, latent_z):
        """
        Handles the connection between encoder and decoder by transforming
        the latent in such a way the decoder can use it.

        Args:
            latent_z: Tensor [batch, latent_size]
                The latents sampled from the encoded input posterior.
        Returns:
            output: Dict[str, Tensor]
                Depending on whether or not to add via memory and/or embeddings
                it returns a dict containing the right information to be used by decoder.
        """

        output = {"latent_to_memory":       None,
                  "latent_to_embeddings":   None}

        if self.add_latent_via_memory:
            latent_to_memory = self.latent_to_memory_projection(latent_z)
            # Makes tuple of equally sized tensors of (batch x 1 x hidden_size)
            latent_to_memory = torch.split(latent_to_memory.unsqueeze(1), self.hidden_size, dim=2)
            output["latent_to_memory"] = latent_to_memory

        if self.add_latent_via_embeddings:
            latent_to_embeddings = self.latent_to_embedding_projection(latent_z)
            output["latent_to_embeddings"] = latent_to_embeddings

        return output


class DecoderNewsVAE(torch.nn.Module):
    def __init__(self, base_checkpoint="roberta-base", gradient_checkpointing=False):
        super(DecoderNewsVAE, self).__init__()

        self.decoder = VAE_Decoder_RobertaForCausalLM.from_pretrained(base_checkpoint,
                                                                      gradient_checkpointing=gradient_checkpointing)

    def forward(self, latent_to_decoder_output, input_ids, attention_mask,
                return_predictions=False, return_exact_match_acc=True):
        """
        Make a forward pass through the decoder.

        Args:
            latent_to_decoder_output:
                Latents information transformed to a format the decoder can use
                via memory & embeddings mechanism
            input_ids: Tensor [batch x seq_len]
                Token input ids of the output so far or teacher-forced the whole seq.
            attention_mask: Tensor [batch x seq_len]
                Mask marking the padded tokens (0)
            return_predictions: bool
                Whether or not to return predictions
            return_exact_match_acc: bool
                Whether or not to return exact match accuracy

        Returns:
            decoder_outs: Dict[str, Union[Tensor, float]]
                Everything the decoder returns (predictions, reconstruction loss, etc.)
        """

        # Forward the decoder
        decoder_outs = self.decoder(input_ids=input_ids, attention_mask=attention_mask,
                                    latent_to_decoder_output=latent_to_decoder_output, labels=copy.copy(input_ids),
                                    return_cross_entropy=True,
                                    return_predictions=return_predictions,
                                    return_exact_match_acc=return_exact_match_acc)

        return decoder_outs

    def reset_to_base_checkpoint(self, checkpoint_name="roberta-base", gradient_checkpointing=False, do_tie_weights=False):
        """
        This function resets the decoder (re-initialise with base checkpoint).

        Args:
            checkpoint_name: str
                The name of the base checkpoint to re-initialise the decoder with, default: 'roberta=base'
            gradient_checkpointing: bool
                Whether or not to use gradient checkpointing, default: False
            do_tie_weights: bool
                Whether or not the weights between encoder and decoder are shared (for warning), default: False

        """

        print("Checking if shared_weights == False, yields {}".format(do_tie_weights == False))
        assert do_tie_weights == False, "Not resetting the decoder if the weights are shared. Aborting!"
        print(f"Resetting the decoder to {checkpoint_name} checkpoint.")
        self.decoder = VAE_Decoder_RobertaForCausalLM.from_pretrained(checkpoint_name,
                                                                      gradient_checkpointing=gradient_checkpointing)
