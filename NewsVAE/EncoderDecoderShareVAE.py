import torch.nn as nn
from transformers import RobertaTokenizer, RobertaConfig, AutoModelForCausalLM, AutoModel  # type: ignore
from utils_external import tie_weights  # type: ignore
from transformers.modeling_outputs import BaseModelOutputWithPooling, CausalLMOutput
from typing import Optional
import torch
import types
import NewsVAEArguments
from VAE_Decoder_Roberta import VAE_Decoder_RobertaForCausalLM


class EncoderDecoderShareVAE(nn.Module):
    def __init__(self, roberta_ckpt_name: str = "roberta-base"):
        super(EncoderDecoderShareVAE, self).__init__()

        # Tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_ckpt_name)

        # Encoder
        config_encoder = RobertaConfig.from_pretrained(roberta_ckpt_name)
        config_encoder.return_dict = True

        # TODO: careful, the pooling layer now is different from how it's done in the Optimus paper
        # there they have a linear layer without non-linearity
        self.encoder = AutoModel.from_config(config_encoder)

        # Decoder
        self.decoder = VAE_Decoder_RobertaForCausalLM.from_pretrained(roberta_ckpt_name)

        # Tie the weights of the Encoder and Decoder
        base_model_prefix = self.decoder.base_model_prefix
        tie_weights(self.encoder, self.decoder._modules[base_model_prefix], base_model_prefix)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """

        :param input_ids: batch of sequences of token ids
        :param attention_mask: 2d attention mask, masking the padded tokens
        :return: kl_loss and recon_loss
                (kl divergence loss and reconstruction loss = cross entropy loss)
        """

        # Forward the encoder
        encoder_outs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Pool the hidden features of the first token (classification token)
        pooled_output = encoder_outs.pooler_output

        # Make use of the outputted features to sample a latent z vector
        latent_z, kl_loss = self.connect_encoder_decoder(pooled_output)

        # TODO: make sure the attention mask does what it is supposed to d
        # TODO: connect latent to decoder
        decoder_outs = self.decoder(input_ids=input_ids, attention_mask=attention_mask,
                                    latent_z=latent_z, labels=input_ids)

        # decoder_outs = self.decoder(input_ids, attention_mask)

        recon_loss = decoder_outs.loss

        print("recon_loss", recon_loss)
        print("kl_loss", kl_loss)

        return kl_loss, recon_loss

    def train_step(self, batch):
        pass

    @staticmethod
    def reparameterize(mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample from posterior Gaussian family.

        :param mu: mean of the posterior (batch_size x vae_latent_dim)
        :param logvar: log variance of the posterior (batch_size x vae_latent_dim)

        :return: sampled_latent: sampled latent (batch_size x vae_latent_dim)
        """
        batch_size, nz = mu.size()
        std = torch.mul(logvar, 0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, 1, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, 1, nz)

        eps = torch.zeros_like(std_expd).normal_()

        sampled_latent = mu_expd + torch.mul(eps, std_expd)
        sampled_latent = sampled_latent.squeeze(1)

        return sampled_latent

    def connect_encoder_decoder(self, encoder_pooled_output: torch.FloatTensor,
                                deterministic: bool = False):
        """
        This function connects the encoder to the decoder, either deterministically
        or probabilistically, in which case a sample is drawn.

        :param encoder_pooled_output:
        :param deterministic:
        :return: latent_z
        """
        # Interpret the latent to be mean, log variance vector
        # to ensure positivity for the standard deviation
        # Chunk in half not over batch but representation dimension
        mu, logvar = encoder_pooled_output.chunk(2, dim=1)

        # Deterministic connection: take the posterior mean as latent z
        if deterministic:
            latent_z = mu

        # Sample with reparameterization trick
        else:
            latent_z = self.reparameterize(mu, logvar)

        loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)

        # TODO: add mask on KL loss (find out what THRESH was self.args.dim_target_kl)
        kl_mask = (loss_kl > 3).float()
        loss_kl = (kl_mask * loss_kl).sum(dim=1)

        return latent_z, loss_kl


if __name__ == "__main__":
    model = EncoderDecoderShareVAE()
