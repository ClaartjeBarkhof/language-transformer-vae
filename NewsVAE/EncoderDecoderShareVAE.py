import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AutoModelForCausalLM, AutoModel  # type: ignore
from utils_external import tie_weights  # type: ignore
import torch
import argparse
from VAE_Decoder_Roberta import VAE_Decoder_RobertaForCausalLM, VAE_Decoder_RobertaPooler
import copy


class EncoderDecoderShareVAE(nn.Module):
    def __init__(self, args, roberta_ckpt_name: str = "roberta-base"):
        super(EncoderDecoderShareVAE, self).__init__()

        # Tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_ckpt_name)

        # Encoder
        # Don't use the pre-trained pooler
        self.encoder = RobertaModel.from_pretrained(roberta_ckpt_name, add_pooling_layer=False,
                                                    return_dict=True)
        # Add a fresh pooling layer
        self.encoder.pooler = VAE_Decoder_RobertaPooler(self.encoder.config, args.latent_size)

        # Decoder
        self.decoder = VAE_Decoder_RobertaForCausalLM.from_pretrained(roberta_ckpt_name)
        self.decoder.add_latent_projection_layers(args.latent_size, args.hidden_size, args.n_layers)

        # Tie the weights of the Encoder and Decoder
        base_model_prefix = self.decoder.base_model_prefix
        tie_weights(self.encoder, self.decoder._modules[base_model_prefix], base_model_prefix)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                args: argparse.Namespace):
        """
        Implements the forward pass of the shared Encoder-Decoder VAE.

        :param args: configuration parameters of the VAE
        :param input_ids: batch of sequences of token ids
        :param attention_mask: 2d attention mask, masking the padded tokens

        :return: kl_loss (KL divergence) and recon_loss (cross-entropy loss)
        """

        # Forward the encoder
        encoder_outs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Make use of the pooled features to sample a latent z vector
        latent_z, kl_loss, hinge_kl_loss = self.connect_encoder_decoder(encoder_outs.pooler_output,
                                                                        deterministic=args.deterministic_connect,
                                                                        hinge_loss_lambda=args.hinge_loss_lambda)

        # Forward the decoder
        decoder_outs = self.decoder(input_ids=input_ids, attention_mask=attention_mask,
                                    latent_z=latent_z, labels=copy.copy(input_ids),
                                    add_latent_via_embeddings=args.add_latent_via_embeddings,
                                    add_latent_via_memory=args.add_latent_via_memory)
        recon_loss = decoder_outs.loss

        total_loss = hinge_kl_loss + (args.beta * recon_loss)

        losses = {'kl_loss': kl_loss, 'hinge_kl_loss': hinge_kl_loss,
                  'recon_loss': recon_loss, 'total_loss': total_loss}

        return losses

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
                                deterministic: bool = False,
                                hinge_loss_lambda: float = 0.5):
        """
        This function connects the encoder to the decoder, either deterministically
        or probabilistically, in which case a sample is drawn.

        :param hinge_loss_lambda: maximum value the KL-loss elements can take (for hinge loss capping)
        :param encoder_pooled_output: pooled features from the encoder
        :param deterministic: whether or not to connect the encoder and decoder deterministically

        :return: latent_z: the latent vector that can be injected in the decoder
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

        kl_loss = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)

        # Cap the loss (hinge loss)
        hinge_kl_loss = torch.clamp(kl_loss, 0, hinge_loss_lambda)

        # Sum over the latent dimension and average over the batch dimension
        hinge_kl_loss = hinge_kl_loss.sum(dim=1).mean(dim=0)
        kl_loss = kl_loss.sum(dim=1).mean(dim=0)

        return latent_z, kl_loss, hinge_kl_loss


if __name__ == "__main__":
    print("Not implemented this main.")
