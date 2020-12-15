import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AutoModelForCausalLM, AutoModel  # type: ignore
from utils_external import tie_weights  # type: ignore
import torch
import argparse
from VAE_Decoder_Roberta import VAE_Decoder_RobertaForCausalLM, VAE_Decoder_RobertaPooler
import copy
from NewsVAEArguments import preprare_parser
import utils
import math


class EncoderDecoderShareVAE(nn.Module):
    def __init__(self, args, roberta_ckpt_name: str = "roberta-base", do_tie_weights=True):
        super(EncoderDecoderShareVAE, self).__init__()
        self.arguments = args

        # Tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_ckpt_name)

        # Encoder
        # Don't use the pre-trained pooler
        self.encoder = RobertaModel.from_pretrained(roberta_ckpt_name, add_pooling_layer=False,
                                                    return_dict=True,
                                                    gradient_checkpointing=args.gradient_checkpointing)

        # Add a fresh pooling layer (different size) & init weights
        self.encoder.pooler = VAE_Decoder_RobertaPooler(self.encoder.config, args.latent_size)
        self.encoder.pooler.dense.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)

        # Decoder
        self.decoder = VAE_Decoder_RobertaForCausalLM.from_pretrained(roberta_ckpt_name,
                                                                      gradient_checkpointing=args.gradient_checkpointing)
        self.decoder.add_latent_projection_layers(args.latent_size, args.hidden_size, args.n_layers,
                                                  args.add_latent_via_memory, args.add_latent_via_embeddings)

        self.latent_size = args.latent_size

        # Tie the weights of the Encoder and Decoder (encoder weights are pointers to decoder weights)
        if do_tie_weights:
            base_model_prefix = self.decoder.base_model_prefix
            tie_weights(self.encoder, self.decoder._modules[base_model_prefix], base_model_prefix)

    def forward(self, input_ids, attention_mask,
                beta, args, return_predictions,
                return_exact_match_acc):
        """
        Implements the forward pass of the shared Encoder-Decoder VAE.

        :param return_predictions:
        :param beta: how to balance the KL-loss with the reconstruction loss in beta-vae
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

        mmd_loss = self.compute_maximum_mean_discrepancy(latent_z)
        mmd_loss = mmd_loss * args.mmd_lambda

        # Forward the decoder
        decoder_outs = self.decoder(input_ids=input_ids, attention_mask=attention_mask,
                                    latent_z=latent_z, labels=copy.copy(input_ids),
                                    add_latent_via_embeddings=args.add_latent_via_embeddings,
                                    add_latent_via_memory=args.add_latent_via_memory,
                                    return_cross_entropy=True,
                                    output_attentions=True,
                                    return_predictions=return_predictions,
                                    return_exact_match_acc=return_exact_match_acc)

        recon_loss = decoder_outs["cross_entropy"]

        if args.objective == 'beta-vae':
            total_loss = recon_loss + (beta * hinge_kl_loss)
        elif args.objective == 'mmd-vae':
            total_loss = recon_loss + mmd_loss
        else:
            print("Not supported objective. Set valid option: beta-vae or mmd-vae.")

        # Detach all except the total loss on which we need to base our backward pass
        losses = {'kl_loss': kl_loss.item(), 'hinge_kl_loss': hinge_kl_loss.item(),
                  'recon_loss': recon_loss.item(), 'total_loss': total_loss,
                  'mmd_loss': mmd_loss.item()}

        if return_predictions:
            losses['logits'] = decoder_outs["logits"]
            losses["predictions"] = decoder_outs["predictions"]

        if return_exact_match_acc:
            losses["exact_match_acc"] = decoder_outs["exact_match_acc"].item()

        return losses

    def reset_decoder(self, args):
        print("Checking if shared_weights == False, yields {}".format(self.arguments.do_tie_weights is False))
        assert not args.do_tie_weights, "Not resetting the decoder if the weights are shared. Aborting!"
        print(f"Resetting the decoder to {args.base_checkpoint_name} checkpoint.")
        self.decoder = VAE_Decoder_RobertaForCausalLM.from_pretrained(args.base_checkpoint_name,
                                                                      gradient_checkpointing=args.gradient_checkpointing)
        self.decoder.add_latent_projection_layers(args.latent_size, args.hidden_size, args.n_layers,
                                                  args.add_latent_via_memory, args.add_latent_via_embeddings)

    @staticmethod
    def compute_kernel(x, y):
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

    def compute_maximum_mean_discrepancy(self, latent_z):
        """
        Taken from: https://github.com/aktersnurra/information-maximizing-variational-autoencoders/blob/master/model/loss_functions/mmd_loss.py

        :param latent_z:
        :return:
        """
        x = torch.randn_like(latent_z).to(latent_z.device)
        y = latent_z
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

    @staticmethod
    def reparameterize(mu, logvar, n_samples=1):
        """Sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, n_samples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, n_samples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)

    def sample(self, encoder_pooled_output, n_samples):
        # interpret output from encoder as mu, logvar vector
        mu, logvar = encoder_pooled_output.chunk(2, dim=1)

        # batch x n_samples x seq_len
        latent_z = self.reparameterize(mu, logvar, n_samples=n_samples)

        return latent_z, (mu, logvar)

    def connect_encoder_decoder(self, encoder_pooled_output: torch.FloatTensor,
                                deterministic: bool = False,
                                hinge_loss_lambda: float = 0.5,
                                return_mu_logvar: bool = False):
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
            latent_z = self.reparameterize(mu, logvar, n_samples=1).squeeze(1)

        kl_loss = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)

        # Ignore the dimensions of which the KL-div is already under the
        # threshold, avoiding driving it down even further. Those values do
        # not have to be replaced by the threshold because that would not mean
        # anything to the gradient. That's why they are simply removed. This
        # confused me at first.
        kl_mask = (kl_loss > hinge_loss_lambda).float()

        # Sum over the latent dimensions and average over the batch dimension
        hinge_kl_loss = (kl_mask * kl_loss).sum(dim=1).mean(dim=0)
        kl_loss = kl_loss.sum(dim=1).mean(dim=0)

        if return_mu_logvar:
            return latent_z, kl_loss, hinge_kl_loss, (mu, logvar)
        else:
            return latent_z, kl_loss, hinge_kl_loss

    def eval_complete_ll(self, input_ids, attention_mask, latent_z, args):
        """
        compute log p(z,x) = log joint

        Args:
            x: Tensor
                input with shape [batch, seq_len]
            z: Tensor
                evaluation points with shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log p(z,x) Tensor with shape [batch, nsamples]

        """

        # [batch, nsamples]
        log_prior = self.eval_prior_dist(latent_z)
        log_gen = self.eval_cond_ll(input_ids, attention_mask, latent_z, args)

        # p(x, z) = p(z) * p(x|z) -> log p(x, z) = log p(z) + log p(x|z)
        log_p_z_x = log_prior + log_gen

        return log_p_z_x

    def eval_prior_dist(self, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/space
        """

        # (k^2)

        loc = torch.zeros(self.latent_size, device=zrange.device)
        scale = torch.ones(self.latent_size, device=zrange.device)

        return torch.distributions.normal.Normal(loc, scale).log_prob(zrange).sum(dim=-1)

    def eval_cond_ll(self, input_ids, attention_mask, latent_z, args):
        """
        compute log p(x|z) = log likelihood

        """

        assert len(latent_z.shape) == 3, "only implemented for the multi-sample case"

        batch_size, seq_len = input_ids.shape
        n_samples = latent_z.shape[1]

        losses = []

        # Loop over batch dimension, sample dimension is interpreted as batch dimension
        for i in range(batch_size):
            z = latent_z[i, :, :].squeeze(0)
            x = input_ids[i, :].expand(n_samples, seq_len)
            a = attention_mask[i, :].expand(n_samples, seq_len)

            # Forward the decoder
            decoder_outs = self.decoder(input_ids=x, attention_mask=a,
                                        latent_z=z, labels=copy.copy(x),
                                        add_latent_via_embeddings=args.add_latent_via_embeddings,
                                        add_latent_via_memory=args.add_latent_via_memory,
                                        return_cross_entropy=True,
                                        reduce_loss=False,
                                        return_predictions=False,
                                        return_exact_match_acc=False)

            recon_loss = decoder_outs["cross_entropy"]

            losses.append(recon_loss)

        # Stack so the dimensions are batch_size x n_samples
        losses = torch.stack(losses)

        return - losses  # TODO: minus? I think so...

    def eval_inference_dist(self, input_ids, attention_mask, z, param=None):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z.size(2)

        if not param:
            encoder_outs = self.encoder(input_ids=input_ids,
                                        attention_mask=attention_mask)
            mu, logvar = encoder_outs.pooler_output.chunk(2, dim=1)
        else:
            mu, logvar = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - 0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density


if __name__ == "__main__":
    params = preprare_parser()

    model = EncoderDecoderShareVAE(args=params, do_tie_weights=True)

    print("With tying weights")
    print('Trainable params {:.3f} x 1e6'.format(utils.get_number_of_params(model) / 1e6))
    print("of which are encoder weights: {:.3f} x 1e6".format(utils.get_number_of_params(model.encoder) / 1e6))
    print("of which are decoder weights: {:.3f} x 1e6".format(utils.get_number_of_params(model.decoder) / 1e6))

    # model = EncoderDecoderShareVAE(args=params, do_tie_weights=False)
    #
    # print("Without tying weights")
    # print('Trainable params {:.3f} x 1e6'.format(utils.get_number_of_params(model)/1e6))
    # print("of which are encoder weights: {:.3f} x 1e6".format(utils.get_number_of_params(model.encoder)/1e6))
    # print("of which are decoder weights: {:.3f} x 1e6".format(utils.get_number_of_params(model.decoder) / 1e6))
