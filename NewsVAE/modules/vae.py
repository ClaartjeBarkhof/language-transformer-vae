import torch.nn as nn
from utils_external import tie_weights
from utils_evaluation import tokenizer_batch_decode
import torch
from modules.decoder import DecoderNewsVAE
from modules.encoder import EncoderNewsVAE
import copy

class NewsVAE(torch.nn.Module):
    def __init__(self, encoder, decoder,
                 latent_size=768,
                 add_latent_via_memory=True,
                 add_latent_via_embeddings=True,
                 do_tie_weights=True):
        super(NewsVAE, self).__init__()

        # Main parts
        self.encoder = encoder
        self.decoder = decoder

        # Weight tying / sharing
        if do_tie_weights:
            base_model_prefix = self.decoder.model.base_model_prefix
            tie_weights(self.encoder.model, self.decoder.model._modules[base_model_prefix], base_model_prefix)

    def forward(self, input_ids, beta, attention_mask,
                auto_regressive=False,

                objective="beta-vae",
                hinge_kl_loss_lambda=0.5,

                return_latents=False,
                return_mu_logvar=False,

                return_exact_match=False,
                return_cross_entropy= True,

                return_predictions=False,
                return_probabilities=False,
                return_logits=False,

                return_hidden_states=False,
                return_last_hidden_state=False,

                return_attention_to_latent=False,
                return_attention_probs=False,

                return_text_predictions=False,
                tokenizer=None,

                reduce_seq_dim_ce="sum",
                reduce_seq_dim_exact_match="mean",
                reduce_batch_dim_exact_match="mean",
                reduce_batch_dim_ce="mean",

                nucleus_sampling=False,
                top_k=0,
                top_p=0.9,

                device_name="cuda:0"):

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
            hinge_kl_loss_lambda: float
                Losses under this threshold are remitted.
            return_latents: bool
            return_mu_logvar: bool
            return_attention_probs: bool
            return_attention_to_latent: bool
            return_exact_match: bool
            return_predictions: bool
            return_probabilities: bool
            return_last_hidden_state: bool
            return_hidden_states: bool
            return_logits: bool
            return_cross_entropy: bool
            reduce_seq_dim_ce: str
            reduce_seq_dim_exact_match: str
            reduce_batch_dim_exact_match: str
            reduce_batch_dim_ce: str
            nucleus_sampling: bool
            top_k: int
            top_p: float
        Returns:
            losses: Dict[str, Union[float, Tensor]
                The result dictionary of the full forward pass with metrics
                and possibly predictions.
        """
        # Forward through encoder and sample
        mu, logvar, latent_z, kl_loss, hinge_kl_loss, mmd_loss = self.encoder.encode(input_ids=input_ids,
                                                                                     attention_mask=attention_mask,
                                                                                     n_samples=1,
                                                                                     hinge_kl_loss_lambda=hinge_kl_loss_lambda)
        beta_hinge_kl = beta * hinge_kl_loss

        if auto_regressive is False:
            decoder_outs = self.decoder(latent_z, input_ids, attention_mask,
                                        return_attention_probs=return_attention_probs,
                                        return_attention_to_latent=return_attention_to_latent,
                                        return_hidden_states=return_hidden_states,
                                        return_exact_match=return_exact_match,
                                        return_predictions=return_predictions,
                                        return_probabilities=return_probabilities,
                                        return_last_hidden_state=return_last_hidden_state,
                                        return_logits=return_logits,
                                        return_cross_entropy=return_cross_entropy,
                                        reduce_seq_dim_ce=reduce_seq_dim_ce,
                                        reduce_seq_dim_exact_match=reduce_seq_dim_exact_match,
                                        reduce_batch_dim_exact_match=reduce_batch_dim_exact_match,
                                        reduce_batch_dim_ce=reduce_batch_dim_ce,
                                        nucleus_sampling=nucleus_sampling,
                                        top_k=top_k,
                                        top_p=top_p,
                                        labels=copy.copy(input_ids))
        else:
            decoder_outs = self.decoder.autoregressive_decode(
                                        latent_z,
                                        labels=copy.copy(input_ids),
                                        max_seq_len=input_ids.shape[1],
                                        return_exact_match=return_exact_match,
                                        return_cross_entropy=return_cross_entropy,
                                        return_attention_probs=return_attention_probs,
                                        return_attention_to_latent=return_attention_to_latent,
                                        return_hidden_states=return_hidden_states,
                                        return_last_hidden_state=return_last_hidden_state,
                                        return_predictions=return_predictions,
                                        return_probabilities=return_probabilities,
                                        return_logits=return_logits,
                                        nucleus_sampling=nucleus_sampling,
                                        reduce_seq_dim_ce=reduce_seq_dim_ce,
                                        reduce_seq_dim_exact_match=reduce_seq_dim_exact_match,
                                        reduce_batch_dim_exact_match=reduce_batch_dim_exact_match,
                                        reduce_batch_dim_ce=reduce_batch_dim_ce,
                                        top_k=top_k,
                                        top_p=top_p,
                                        device_name=device_name
                                       )

        if return_text_predictions:
            if tokenizer is None:
                print("You have to provide a tokenizer in order to get text predictions.")
            else:
                decoder_outs["text_predictions"] = tokenizer_batch_decode(decoder_outs["predictions"], tokenizer)

        # Make sure the reconstruction loss that is part of the
        # total loss is always reduce with a sum over sequence dimension
        # and mean over batch dimension. This is called "recon_loss"
        # cross entropy may be reduced differently or not reduced
        recon_loss = decoder_outs["cross_entropy"]
        batch_size, seq_len = input_ids.shape

        if recon_loss.dim() != 0:
            if recon_loss.shape == (batch_size, seq_len - 1):
                recon_loss = recon_loss.sum(dim=1).mean(dim=0)
            elif len(recon_loss) == (seq_len - 1):
                recon_loss = recon_loss.sum()
            elif len(recon_loss) == batch_size:
                recon_loss = recon_loss.mean()

        # Construct the total loss
        total_loss = None

        if objective == 'beta-vae':
            total_loss = recon_loss + beta_hinge_kl
        elif objective == 'mmd-vae':
            total_loss = recon_loss + mmd_loss
        else:
            print("Not supported objective. Set valid option: beta-vae or mmd-vae.")

        # Detach all except the total loss on which we need to base our backward pass
        vae_outputs = {'kl_loss': kl_loss.item(),
                       'hinge_kl_loss': hinge_kl_loss.item(),
                       'beta_hinge_kl_loss': beta_hinge_kl.item(),
                       'recon_loss': recon_loss,
                       'total_loss': total_loss,
                       'mmd_loss': mmd_loss.item()}

        # Delete to avoid confusion
        # del decoder_outs['cross_entropy']

        if return_latents:
            vae_outputs["latents"] = latent_z

        if return_mu_logvar:
            vae_outputs["mu_logvar"] = torch.cat([mu, logvar], dim=1)

        # Merge all the outputs together
        vae_outputs = {**vae_outputs, **decoder_outs}

        # Detach everything except for floats and total loss
        for k, v in vae_outputs.items():
            if torch.is_tensor(v) and k != "total_loss":
                vae_outputs[k] = v.detach()

        # Delete all that is None
        key_list = list(vae_outputs.keys())
        for k in key_list:
            if vae_outputs[k] is None:
                del vae_outputs[k]

        return vae_outputs

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


if __name__ == "__main__":
    latent_dim = 768
    embedding_mechanism = True
    memory_mechanism = True
    tied_weights = True

    dec_model = DecoderNewsVAE(gradient_checkpointing=False, )
    enc_model = EncoderNewsVAE(gradient_checkpointing=False, latent_size=latent_dim)

    vae_model = NewsVAE(enc_model, dec_model, latent_size=latent_dim,
                        add_latent_via_memory=memory_mechanism,
                        add_latent_via_embeddings=embedding_mechanism,
                        do_tie_weights=tied_weights)

    print("done loading VAE!")
