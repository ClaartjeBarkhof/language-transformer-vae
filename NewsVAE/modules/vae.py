import torch.nn as nn
from utils_external import tie_weights
import torch
from modules.decoder import DecoderNewsVAE
from modules.encoder import EncoderNewsVAE


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

                reduce_seq_dim_ce="sum",
                reduce_seq_dim_exact_match="mean",
                reduce_batch_dim_exact_match="mean",

                nucleus_sampling=False,
                top_k=0,
                top_p=0.9):

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
                                    nucleus_sampling=nucleus_sampling,
                                    top_k=top_k,
                                    top_p=top_p)

        # Construct the total loss
        total_loss = None
        if objective == 'beta-vae':
            total_loss = decoder_outs["cross_entropy"] + beta_hinge_kl
        elif objective == 'mmd-vae':
            total_loss = decoder_outs["cross_entropy"] + mmd_loss
        else:
            print("Not supported objective. Set valid option: beta-vae or mmd-vae.")

        # Detach all except the total loss on which we need to base our backward pass
        vae_outputs = {'kl_loss': kl_loss.item(),
                       'hinge_kl_loss': hinge_kl_loss.item(),
                       'beta_hinge_kl_loss': beta_hinge_kl.item(),
                       'recon_loss': decoder_outs["cross_entropy"].item(),
                       'total_loss': total_loss,
                       'mmd_loss': mmd_loss.item()}

        # Delete to avoid confusion
        del decoder_outs['cross_entropy']

        if return_latents:
            vae_outputs["latents"] = latent_z

        if return_mu_logvar:
            vae_outputs["mu_logvar"] = torch.cat([mu, logvar], dim=1)

        # Merge all the outputs together
        vae_outputs = {**vae_outputs, **decoder_outs}

        # Detach everything except for floats and total loss
        for k, v in vae_outputs.items():
            if type(v) != float:
                if k != "total_loss":
                    vae_outputs[k] = v.detach()

        return vae_outputs


if __name__ == "__main__":
    latent_size = 768
    embedding_mechanism = True
    memory_mechanism = True
    tied_weights = True

    dec_model = DecoderNewsVAE(gradient_checkpointing=False, )
    enc_model = EncoderNewsVAE(gradient_checkpointing=False, latent_size=latent_size)

    vae_model = NewsVAE(enc_model, dec_model, latent_size=latent_size,
                        add_latent_via_memory=memory_mechanism,
                        add_latent_via_embeddings=embedding_mechanism,
                        do_tie_weights=tied_weights)

    print("done loading VAE!")
