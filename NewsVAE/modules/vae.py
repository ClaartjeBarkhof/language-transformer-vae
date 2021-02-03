import torch.nn as nn
from utils_external import tie_weights
from utils_evaluation import tokenizer_batch_decode
import torch
from modules.decoder import DecoderNewsVAE
from modules.encoder import EncoderNewsVAE
import copy


class NewsVAE(torch.nn.Module):
    def __init__(self, encoder, decoder,
                 do_tie_weights=True, do_tie_embedding_spaces=True):
        super(NewsVAE, self).__init__()

        # Main parts
        self.encoder = encoder
        self.decoder = decoder

        # Weight tying / sharing between encoder and decoder RoBERTa part
        if do_tie_weights:
            print("Tying encoder decoder RoBERTa checkpoint weights!")
            base_model_prefix = self.decoder.model.base_model_prefix
            tie_weights(self.encoder.model, self.decoder.model._modules[base_model_prefix], base_model_prefix)

        # Make all embedding spaces the same (encoder input, decoder input, decoder output)
        if do_tie_embedding_spaces:
            print("Tying embedding spaces!")
            self.tie_all_embeddings()

    def tie_all_embeddings(self):
        # Get all relevant embeddings
        encoder_input_embeddings = self.encoder.model.embeddings.word_embeddings
        decoder_input_embeddings = self.decoder.model.roberta.embeddings.word_embeddings
        decoder_output_layer = self.decoder.model.lm_head.decoder

        # Set all equal to encoder input embeddings
        decoder_input_embeddings.weight = encoder_input_embeddings.weight
        decoder_output_layer.weight = encoder_input_embeddings.weight

        # Pad bias in decoder output layer if necessary (not really sure when this needs to happen).
        if getattr(decoder_output_layer, "bias", None) is not None:
            decoder_output_layer.bias.data = torch.nn.functional.pad(decoder_output_layer.bias.data, (
                0, decoder_output_layer.weight.shape[0] - decoder_output_layer.bias.shape[0],), "constant", 0, )

        # Set out_features and num_embeddings features to correct value
        if hasattr(decoder_output_layer, "out_features") and hasattr(encoder_input_embeddings, "num_embeddings"):
            decoder_input_embeddings.out_features = encoder_input_embeddings.num_embeddings

        if hasattr(decoder_input_embeddings, "num_embeddings") and hasattr(encoder_input_embeddings, "num_embeddings"):
            decoder_input_embeddings.num_embeddings = encoder_input_embeddings.num_embeddings

    def forward(self, input_ids, beta, attention_mask,
                auto_regressive=False,

                objective="beta-vae",
                hinge_kl_loss_lambda=0.5,

                return_latents=False,
                return_log_q_z_x=False,
                return_log_p_z=False,
                return_mu_logvar=False,

                return_exact_match=False,
                return_cross_entropy=True,

                return_embedding_distance=False,
                reduce_seq_dim_embedding_loss="mean",
                reduce_batch_dim_embedding_loss="mean",

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

                decode_sample_from_prior=False,

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
            device_name: str
        Returns:
            losses: Dict[str, Union[float, Tensor]
                The result dictionary of the full forward pass with metrics
                and possibly predictions.
        """
        # Forward through encoder and sample
        enc_out = self.encoder.encode(input_ids=input_ids, attention_mask=attention_mask,
                                      n_samples=1, hinge_kl_loss_lambda=hinge_kl_loss_lambda,
                                      return_log_q_z_x=return_log_q_z_x, return_log_p_z=return_log_p_z,
                                      return_embeddings=return_embedding_distance)

        beta_hinge_kl = beta * enc_out["hinge_kl_loss"]


        if decode_sample_from_prior:
            latent_z = self.sample_from_prior(latent_size=self.decoder.latent_size,
                                         n_samples=input_ids.shape[0],
                                         device_name=input_ids.get_device())
        else:
            latent_z = enc_out["latent_z"]

        if auto_regressive is False:
            decoder_outs = self.decoder(latent_z, input_ids, attention_mask,
                                        return_attention_probs=return_attention_probs,
                                        return_attention_to_latent=return_attention_to_latent,
                                        return_hidden_states=return_hidden_states,
                                        return_exact_match=return_exact_match,
                                        return_predictions=return_predictions,
                                        return_output_word_embeddings=return_embedding_distance,
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
                return_output_word_embeddings=return_embedding_distance,
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

        ########################
        # EMBEDDING SPACE LOSS #
        ########################

        embedding_loss = None
        if return_embedding_distance:
            if auto_regressive is False:
                # cut off last token prediction (already done with auto-regressive)
                decoder_outs["output_word_embeddings"] = decoder_outs["output_word_embeddings"][:, :-1, :]
            embedding_loss = self.calculate_embedding_space_loss(copy.deepcopy(input_ids), enc_out["word_embeddings"],
                                                                 decoder_outs["output_word_embeddings"],
                                                                 reduce_seq_dim_embedding_loss,
                                                                 reduce_batch_dim_embedding_loss)

            embedding_loss = embedding_loss.item()
            del decoder_outs["output_word_embeddings"]

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

        if return_cross_entropy and auto_regressive is False:
            ce_per_word = decoder_outs["cross_entropy_per_word"].item()
        else:
            ce_per_word = None

        # Construct the total loss
        total_loss = None

        if objective == 'beta-vae':
            total_loss = recon_loss + beta_hinge_kl
        elif objective == 'mmd-vae':
            total_loss = recon_loss + enc_out["mmd_loss"]
        else:
            print("Not supported objective. Set valid option: beta-vae or mmd-vae.")

        minus_elbo = recon_loss + enc_out["kl_loss"]

        # Detach all except the total loss on which we need to base our backward pass
        vae_outputs = {'kl_loss': enc_out["kl_loss"].item(),
                       'hinge_kl_loss': enc_out["hinge_kl_loss"].item(),
                       'beta_hinge_kl_loss': beta_hinge_kl.item(),
                       'recon_loss': recon_loss.item(),
                       'total_loss': total_loss,
                       'mmd_loss': enc_out["mmd_loss"].item(),
                       'log_q_z_x': enc_out["log_q_z_x"],
                       "log_p_z": enc_out["log_p_z"],
                       "-ELBO": minus_elbo.item(),
                       "embedding_loss": embedding_loss,
                       "ce_per_word": ce_per_word}

        # Delete to avoid confusion
        # del decoder_outs['cross_entropy']

        if return_latents:
            vae_outputs["latents"] = enc_out["latent_z"]

        if return_mu_logvar:
            vae_outputs["mu"] = enc_out["mu"].detach()
            vae_outputs["logvar"] = enc_out["logvar"].detach()

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

    def calculate_embedding_space_loss(self, input_ids, in_w_emb, out_w_emb,
                                       reduce_seq_dim_embedding_loss,
                                       reduce_batch_dim_embedding_loss):
        labels = input_ids[:, 1:].contiguous()  # skip <s> token

        # pad token is int 1
        label_mask = (labels != 1).float()

        # cut off start token
        in_w_emb = in_w_emb[:, 1:, :]

        latent_size = in_w_emb.shape[-1]
        embedding_loss = torch.nn.functional.mse_loss(in_w_emb.reshape(-1, latent_size),
                                                      out_w_emb.reshape(-1, latent_size),
                                                      reduce=False, reduction='none')
        embedding_loss = embedding_loss.mean(dim=-1)
        embedding_loss = embedding_loss.reshape(labels.shape)
        embedding_loss = embedding_loss * label_mask

        if reduce_seq_dim_embedding_loss == "mean":
            embedding_loss = embedding_loss.mean(dim=-1)
        elif reduce_seq_dim_embedding_loss == "sum":
            embedding_loss = embedding_loss.sum(dim=-1)

        if reduce_batch_dim_embedding_loss == "mean":
            embedding_loss = embedding_loss.mean(dim=0)
        elif reduce_batch_dim_embedding_loss == "sum":
            embedding_loss = embedding_loss.sum(dim=0)

        return embedding_loss



if __name__ == "__main__":
    latent_dim = 768
    embedding_mechanism = True
    memory_mechanism = True
    tied_weights = True

    dec_model = DecoderNewsVAE(gradient_checkpointing=False,
                               add_latent_via_memory=memory_mechanism,
                               add_latent_via_embeddings=embedding_mechanism,
                               latent_size=latent_dim)
    enc_model = EncoderNewsVAE(gradient_checkpointing=False, latent_size=latent_dim)

    vae_model = NewsVAE(enc_model, dec_model,
                        do_tie_weights=tied_weights)

    print("done loading VAE!")
