from utils_external import tie_weights
# from utils_evaluation import tokenizer_batch_decode
import torch
from loss_and_optimisation import LossTermManager
from modules.decoder import DecoderNewsVAE
from modules.encoder import EncoderNewsVAE
import copy


class NewsVAE(torch.nn.Module):
    def __init__(self, encoder, decoder, dataset_size, config):
        super(NewsVAE, self).__init__()
        """
        This class implements a VAE by wrapping an EncoderNewsVAE and DecoderNewsVAE.
        """

        # Main parts
        self.encoder = encoder
        self.decoder = decoder

        self.dataset_size = dataset_size

        self.config = config
        self.objective = config.objective

        # Weight tying / sharing between encoder and decoder RoBERTa part
        if config.do_tie_weights:
            print("Tying encoder decoder RoBERTa checkpoint weights!")
            base_model_prefix = self.decoder.model.base_model_prefix
            tie_weights(self.encoder.model, self.decoder.model._modules[base_model_prefix], base_model_prefix)

        # Make all embedding spaces the same (encoder input, decoder input, decoder output)
        if config.do_tie_embedding_spaces:
            print("Tying embedding spaces!")
            self.tie_all_embeddings()

    def tie_all_embeddings(self):
        """
        This function ties all embedding matrices: input encoder, input decoder, output decoder.
        """

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

    def forward(self,
                input_ids,
                attention_mask=None,

                auto_regressive=False,

                return_latents=False,
                return_mu_logvar=False,

                return_exact_match=False,
                return_cross_entropy=False,
                return_reconstruction_loss=True,

                return_embedding_distance=False,

                return_predictions=False,
                return_probabilities=False,
                return_logits=False,

                return_hidden_states=False,
                return_last_hidden_state=False,

                return_attention_to_latent=False,
                return_attention_probs=False,

                return_text_predictions=False,
                tokenizer=None,
                loss_term_manager=None,

                return_posterior_stats=True,

                reduce_seq_dim_ce="mean",
                reduce_seq_dim_exact_match="mean",
                reduce_batch_dim_exact_match="mean",
                reduce_batch_dim_ce="mean",

                nucleus_sampling=False,
                top_k=0,
                top_p=1.0,

                decode_sample_from_prior=False,

                device_name="cuda:0"):

        """
        Perform a forward pass through the whole VAE with the sampling operation in between (reparameterisation).
        """

        # Forward through encoder and sample (reparameterisation)
        enc_out = self.encoder.encode(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      n_samples=1,
                                      dataset_size=self.dataset_size,
                                      return_log_q_z_x=True,
                                      return_log_p_z=True,
                                      return_log_q_z=True,
                                      return_embeddings=False)

        # If instead use a sample from the prior, go ahead and sample it
        if decode_sample_from_prior:
            latent_z = self.sample_from_prior(latent_size=self.decoder.latent_size,
                                              n_samples=input_ids.shape[0],
                                              device_name=input_ids.get_device())
        else:
            latent_z = enc_out["latent_z"]

        # Parallel predictions with teacher forcing (during training)
        if auto_regressive is False:
            dec_out = self.decoder(latent_z, input_ids, attention_mask,
                                   return_attention_probs=return_attention_probs,
                                   return_attention_to_latent=return_attention_to_latent,
                                   return_hidden_states=return_hidden_states,
                                   return_exact_match=return_exact_match,
                                   return_predictions=return_predictions,
                                   return_probabilities=return_probabilities,
                                   return_last_hidden_state=return_last_hidden_state,
                                   return_output_embeddings=return_embedding_distance,
                                   return_logits=return_logits,
                                   return_cross_entropy=return_cross_entropy,
                                   return_reconstruction_loss=return_reconstruction_loss,
                                   reduce_seq_dim_ce=reduce_seq_dim_ce,
                                   reduce_seq_dim_exact_match=reduce_seq_dim_exact_match,
                                   reduce_batch_dim_exact_match=reduce_batch_dim_exact_match,
                                   reduce_batch_dim_ce=reduce_batch_dim_ce,
                                   nucleus_sampling=nucleus_sampling,
                                   top_k=top_k,
                                   top_p=top_p,
                                   labels=copy.copy(input_ids))

        # Recurrent, auto-regressive predictions during inference
        else:
            dec_out = self.decoder.autoregressive_decode(
                latent_z,
                labels=copy.copy(input_ids),
                max_seq_len=input_ids.shape[1],
                return_exact_match=return_exact_match,
                return_cross_entropy=return_cross_entropy,
                return_reconstruction_loss=return_reconstruction_loss,
                return_attention_probs=return_attention_probs,
                return_attention_to_latent=return_attention_to_latent,
                return_hidden_states=return_hidden_states,
                return_last_hidden_state=return_last_hidden_state,
                return_output_embeddings=return_embedding_distance,
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

        # Return text predictions
        # if return_text_predictions and tokenizer is not None:
        #     dec_out["text_predictions"] = tokenizer_batch_decode(dec_out["predictions"], tokenizer)

        ########################
        # TOTAL TRAIN LOSS     # <- not anymore
        ########################

        loss_dict = dict()

        # Detach all except the total loss on which we need to base our backward pass
        loss_dict["log_q_z"] = enc_out["log_q_z"] #.item()
        loss_dict["log_p_z"] = enc_out["log_p_z"] #.item()
        loss_dict["log_q_z_x"] = enc_out["log_q_z_x"] #.item()
        loss_dict["log_q_z_prod_marg"] = enc_out["log_q_z_prod_marg"] #.item()
        loss_dict["ce_per_word"] = dec_out["cross_entropy_per_word"].item() if return_cross_entropy and auto_regressive is False else None

        if return_latents:
            loss_dict["latents"] = enc_out["latent_z"] #.detach()

        if return_mu_logvar:
            loss_dict["mu"] = enc_out["mu"] #.detach()
            loss_dict["logvar"] = enc_out["logvar"] #.detach()

        if return_posterior_stats and enc_out["mu"] is not None:
            mu, var = enc_out["mu"], enc_out["logvar"].exp()
            mean_mu = mu.mean(dim=1).mean(dim=0)
            var_mu = torch.var(mu, dim=1).mean(dim=0)

            mean_var = var.mean(dim=1).mean(dim=0)
            var_var = torch.var(var, dim=1).mean(dim=0)

            loss_dict["mean_mu"] = mean_mu.item()
            loss_dict["var_mu"] = var_mu.item()
            loss_dict["mean_var"] = mean_var.item()
            loss_dict["var_var"] = var_var.item()

        # Merge all the outputs together
        vae_outputs = {**loss_dict, **dec_out}

        # Detach everything except for non-tensors and total loss
        # for k, v in vae_outputs.items():
        #     if torch.is_tensor(v) and k != "total_loss":
        #         vae_outputs[k] = v.detach()

        # Delete all that is None
        key_list = list(vae_outputs.keys())
        for k in key_list:
            if vae_outputs[k] is None:
                del vae_outputs[k]

        return vae_outputs

    @staticmethod
    def calculate_embedding_space_loss(input_ids, in_w_emb, out_w_emb,
                                       reduce_seq_dim_embedding_loss,
                                       reduce_batch_dim_embedding_loss):

        labels = input_ids[:, 1:].contiguous()  # skip <s> token

        # pad token is int 1
        label_mask = (labels != 1).float()

        # cut off start token
        # end token is already cut off for out_w_emb
        in_w_emb = in_w_emb[:, 1:, :]

        embedding_loss = torch.nn.functional.mse_loss(in_w_emb, out_w_emb,
                                                      reduce=False, reduction='none')
        embedding_loss = embedding_loss.mean(dim=-1)
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

    @staticmethod
    def sample_from_prior(latent_size=768, n_samples=8, device_name="cuda:0"):
        return EncoderNewsVAE.sample_from_prior(latent_size=latent_size, n_samples=n_samples, device_name=device_name)


def get_loss_term_manager_with_model(config, world_master=True,
                                     dataset_size=42068, device_name="cuda:0"):
    # Get model
    vae_model = get_model_on_device(config, dataset_size=dataset_size,
                                    device_name=device_name, world_master=world_master)

    # Init loss term manager
    loss_term_manager = LossTermManager(vae_model, config=config)
    if config.objective == "beta-vae":
        if config.b_vae_beta_constant_linear_lagrangian == "lagrangian":
            loss_term_manager.manager["beta_KL"]["constraint"] = \
                loss_term_manager.manager["beta_KL"]["constraint"].to(device_name)

    if config.objective == "beta-tc-vae":
        if config.b_tc_vae_alpha_constant_linear_lagrangian == "lagrangian":
            loss_term_manager.manager["alpha_MI"]["constraint"] = \
                loss_term_manager.manager["alpha_MI"]["constraint"].to(device_name)

        if config.b_tc_vae_gamma_constant_linear_lagrangian == "lagrangian":
            loss_term_manager.manager["gamma_DimKL"]["constraint"] = \
                loss_term_manager.manager["gamma_DimKL"]["constraint"].to(device_name)

    return loss_term_manager

def get_model_on_device(config, dataset_size=42068, device_name="cuda:0", world_master=True):
    """
    Load a fresh VAE model on correct device.
    """

    if world_master: print("Loading model...")

    decoder = DecoderNewsVAE(gradient_checkpointing=config.gradient_checkpointing,
                             add_latent_via_memory=config.add_latent_via_memory,
                             add_latent_via_embeddings=config.add_latent_via_embeddings,
                             latent_size=config.latent_size,
                             add_decoder_output_embedding_bias=config.add_decoder_output_embedding_bias,
                             drop_inputs_decoder=config.drop_inputs_decoder,
                             drop_inputs_decoder_prob=config.drop_inputs_decoder_prob)

    encoder = EncoderNewsVAE(gradient_checkpointing=config.gradient_checkpointing,
                             latent_size=config.latent_size)

    vae_model = NewsVAE(encoder, decoder, dataset_size, config)

    vae_model = vae_model.to(device_name)

    if world_master: print("Done loading model...")

    return vae_model