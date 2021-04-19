import torch
from modules.decoder_roberta import VaeDecoderRobertaForCausalLM
from transformers import RobertaConfig
import torch.nn as nn


class DecoderNewsVAE(torch.nn.Module):
    def __init__(self,
                 gradient_checkpointing=False,
                 add_latent_via_memory=True,
                 add_latent_via_embeddings=True,
                 add_latent_via_cross_attention=False,
                 add_latent_via_gating=False,
                 latent_size=768,
                 add_decoder_output_embedding_bias=True,
                 drop_inputs_decoder=False,
                 drop_inputs_decoder_prob=0.2):
        """
        This class serves as a wrapper to the functional class VaeDecoderRobertaForCausalLM class.
        It additionally can take care of weight tying, auto-regressive decoding, etc. It also
        contains the weights to connect the latent representation to the decoder.
        """
        super(DecoderNewsVAE, self).__init__()

        self.add_latent_via_gating = add_latent_via_gating

        # The functional part of this class
        config = RobertaConfig.from_pretrained("roberta-base")
        self.model = VaeDecoderRobertaForCausalLM.from_pretrained("roberta-base",
                                                                  gradient_checkpointing=gradient_checkpointing)
        # A bit cumbersome, but delete the cross attention if not going to be used
        # (otherwise I need to change the from_pretrained function) <- I could also manipulate the .is_decoder and
        # .add_cross_attention attributes, but not doing that now
        if add_latent_via_cross_attention is False:
            for l in self.model.roberta.encoder.layer:
                del l.crossattention

        # Set dropout variables
        self.model.roberta.embeddings.drop_inputs_decoder = drop_inputs_decoder
        self.model.roberta.embeddings.drop_inputs_decoder_prob = drop_inputs_decoder_prob

        print("self.model.roberta.embeddings.drop_inputs_decoder ", self.model.roberta.embeddings.drop_inputs_decoder)
        print("self.model.roberta.embeddings.drop_inputs_decoder_prob",  self.model.roberta.embeddings.drop_inputs_decoder_prob)

        # Replace the output embedding layer with one without bias if in config
        if add_decoder_output_embedding_bias is False:
            print("Replacing linear output layer with one without bias!")
            self.model.lm_head.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            del self.model.lm_head.bias

        # Store some variables for convenience
        self.latent_size = latent_size
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        self.initializer_range = self.model.config.initializer_range

        # Initialise dense layers to connect the encoder and the decoder
        # via embeddings or via memory or both
        self.latent_to_decoder = LatentToDecoderNewsVAE(add_latent_via_memory=add_latent_via_memory,
                                                        add_latent_via_embeddings=add_latent_via_embeddings,
                                                        add_latent_via_cross_attention=add_latent_via_cross_attention,
                                                        add_latent_via_gating=add_latent_via_gating,
                                                        latent_size=self.latent_size,
                                                        hidden_size=self.hidden_size,
                                                        n_layers=self.n_layers,
                                                        initializer_range=self.initializer_range)

    def forward(self, latent_z, input_ids, attention_mask,
                labels=None,
                return_attention_probs=False,
                return_attention_to_latent=False,
                return_hidden_states=False,
                return_exact_match=False,
                return_predictions=False,
                return_probabilities=False,
                return_last_hidden_state=False,
                return_output_embeddings=False,
                return_logits=False,
                return_cross_entropy=False,
                return_reconstruction_loss=True,  # <-- standard for train
                return_log_probs=False,
                reduce_seq_dim_exact_match="mean",
                reduce_batch_dim_exact_match="mean",
                reduce_seq_dim_ce="sum",
                reduce_batch_dim_ce="mean",
                nucleus_sampling=False,
                top_k=0,
                top_p=0.9):
        """
        Make a parallel, forward pass (with teacher-forcing) through the decoder.

        Args:
            latent_z: Tensor [batch x latent_size]:
                Latent code in batch.
            input_ids: Tensor [batch x seq_len]
                Token input ids of the output so far or teacher-forced the whole seq.
            attention_mask: Tensor [batch x seq_len]
                Mask marking the padded tokens (0)
            labels: Tensor [batch x seq_len]

            return_attention_probs: bool
            return_attention_to_latent: bool

            return_hidden_states: bool
            return_last_hidden_state: bool

            return_predictions: bool
            return_probabilities: bool
            return_logits: bool

            return_cross_entropy: bool
            return_exact_match: bool
            reduce_seq_dim_ce: str
                How to reduce the sequence dimension for Cross Entropy: 'mean', 'sum', 'none'
            reduce_seq_dim_exact_match:
                How to reduce the sequence dimension for exact match: 'mean', 'sum', 'none'
            reduce_batch_dim_exact_match:
                How to reduce the batch dimension for exact match: 'mean', 'sum', 'none'
            reduce_batch_dim_ce:
                How to reduce the batch dimension for Cross Entropy: 'mean', 'sum', 'none'
            -> if the last two are mean, it returns exact match accuracy

            nucleus_sampling: bool
            top_k: int
            top_p: float

        Returns:
            decoder_outs: Dict[str, Union[Tensor, float]]
                Everything the decoder returns (predictions, reconstruction loss, etc.)
        """

        latent_to_decoder_output = self.latent_to_decoder(latent_z)

        # Forward the decoder
        decoder_outs = self.model(latent_to_decoder_output=latent_to_decoder_output,
                                  input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  labels=labels,
                                  gating=self.add_latent_via_gating,
                                  return_attention_probs=return_attention_probs,
                                  return_attention_to_latent=return_attention_to_latent,
                                  return_hidden_states=return_hidden_states,
                                  return_exact_match=return_exact_match,
                                  return_predictions=return_predictions,
                                  return_probabilities=return_probabilities,
                                  return_last_hidden_state=return_last_hidden_state,
                                  return_output_embeddings=return_output_embeddings,
                                  return_logits=return_logits,
                                  return_cross_entropy=return_cross_entropy,
                                  return_reconstruction_loss=return_reconstruction_loss,
                                  return_log_probs=return_log_probs,
                                  reduce_seq_dim_ce=reduce_seq_dim_ce,
                                  reduce_seq_dim_exact_match=reduce_seq_dim_exact_match,
                                  reduce_batch_dim_exact_match=reduce_batch_dim_exact_match,
                                  reduce_batch_dim_ce=reduce_batch_dim_ce,
                                  nucleus_sampling=nucleus_sampling,
                                  top_k=top_k,
                                  top_p=top_p)

        return decoder_outs

    def autoregressive_decode(self, latent_z, max_seq_len=32,

                              labels=None,

                              return_exact_match=False,
                              return_cross_entropy=False,
                              return_reconstruction_loss=False,

                              reduce_seq_dim_ce="sum",
                              reduce_batch_dim_ce="mean",
                              reduce_seq_dim_exact_match="mean",
                              reduce_batch_dim_exact_match="mean",

                              return_attention_probs=False,
                              return_attention_to_latent=False,

                              return_hidden_states=False,
                              return_last_hidden_state=False,
                              return_output_embeddings=False,

                              return_predictions=True,

                              return_probabilities=False,
                              return_logits=False,

                              return_log_probs=False,

                              nucleus_sampling=False, top_k=0, top_p=0.0,
                              device_name="cuda:0"):
        """
        This function performs auto-regressive decoding (no grads), given samples from the latent space.

        Args:
            latent_z: Dict[str, Tensor]
                Latents transformed into correct forms (embeddings / memory) for the decoder.
            max_seq_len: int:
                How many sequential forward passes to perform:
                maximum sequence length for the whole batch.
            labels: Union[None, str]
            nucleus_sampling: bool
                Whether or not to perform top_k_top_p_filtering (nucleus sampling)
                top_k_top_p_filtering: Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            top_k: int
            top_p: float
            device_name: str:
                Which device to use for decoding (default: 'cuda:0')

        Returns:
            generated_so_far: Tensor [batch, max_seq_len]
                Batch of decoded / generated token ids
        """

        assert (return_exact_match or return_cross_entropy) is (
                labels is not None), "provide labels if return_exact_match_acc or return_cross_entropy is set to True"

        latent_to_decoder = self.latent_to_decoder(latent_z)

        batch_size = latent_z.shape[0]

        # Add <s> and </s>
        eos_token_id, bos_token_id = 2, 0

        generated_so_far = torch.tensor([[bos_token_id, eos_token_id] for _ in range(batch_size)])
        generated_so_far = generated_so_far.to(device_name)

        attention_to_latent, attention_probs = [], []
        hidden_states, last_hidden_state = [], []
        logits, probabilities = [], []
        exact_match, cross_entropy = [], []
        out_w_embs = []
        log_probs = []

        # Init with nothing
        past_key_values = None

        if labels is not None:
            label_mask = labels[:, 1:].contiguous()
            # pad token is int 1
            label_mask = (label_mask != 1).float()

        predictions = []
        # Sequence length includes start and end token
        for i in range(max_seq_len - 1):
            if labels is not None:
                labels_so_far = labels[:, i:i + 2]
            else:
                labels_so_far = None

            decoder_outs = self.model(input_ids=generated_so_far,
                                      attention_mask=None,
                                      latent_to_decoder_output=latent_to_decoder,
                                      labels=labels_so_far,

                                      use_cache=True,
                                      past_key_values=past_key_values,

                                      return_cross_entropy=return_cross_entropy,
                                      return_exact_match=return_exact_match,
                                      return_reconstruction_loss=False, # need to reduce on your own in the end

                                      reduce_seq_dim_ce="none",
                                      reduce_batch_dim_ce="none",
                                      reduce_seq_dim_exact_match="none",
                                      reduce_batch_dim_exact_match="none",

                                      return_predictions=True,  # needed for input at every time step
                                      return_probabilities=return_probabilities,
                                      return_attention_to_latent=return_attention_to_latent,
                                      return_attention_probs=return_attention_probs,
                                      return_hidden_states=return_hidden_states,
                                      return_last_hidden_state=return_last_hidden_state,
                                      return_output_embeddings=return_output_embeddings,
                                      return_logits=return_logits,
                                      nucleus_sampling=nucleus_sampling,
                                      return_log_probs=return_log_probs,
                                      top_k=top_k,
                                      top_p=top_p)

            # print("decoder_outs[predictions].shape", decoder_outs["predictions"].shape)

            # Since I pass <t_1> <eos>, the cache will also consist of <eos> key-, value-pairs.
            # we do not want that in the cache, we only want things that belong to the actual past.
            past_key_values = decoder_outs["past_key_values"]
            # p_k_v is a tuple of 12 layers, with tuples of key, value matrices
            # of dimension [batch, n_heads, seq_len, head_dim]. we want to trim seq_len by 1
            past_key_values = tuple([tuple([pair[0][:, :, 1:-1, :], pair[1][:, :, 1:-1, :]]) for pair in past_key_values])
            #print("past_key_values[0][0].shape", past_key_values[0][0].shape)

            # Get the predictions of this time step
            new_preds = decoder_outs['predictions'][:, -1]
            #print("new_preds.shape", new_preds.shape)
            predictions.append(new_preds)

            if return_exact_match:
                # eos has already been cut off, so take the last element
                exact_match.append(decoder_outs["exact_match"][:, -1])

            if return_log_probs:
                # eos has already been cut off, so take the last element
                log_probs.append(decoder_outs["log_probs"][:, -1])

            if return_output_embeddings:
                out_w_embs.append(decoder_outs["output_embeddings"][:, -2, :])

            if return_cross_entropy:
                # eos has already been cut off, so take the last element
                cross_entropy.append(decoder_outs["cross_entropy"][:, -1])

            if return_probabilities:
                # eos has already been cut off, so take the last element
                probabilities.append(decoder_outs["probabilities"][:, -1, :].cpu())

            if return_logits:
                # eos has already been cut off, so take the last element
                logits.append(decoder_outs["logits"][:, -1, :].cpu())

            if return_attention_probs:
                # batch, n_heads, n_layers, seq_len_query, seq_len_val
                # we only want the query dimension of the current prediction (-1 would be </s> prediction)
                attention_probs.append(decoder_outs["attention_probs"][:, :, :, -2, :])

            if return_attention_to_latent:
                # Last one is already cut-off so the last one here is the right one
                attention_to_latent.append(decoder_outs["attention_to_latent"][:, :, :, -1].cpu())

            if return_hidden_states:
                # eos has not been cut off, so take the second to last element
                hidden_states.append(decoder_outs["hidden_states"][:, :, -2, :].cpu())

            if return_last_hidden_state:
                # eos has not been cut off, so take the second to last element
                last_hidden_state.append(decoder_outs["last_hidden_state"][:, -2, :].cpu())

            # Concat into <last prediction> </s> format for next round
            generated_so_far = torch.cat(
                (new_preds.unsqueeze(1), generated_so_far[:, -1].unsqueeze(1)), dim=1)

        outputs = {}

        if return_logits:
            outputs["logits"] = torch.stack(logits, dim=1)

        if return_log_probs:
            outputs["log_probs"] = torch.stack(log_probs, dim=1)

        if return_probabilities:
            outputs["probabilities"] = torch.stack(probabilities, dim=1)

        # what to do with attention_probs?
        if return_attention_probs:
            print("an ugly list of all (masked) attentionprobs coming back to you")
            outputs["attention_probs"] = attention_probs

        if return_attention_to_latent:
            outputs["attention_to_latent"] = torch.stack(attention_to_latent, dim=-1)

        predictions = torch.stack(predictions, dim=1)
        if return_predictions:
            # the </s> is not predicted, neither is the added </s>
            outputs["predictions"] = predictions

        if return_hidden_states:
            outputs["hidden_states"] = torch.stack(hidden_states, dim=2)

        if return_last_hidden_state:
            outputs["last_hidden_state"] = torch.stack(last_hidden_state, dim=1)

        if return_output_embeddings:
            outputs["output_embeddings"] = torch.stack(out_w_embs, dim=1)

        if return_exact_match:
            outputs["exact_match"] = torch.stack(exact_match, dim=-1)
            outputs["exact_match"] = outputs["exact_match"] * label_mask
            outputs["exact_match"] = self.model.reduce_correct(outputs["exact_match"], reduce_seq_dim_exact_match, -1,
                                                               label_mask)  # seq dim
            outputs["exact_match"] = self.model.reduce_correct(outputs["exact_match"], reduce_batch_dim_exact_match, 0,
                                                               label_mask)  # batch dim

        if return_cross_entropy:
            outputs["cross_entropy"] = torch.stack(cross_entropy, dim=-1)
            outputs["cross_entropy"] = outputs["cross_entropy"] * label_mask
            if return_reconstruction_loss:
                outputs["reconstruction_loss"] = outputs["cross_entropy"].sum(dim=-1).mean(dim=0)
            outputs["cross_entropy"] = self.model.reduce_correct(outputs["cross_entropy"], reduce_seq_dim_ce, -1,
                                                                 label_mask)  # seq dim
            outputs["cross_entropy"] = self.model.reduce_correct(outputs["cross_entropy"], reduce_batch_dim_ce, 0,
                                                                 label_mask)  # batch dim <- always mean

        return outputs


class LatentToDecoderNewsVAE(nn.Module):
    def __init__(self, add_latent_via_memory=True,
                 add_latent_via_embeddings=True,
                 add_latent_via_cross_attention=False,
                 add_latent_via_gating=False,
                 latent_size=768, hidden_size=768, n_layers=12,
                 initializer_range=0.02):
        """
        A module to connect the latents to a format that can be used by the DecoderNewsVAE.
        """

        super(LatentToDecoderNewsVAE, self).__init__()

        self.add_latent_via_memory = add_latent_via_memory
        self.add_latent_via_embeddings = add_latent_via_embeddings
        self.add_latent_via_cross_attention = add_latent_via_cross_attention
        self.add_latent_via_gating = add_latent_via_gating

        self.hidden_size = hidden_size

        # Latent via memory layer
        if self.add_latent_via_memory or self.add_latent_via_gating:
            self.latent_to_memory_projection = nn.Linear(latent_size, hidden_size * n_layers)
            self.latent_to_memory_projection.weight.data.normal_(mean=0.0, std=initializer_range)

        # Latent via embedding layer
        if self.add_latent_via_embeddings:
            self.latent_to_embedding_projection = nn.Linear(latent_size, hidden_size)
            self.latent_to_embedding_projection.weight.data.normal_(mean=0.0, std=initializer_range)

        if self.add_latent_via_cross_attention:
            self.latent_to_cross_projection = nn.Linear(latent_size, hidden_size * n_layers)
            self.latent_to_cross_projection.weight.data.normal_(mean=0.0, std=initializer_range)

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

        output = {"latent_to_memory": None,
                  "latent_to_embeddings": None,
                  "latent_to_cross": None}

        if self.add_latent_via_memory or self.add_latent_via_gating:
            latent_to_memory = self.latent_to_memory_projection(latent_z)
            # Makes tuple of equally sized tensors of (batch x 1 x hidden_size)
            latent_to_memory = torch.split(latent_to_memory.unsqueeze(1), self.hidden_size, dim=2)
            output["latent_to_memory"] = latent_to_memory

        if self.add_latent_via_cross_attention:
            latent_to_cross = self.latent_to_cross_projection(latent_z)
            # Makes tuple of equally sized tensors of (batch x 1 x hidden_size)
            latent_to_cross = torch.split(latent_to_cross.unsqueeze(1), self.hidden_size, dim=2)
            output["latent_to_cross"] = latent_to_cross

        if self.add_latent_via_embeddings:
            latent_to_embeddings = self.latent_to_embedding_projection(latent_z)
            output["latent_to_embeddings"] = latent_to_embeddings

        return output
