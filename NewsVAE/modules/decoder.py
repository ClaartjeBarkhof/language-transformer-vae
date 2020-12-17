import torch
from modules.decoder_roberta import VAE_Decoder_RobertaForCausalLM
import copy
from transformers import top_k_top_p_filtering
import torch.functional as F

class DecoderNewsVAE(torch.nn.Module):
    def __init__(self, gradient_checkpointing=False):
        """
        Decoder of VAE based on a RobertaForCausalLM initialised with roberta-base checkpoint.
        """
        super(DecoderNewsVAE, self).__init__()

        self.model = VAE_Decoder_RobertaForCausalLM.from_pretrained("roberta-base",
                                                                    gradient_checkpointing=gradient_checkpointing)

    def forward(self, latent_to_decoder_output, input_ids, attention_mask,
                return_predictions=False, return_exact_match_acc=True, return_attention_probs=False):
        """
        Make a (teacher-forced) forward pass through the decoder.

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
            return_attention_probs: bool
                Whether or not to return attention probabilities.
        Returns:
            decoder_outs: Dict[str, Union[Tensor, float]]
                Everything the decoder returns (predictions, reconstruction loss, etc.)
        """

        # Forward the decoder
        decoder_outs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                  latent_to_decoder_output=latent_to_decoder_output, labels=copy.copy(input_ids),
                                  return_cross_entropy=True,
                                  return_predictions=return_predictions,
                                  return_exact_match_acc=return_exact_match_acc,
                                  return_attention_probs=return_attention_probs)

        return decoder_outs

    # TODO: check whether this works. The performance of this run was very poort
    def reset_to_base_checkpoint(self, gradient_checkpointing=False,
                                 do_tie_weights=False):
        """
        This function resets the decoder (re-initialise with base checkpoint).

        Args:
            gradient_checkpointing: bool
                Whether or not to use gradient checkpointing, default: False
            do_tie_weights: bool
                Whether or not the weights between encoder and decoder are shared (for warning), default: False

        """

        print("Checking if shared_weights == False, yields {}".format(do_tie_weights == False))
        assert do_tie_weights is False, "Not resetting the decoder if the weights are shared. Aborting!"

        print(f"Resetting the decoder to roberta-base checkpoint.")
        self.model = VAE_Decoder_RobertaForCausalLM.from_pretrained("roberta-base",
                                                                    gradient_checkpointing=gradient_checkpointing)

    # TODO: NOT TESTED YET
    def log_p_x_z(self, input_ids, attention_mask, latent_z, args):
        """
        This function evaluates the likelihood (negative cross-entropy) of outputs given some latents z.
        """

        assert len(latent_z.shape) == 3, "Latent z must be of shape [batch, n_samples, latent_size]"

        batch_size, seq_len = input_ids.shape
        n_samples = latent_z.shape[1]

        losses = []

        # Loop over batch dimension, sample dimension is interpreted as batch dimension
        for i in range(batch_size):
            z = latent_z[i, :, :].squeeze(0)
            x = input_ids[i, :].expand(n_samples, seq_len)
            a = attention_mask[i, :].expand(n_samples, seq_len)

            # Forward the decoder
            decoder_outs = self.model(input_ids=x, attention_mask=a,
                                      latent_z=z, labels=copy.copy(x),
                                      add_latent_via_embeddings=args.add_latent_via_embeddings,
                                      add_latent_via_memory=args.add_latent_via_memory,
                                      return_cross_entropy=True,
                                      reduce_loss=False,
                                      return_predictions=False,
                                      return_exact_match_acc=False)

            # Reconstruction loss = cross entropy = negative log likelihood
            recon_loss = decoder_outs["cross_entropy"]
            losses.append(recon_loss)

        # Stack so the dimensions are batch_size x n_samples
        losses = torch.stack(losses)

        # Reconstruction loss = cross entropy = negative log likelihood
        # so likelihood is the negative of that
        return - losses


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

    def autoregressive_decode(self, latent_z, tokenizer, add_latent_via_embeddings=True,
                              add_latent_via_memory=True, max_seq_len=32, nucleus_sampling=False,
                              temperature=1.0, top_k=0, top_p=0.0, device_name="cuda:0"):
        """
        This function performs auto-regressive decoding (no grads), given samples from the latent space.

        Args:
            latent_z: Tensor [batch, latent_size]
                Samples from the latent space.
            tokenizer:
            add_latent_via_memory: bool
            add_latent_via_embeddings: bool
            max_seq_len: int:
                How many sequential forward passes to perform:
                maximum sequence length for the whole batch.
            nucleus_sampling: bool
                Whether or not to perform top_k_top_p_filtering (nucleus sampling)
                top_k_top_p_filtering: Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            temperature: float
            top_k: int
            top_p: float
            device_name: str:
                Which device to use for decoding (default: 'cuda:0')

        Returns:
            generated_so_far: Tensor [batch, max_seq_len]
                Batch of decoded / generated token ids
        """

        batch_size = latent_z.shape[0]

        # Add <s> and </s>
        generated_so_far = torch.tensor(
            [[tokenizer.bos_token_id, tokenizer.eos_token_id] for _ in range(batch_size)])
        generated_so_far = generated_so_far.to(device_name)

        for i in range(max_seq_len):
            # Forward the decoder
            decoder_outs = self.model(input_ids=generated_so_far, attention_mask=None,
                                      latent_z=latent_z, labels=None,
                                      add_latent_via_embeddings=add_latent_via_embeddings,
                                      add_latent_via_memory=add_latent_via_memory,
                                      return_cross_entropy=False, return_predictions=False,
                                      return_exact_match_acc=False)

            if nucleus_sampling:
                # The forward of the decoder already cuts of the prediction for the last token
                logits = decoder_outs["logits"][:, -1, :]

                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature

                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                new_preds = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                new_preds = decoder_outs['predictions'][:, -1]  # argmax predictions

            generated_so_far = torch.cat(
                (generated_so_far[:, :-1], new_preds.unsqueeze(1), generated_so_far[:, -1].unsqueeze(1)), dim=1)

        return generated_so_far
