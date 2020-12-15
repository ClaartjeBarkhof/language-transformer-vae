import torch.nn as nn
from utils_external import tie_weights
import torch
from decoder import DecoderNewsVAE
from encoder import EncoderNewsVAE

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
        self.n_layers = self.encoder.model.config.num_hidden_layers
        self.hidden_size = self.encoder.model.config.hidden_size
        self.initializer_range = self.encoder.model.config.initializer_range

        # To connect the encoder and the decoder through the latent space
        self.latent_to_decoder = LatentToDecoderNewsVAE(add_latent_via_memory=add_latent_via_memory,
                                                        add_latent_via_embeddings=add_latent_via_embeddings,
                                                        latent_size=self.latent_size, hidden_size=self.hidden_size,
                                                        n_layers=self.n_layers,
                                                        initializer_range=self.initializer_range)

        # Weight tying / sharing
        if do_tie_weights:
            base_model_prefix = self.decoder.model.base_model_prefix
            tie_weights(self.encoder.model, self.decoder.model._modules[base_model_prefix], base_model_prefix)

    def forward(self, input_ids, attention_mask, beta,
                return_predictions=False,
                return_attention_probs=False,
                return_exact_match_acc=True,
                objective='beta-vae'):
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
                                    return_predictions=return_predictions,
                                    return_exact_match_acc=return_exact_match_acc,
                                    return_attention_probs=return_attention_probs)

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
            losses['attention_probs'] = decoder_outs["attention_probs"]

        if return_exact_match_acc:
            losses["exact_match_acc"] = decoder_outs["exact_match_acc"].item()

        return losses


class LatentToDecoderNewsVAE(nn.Module):
    def __init__(self, add_latent_via_memory=True,
                 add_latent_via_embeddings=True,
                 latent_size=768, hidden_size=768, n_layers=12,
                 initializer_range=0.02):
        """
        A module to connect the latents to a format that can be used by the DecoderNewsVAE.
        """

        super(LatentToDecoderNewsVAE, self).__init__()

        self.add_latent_via_memory = add_latent_via_memory
        self.add_latent_via_embeddings = add_latent_via_embeddings

        self.hidden_size = hidden_size

        # Latent via memory layer
        if self.add_latent_via_memory:
            self.latent_to_memory_projection = nn.Linear(latent_size, hidden_size * n_layers)
            self.latent_to_memory_projection.weight.data.normal_(mean=0.0, std=initializer_range)

        # Latent via embedding layer
        if self.add_latent_via_embeddings:
            self.latent_to_embedding_projection = nn.Linear(latent_size, hidden_size)
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

        output = {"latent_to_memory": None,
                  "latent_to_embeddings": None}

        if self.add_latent_via_memory:
            latent_to_memory = self.latent_to_memory_projection(latent_z)
            # Makes tuple of equally sized tensors of (batch x 1 x hidden_size)
            latent_to_memory = torch.split(latent_to_memory.unsqueeze(1), self.hidden_size, dim=2)
            output["latent_to_memory"] = latent_to_memory

        if self.add_latent_via_embeddings:
            latent_to_embeddings = self.latent_to_embedding_projection(latent_z)
            output["latent_to_embeddings"] = latent_to_embeddings



if __name__=="__main__":
    latent_size = 768
    embedding_mechanism = True
    memory_mechanism = True
    tied_weights = True

    dec_model = DecoderNewsVAE(gradient_checkpointing=False)
    enc_model = EncoderNewsVAE(gradient_checkpointing=False, latent_size=latent_size)

    vae_model = NewsVAE(enc_model, dec_model, latent_size=latent_size,
                        add_latent_via_memory=memory_mechanism,
                        add_latent_via_embeddings=embedding_mechanism,
                        do_tie_weights=tied_weights)

    print("done loading VAE!")
