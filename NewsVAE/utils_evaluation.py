import torch
from datasets import load_from_disk
from functools import partial
from transformers import RobertaTokenizerFast
from torch.utils.data import DataLoader
import numpy as np

# Functions in file:

# valid_dataset_loader_tokenizer
# sample_text_autoregressive
# tokenizer_batch_decode
# reconstruct_autoregressive


def valid_dataset_loader_tokenizer(batch_size=32, num_workers=4,
                                   dataset_path="/home/cbarkhof/code-thesis/NewsVAE/"
                                                "NewsData/22DEC-cnn_dailymail-roberta-seqlen64/validation"):

    def collate_fn(encoded_samples, tokenizer):
        """
        A function that assembles a batch. This is where padding is done, since it depends on
        the maximum sequence length in the batch.

        :param examples: list of truncated, tokenised & encoded sequences
        :return: padded_batch (batch x max_seq_len)
        """

        # Combine the tensors into a padded batch
        padded_batch = tokenizer.pad(encoded_samples, return_tensors='pt')

        return padded_batch

    # VALIDATION DATA
    valid_dataset = load_from_disk(dataset_path)

    # TOKENIZER
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    # TEST DATA LOADER
    valid_loader = DataLoader(valid_dataset, collate_fn=partial(collate_fn, tokenizer=tokenizer),
                              batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    print(f"Number of valid samples: {len(valid_dataset)}, "
          f"number of batches of size {batch_size}: "
          f"{int(np.floor(len(valid_dataset) / batch_size))}")

    return valid_dataset, tokenizer, valid_loader


def sample_text_autoregressive(vae_model, tokenizer, add_latent_via_embeddings=True,
                               add_latent_via_memory=True, n_samples=8, max_seq_len=64, nucleus_sampling=True,
                               temperature=1.0, top_k=0.0, top_p=0.9):
    """
    Given a model and tokenizer this function samples from the prior and decodes
    the samples autoregressively into text samples.

    Args:
        vae_model: nn.Module
        tokenizer:
        add_latent_via_memory: bool
        add_latent_via_embeddings: bool
        n_samples: int
        max_seq_len: int
        nucleus_sampling: bool
            Whether or not to sample with nucleus_sampling (top_k_top_p_filtering)
        temperature: float
        top_k: int
        top_p: float

    Returns:
        generated_text: List[str]
            list strings of decoded samples
    """
    print("-> Sampling with gradients turned OFF.")
    with torch.no_grad():
        latent_z = vae_model.decoder.sample_from_prior(vae_model.latent_size, n_samples=n_samples)
        generated = vae_model.decoder.autoregressive_decode(vae_model, latent_z, tokenizer,
                                                            add_latent_via_embeddings=add_latent_via_embeddings,
                                                            add_latent_via_memory=add_latent_via_memory,
                                                            max_seq_len=max_seq_len, nucleus_sampling=nucleus_sampling,
                                                            temperature=temperature, top_k=top_k, top_p=top_p)
        generated_text = tokenizer_batch_decode(generated, tokenizer)

        return generated_text


def tokenizer_batch_decode(batch_of_samples, tokenizer):
    return [tokenizer.decode(a.tolist()) for a in batch_of_samples]


def reconstruct_autoregressive(vae_model, input_batch, tokenizer, add_latent_via_embeddings=True,
                               add_latent_via_memory=True, max_seq_len=32, nucleus_sampling=False,
                               temperature=1.0, top_k=0, top_p=0.0, device_name="cuda:0",
                               return_attention_to_latent=False):
    """
    Reconstruct (forward through VAE with auto-regressive decoding) autoregressively
    without gradients (for evaluation purposes only).

    Args:
        vae_model: nn.Module
        input_batch: Tensor [batch, seq_len]
        tokenizer:
        add_latent_via_memory: bool
        add_latent_via_embeddings: bool
        max_seq_len: int
            Maximum length for auto-regressive decoding.
        nucleus_sampling: bool
            Whether or not to sample with nucleus_sampling (top_k_top_p_filtering)
        temperature: float
        top_k: int
        top_p: float
        device_name: str

    Returns:
        generated_text: List[str]
            List of reconstructed strings consisting of max <max_seq_len> tokens.

        generated: Tensor [batch, max_seq_len]
            Reconstructed inputs (ids)
    """

    with torch.no_grad():
        # return mu, logvar, latent_z, kl_loss, hinge_kl_loss, mmd_loss
        _, _, latent_z, _, _, _ = vae_model.encoder.encode(input_batch['input_ids'], input_batch['attention_mask'])

        latent_to_decoder = vae_model.latent_to_decoder(latent_z)

        autoreg_decode = vae_model.decoder.autoregressive_decode(latent_to_decoder, tokenizer,
                                                                 max_seq_len=max_seq_len,
                                                                 nucleus_sampling=nucleus_sampling,
                                                                 temperature=temperature, top_k=top_k, top_p=top_p,
                                                                 device_name=device_name,
                                                                 return_attention_to_latent=return_attention_to_latent)

        if return_attention_to_latent:
            generated, atts = autoreg_decode
        else:
            generated = autoreg_decode

        generated_text = tokenizer_batch_decode(generated, tokenizer)

        if return_attention_to_latent:
            return generated_text, generated, atts
        else:
            return generated_text, generated
