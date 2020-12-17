import torch


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
                               temperature=1.0, top_k=0, top_p=0.0, device_name="cuda:0"):
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

    """

    with torch.no_grad():
        _, _, latent_z, _, _, _ = vae_model.encode(input_batch['input_ids'], input_batch['attention_mask'])

        generated = vae_model.decoder.auto_regressive_decode(latent_z, tokenizer,
                                                             add_latent_via_embeddings=add_latent_via_embeddings,
                                                             add_latent_via_memory=add_latent_via_memory,
                                                             max_seq_len=max_seq_len,
                                                             nucleus_sampling=nucleus_sampling,
                                                             temperature=temperature, top_k=top_k, top_p=top_p,
                                                             device_name=device_name)

        generated_text = tokenizer_batch_decode(generated, tokenizer)

        return generated_text
