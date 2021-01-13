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


def ngrams_to_positions(ngrams):
    ngram_to_pos = {}
    for i, ngram in enumerate(ngrams):
        if ngram in ngram_to_pos:
            ngram_to_pos[ngram].append(i)
        else:
            ngram_to_pos[ngram] = [i]
    return ngram_to_pos


def find_ngrams(input_list, n):
    ngrams = list(zip(*[input_list[i:] for i in range(n)]))
    ngrams = [frozenset(ng) for ng in ngrams]
    return ngrams


def get_matching_ngram_stats(pred_list, target_list, n):

    pred_ngrams = find_ngrams(pred_list, n)
    target_ngrams = find_ngrams(target_list, n)
    target_ngrams_to_pos = ngrams_to_positions(target_ngrams)

    matching_pos = []
    for ngram in pred_ngrams:
        if ngram in target_ngrams_to_pos:
            matching_pos.extend(target_ngrams_to_pos[ngram])

    return matching_pos


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