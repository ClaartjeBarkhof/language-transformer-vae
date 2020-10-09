# IMPORTS
import torch
import numpy as np
import os
from argparse import Namespace
import torch.nn.functional as F
import pickle
import copy
import random

import sys
sys.path.append('../code')
sys.path.append('../code/examples/big_ae')
sys.path.append('../code/pytorch_transformers')

from relevant_optimus_code.configuration_bert import BertConfig
from relevant_optimus_code.configuration_gpt2 import GPT2Config
from relevant_optimus_code.tokenization_bert import BertTokenizer
from relevant_optimus_code.tokenization_gpt2 import GPT2Tokenizer
from relevant_optimus_code.modeling_bert import BertForLatentConnector
from relevant_optimus_code.modeling_gpt2 import GPT2ForLatentConnector
from relevant_optimus_code.run_latent_generation import add_special_tokens_to_decoder
from relevant_optimus_code.vae import VAE

# -----------------------------------------------------------------
# GLOBALS

# Model classes & types
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer)}

ENCODER_MODEL_TYPE = 'bert'
ENCODER_MODEL_NAME = 'bert-base-cased'
DECODER_MODEL_TYPE = 'gpt2'
DECODER_MODEL_NAME = 'gpt2'

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# Random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Model settings
LATENT_SIZES = {'snli': 768, 'wikipedia': 32}
DO_LOWERCASE = False

# Paths to models
CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PREFIX_PATH = CURRENT_DIR_PATH + '/../'  # one up is the Optimus folder
SNLI_MODEL_BASE_PATH = 'output/LM/Snli/768/philly_vae_snli_b{}_d5_r00.5_ra0.25_length_weighted/'
WIKIPEDIA_MODEL_BASE_PATH = 'output/pretrain/philly_rr3_vc4_g8_base_vae_wikipedia_pretraining_beta_schedule_beta{' \
                            '}_d1.0_ro0.5_ra0.25_32_v2/'

# Data
SENTENCES_FILE = PREFIX_PATH + "claartje/snli_sentences.pickle"
MAX_SENTENCES = True
MAX_N_SENTENCES = 10000

# Batch size
BATCH_SIZE = 64
N_BATCHES = -1  # -1 for all

# Results
RESULT_DIR = 'evaluation-results'

# Whether or not to use cached keys and value in the auto-regressive generation
USE_CACHE_AUTOREGRESS = True


# -----------------------------------------------------------------

def get_model_path(snli_wikipedia, beta):
    """
    Returns a specific model path, given the snli_wikipedia, beta setting.
    """
    assert snli_wikipedia in ['snli', 'wikipedia']
    if snli_wikipedia == 'snli':
        assert beta in [0.0, 0.5, 1.0]
        global_step = 31250
        cp_dir_path = SNLI_MODEL_BASE_PATH.format(beta)
    elif snli_wikipedia == 'wikipedia':
        assert beta in [0.0, 0.5]
        global_step = 508523
        cp_dir_path = WIKIPEDIA_MODEL_BASE_PATH.format(beta, global_step)

    encoder_path = PREFIX_PATH + cp_dir_path + 'checkpoint-{}/'.format(global_step) + \
                   'checkpoint-encoder-{}'.format(global_step)
    decoder_path = PREFIX_PATH + cp_dir_path + 'checkpoint-{}/'.format(global_step) + \
                   'checkpoint-decoder-{}'.format(global_step)
    full_model_path = PREFIX_PATH + cp_dir_path + 'checkpoint-{}/'.format(global_step) + \
                      'checkpoint-full-{}'.format(global_step)

    return encoder_path, decoder_path, full_model_path


def get_all_model_paths():
    """
    Returns a dict to all model paths given all possible settings: snli-wiki and beta values.
    """
    MODEL_PATHS = {}
    for snli_wikipedia in ['snli', 'wikipedia']:
        for beta in [0.0, 0.5, 1.0]:
            if (snli_wikipedia == 'wikipedia') and (beta == 1.0):
                continue
            if snli_wikipedia not in MODEL_PATHS:
                MODEL_PATHS[snli_wikipedia] = {}
            enc, dec, full = get_model_path(snli_wikipedia, beta)
            MODEL_PATHS[snli_wikipedia][beta] = {'encoder_path': enc, 'decoder_path': dec, 'full_path': full}
    return MODEL_PATHS


def get_model_tokenizer_encoder(path, snli_wikipedia):
    """
    Given a snli-wikipedia settings, return model encoder and encoder tokeniser.
    """
    # Load a trained Encoder model and vocabulary that you have fine-tuned
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES[ENCODER_MODEL_TYPE]
    model_encoder = encoder_model_class.from_pretrained(path, latent_size=LATENT_SIZES[snli_wikipedia])
    tokenizer_encoder = encoder_tokenizer_class.from_pretrained(ENCODER_MODEL_NAME, do_lower_case=DO_LOWERCASE)
    return model_encoder, tokenizer_encoder


def get_model_tokenizer_decoder(path, snli_wikipedia):
    """
    Given a snli-wikipedia settings, return model decoder and decoder tokeniser.
    """
    # Load a trained Decoder model and vocabulary that you have fine-tuned
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[DECODER_MODEL_TYPE]
    model_decoder = decoder_model_class.from_pretrained(path, latent_size=LATENT_SIZES[snli_wikipedia])
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained(DECODER_MODEL_NAME, do_lower_case=DO_LOWERCASE)
    model_decoder, tokenizer_decoder = add_special_tokens_to_decoder(model_decoder, tokenizer_decoder)
    return model_decoder, tokenizer_decoder


def get_model_vae(path, model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, snli_wikipedia):
    """
    Given a model path, encoder, decoder, tokenizer encoder, tokenizer decoder and snli-wiki setting,
    return a full VAE model.
    """
    checkpoint_full = torch.load(os.path.join(path, 'training.bin'),
                                 map_location=torch.device(DEVICE))
    args = {'latent_size': LATENT_SIZES[snli_wikipedia], 'device': DEVICE}
    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, Namespace(**args))
    model_vae.load_state_dict(checkpoint_full['model_state_dict'])
    model_vae.eval()
    return model_vae


def load_model(model_paths, snli_wikipedia, beta):
    """
    Loads VAE models with all settings (snli, wikipedia) & all beta values.
    """

    paths = model_paths[snli_wikipedia][beta]

    model_encoder, tokenizer_encoder = get_model_tokenizer_encoder(paths['encoder_path'], snli_wikipedia)
    model_decoder, tokenizer_decoder = get_model_tokenizer_decoder(paths['decoder_path'], snli_wikipedia)

    vae_model = get_model_vae(paths['full_path'], model_encoder, model_decoder,
                              tokenizer_encoder, tokenizer_decoder, snli_wikipedia)

    return vae_model


def tokenise_pad_sentences(sentences, tokenizer_encoder, tokenizer_decoder):
    """
    Given a list of string sentences, tokenise those and return as as list of
    tensors of token_ids and a 2D tensor that includes the padded
    tokenised batch of sentences.
    """

    tok_sent_batch_enc = []
    tok_sent_batch_dec = []

    tok_sent_lens_enc = []
    tok_sent_lens_dec = []

    for i, s in enumerate(sentences):
        # Encoder BERT tokenise
        tok_enc = [101] + tokenizer_encoder.encode(s) + [102]  # add beginning of sentence and end of sentence
        tok_sent_lens_enc.append(len(tok_enc))
        tok_sent_batch_enc.append(tok_enc)

        # Decoder GPT2 tokenise
        tok_dec = [50258] + tokenizer_decoder.encode(s) + [50259]  # '<BOS>': 50258, '<EOS>': 50259
        tok_sent_lens_dec.append(len(tok_dec))
        tok_sent_batch_dec.append(tok_dec)

    pad_tok_sent_enc = []
    max_len_enc = max(tok_sent_lens_enc)
    for ts in tok_sent_batch_enc:
        padding = [tokenizer_encoder.vocab['[PAD]']] * (max_len_enc - len(ts))  # PAD should be 0
        pad_tok_sent_enc.append(ts + padding)

    pad_tok_sent_dec = []
    max_len_dec = max(tok_sent_lens_dec)
    for ts in tok_sent_batch_dec:
        padding = [tokenizer_decoder.added_tokens_encoder['<PAD>']] * (max_len_dec - len(ts))  # PAD should be 50257
        pad_tok_sent_dec.append(ts + padding)

    pad_tok_sent_dec = torch.stack([torch.tensor(x) for x in pad_tok_sent_dec])
    pad_tok_sent_enc = torch.stack([torch.tensor(x) for x in pad_tok_sent_enc])

    pad_tok_sent_dec = pad_tok_sent_dec.to(DEVICE)
    pad_tok_sent_enc = pad_tok_sent_enc.to(DEVICE)

    # print("Padded decoder GPT2 input/label sequence batch shape:", pad_tok_sent_dec.shape)
    # print("Padded encoder BERT input sequence batch shape:", pad_tok_sent_enc.shape)

    return pad_tok_sent_dec, pad_tok_sent_enc, tok_sent_batch_dec, tok_sent_batch_enc


def logits_to_prediction(logits, sample=True):
    """
    Given logits, make predictions by sampling or taking the maximum.
    Sampling is recommended to decrease the risk for loops.
    """

    probs = F.softmax(logits, dim=-1)

    if sample:
        probs_reshape = probs.view(-1, probs.shape[-1])
        preds = torch.multinomial(probs_reshape, num_samples=1).view(probs.shape[:-1])
    else:
        preds = torch.argmax(probs, dim=-1)

    return preds


def teacher_force_decode(VAE_model, latent_z, labels_GPT2, train=False):
    """
    Do a forward pass of the decoder in a teacher forced manner,
    where the ground truth labels condition every step of the sequence.
    It returns the sampled predictions (token ids) and the logits.
    """

    with torch.set_grad_enabled(train):
        # Decode
        outputs = VAE_model.decoder(input_ids=labels_GPT2, past=latent_z)
        lm_logits = outputs[0]  # element 1 is transformer_outputs[1:], don't need that

        # Make predictions
        preds = logits_to_prediction(lm_logits, sample=True)

        return preds, lm_logits


def autoregressive_decode(VAE_model, latent_z, labels_GPT2,
                          train=False, use_cache=True):
    """
    Do an auto-regressive forward pass in the decoder, given a latent vector z.

    :param VAE_model:
    :param latent_z:       latent vector passed from the encoder
    :param labels_GPT2:
    :param train:          whether or not in train mode (grads enabled or not)
    :param use_cache:      whether or not to save key, value vectors from previous
                           forward passes, to be speed up the forward and reduce cuda memory.
    :return:               - predictions for the whole sequence (token ids)
                           - logits for the whole sequences

    """
    max_sent_len = labels_GPT2.shape[1]

    # The first column of labels_GPT2 are <BOS> tokens, used to start decoding (first conditional)
    generated_so_far = labels_GPT2[:, 0].unsqueeze(1)
    prev_token = copy.deepcopy(generated_so_far)

    all_logits = []

    with torch.set_grad_enabled(train):
        past = None

        # Generate for the whole batch token per token (auto-regressively)
        for i in range(max_sent_len):

            # If no cache is used, the full sequence outputted so far is passed through the model
            if not use_cache:
                outputs = VAE_model.decoder(generated_so_far, past=latent_z)

            # Otherwise only the last prediction
            else:
                # First forward, no cache yet, just latents
                if not past:
                    outputs = VAE_model.decoder(prev_token, past=latent_z, reshape_present=True)
                # Pass the cached 'past'
                else:
                    outputs = VAE_model.decoder(prev_token, past=latent_z,
                                                actual_past=past, reshape_present=True)

                # Outputs is a tuple (logits, present)
                past = outputs[1]

            # Outputs is a tuple (logits, present)
            logits_lm = outputs[0]
            all_logits.append(logits_lm)

            # logits_lm is of dim B x N input_tokens x vocab size
            # we want the output for the last hidden state
            next_token_logits = logits_lm[:, -1, :]

            # No top_k, top_p filtering,
            # just apply softmax and sample woth that distribution
            next_token_probs = F.softmax(next_token_logits, dim=1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            # The generated output can be concatted to the previous ouput
            generated_so_far = torch.cat((generated_so_far, next_token), dim=1)

            # Current predictions becomes conditioning in next round (if cache is used)
            prev_token = next_token

        # Get rid of the start token
        predictions = generated_so_far[:, 1:]

        # Concat all logits along the sequence dimension
        all_logits = torch.cat(all_logits, dim=1)

        return predictions, all_logits


def encode_reconstruct_auto_and_tf(VAE_model, inputs_BERT, labels_GPT2, use_cache=True):
    """
    Given a VAE model, encode and decode a batch of tokenised sentences in two ways:
        - Autoregressively
        - Teacher forced

    Return the logits and predictions of both methods.
    """

    # Mask tokens that are 0 (PAD in BERT)
    attention_mask = (inputs_BERT > 0).float()

    # Make a folder to store results
    os.makedirs(RESULT_DIR, exist_ok=True)

    # ENCODE
    _, pooled_hidden_fea = VAE_model.encoder(inputs_BERT, attention_mask)

    # Connect ENCODE - DECODE
    # Now implemented as taking the mean of the std, log var output,
    # but can also use the connect function of the VAE model
    # latent_z, loss_kl = self.connect(pooled_hidden_fea); latent_z = latent_z.squeeze(1)
    mean, _ = VAE_model.encoder.linear(pooled_hidden_fea).chunk(2, -1)
    latent_z = mean.squeeze(1)

    # DECODE AUTO REGRESSIVE
    preds_auto, logits_auto = autoregressive_decode(VAE_model, latent_z, labels_GPT2, use_cache=use_cache)

    # DECODE TEACHER-FORCE
    preds_tf, logits_tf = teacher_force_decode(VAE_model, latent_z, labels_GPT2)

    return preds_auto, logits_auto, preds_tf, logits_tf


def get_masked_accuracy(preds, targets, mask):
    """
    Returns the masked accuracy given predictions for a batch of sequence predictions
    and a mask. The accuracy (or fraction correct) is divided by the sequence length
    and is thus proportional to the length.
    """

    correct = (preds == targets).float() * mask
    correct_sum = correct.sum(dim=1)
    frac_correct = correct_sum / mask.sum(dim=1)

    return frac_correct


def get_cross_entropy_perplexity(logits, targets, mask):
    """
    Returns the cross entropy and perplexity given a batch of logits, targets and a mask.
    The cross entropy is calculated and masked and then summed and divided by sequence lengths.
    The perplexity is taken as the exponent of the cross entropy (averaged over the sequence).
    """

    # Merge seq. dimension into batch dimension
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1, 1)

    # Calc cross entropy
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    ce_losses_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    ce_loss = ce_losses_flat.view(*targets.size())
    ce_loss = ce_loss * mask.float()

    # Sum cross entropy over the sequence
    ce_loss_sum = ce_loss.sum(dim=1)
    ce_loss_prop = ce_loss_sum / mask.sum(dim=1)

    # Perplexity as exp of cross entropy
    # ppl_prop = (torch.exp(ce_loss) * mask).sum(dim=1) / mask.sum(dim=1)
    ppl_prop = torch.exp(ce_loss_prop)

    return ce_loss_prop, ppl_prop


def load_sentences(shuffle=True):
    """
    Load sentences from a pickle defined in the global SENTENCES_FILE.
    Respect the max length, shuffle if True and lowercase all the sentence strings.
    """

    sentences = pickle.load(open(SENTENCES_FILE, 'rb'))

    # Shuffle for lengths to be mixed, the file has them sorted on length
    if shuffle:
        random.shuffle(sentences)

    # Skip a part of the sentences if a maximum is provided.
    if MAX_N_SENTENCES != -1:
        sentences = sentences[:MAX_N_SENTENCES]

    # Lowercase all strings for equal comparison between models
    sentences = [s.lower() for s in sentences]

    return sentences


if __name__ == "__main__":
    model_paths = get_all_model_paths()

    tokenizer_encoder = MODEL_CLASSES[ENCODER_MODEL_TYPE][2].from_pretrained(ENCODER_MODEL_NAME,
                                                                             do_lower_case=DO_LOWERCASE)
    _, tokenizer_decoder = get_model_tokenizer_decoder(model_paths['snli'][1.0]['decoder_path'], 'snli')

    sentences = load_sentences()

    n_batches = int(np.ceil(len(sentences) / BATCH_SIZE))
    N_BATCHES = n_batches if (N_BATCHES == - 1) else N_BATCHES

    for snli_wikipedia in ['snli', 'wikipedia']:
        for beta, VAE_model in model_paths[snli_wikipedia].items():

            all_labels = []
            all_preds_auto, all_preds_tf = [], []
            all_accs_auto, all_accs_tf = [], []
            all_ce_losses_auto, all_ce_losses_tf = [], []
            all_prop_ppls_auto, all_prop_ppls_tf = [], []

            setting_dir = RESULT_DIR + "/{}-{}/".format(snli_wikipedia, beta)
            os.makedirs(setting_dir, exist_ok=True)

            print("-" * 30)
            print("Configuration: {} - beta: {}".format(snli_wikipedia, beta))

            VAE_model = load_model(model_paths, snli_wikipedia, beta)
            VAE_model = VAE_model.to(DEVICE)

            # Tokenizers should be the same for all models? Not sure actually
            tokenizer_encoder = VAE_model.tokenizer_encoder
            tokenizer_decoder = VAE_model.tokenizer_decoder

            for batch_i, idx in enumerate(range(0, len(sentences), BATCH_SIZE)):
                print("Batch {}".format(batch_i), end='\r')

                sentence_batch = sentences[idx:idx + BATCH_SIZE]

                batch_labels_gpt2, batch_inputs_bert, \
                toks_gpt2, toks_bert = tokenise_pad_sentences(sentence_batch,
                                                              tokenizer_encoder,
                                                              tokenizer_decoder)

                batch_inputs_bert = batch_inputs_bert.to(DEVICE)
                batch_labels_gpt2 = batch_labels_gpt2.to(DEVICE)

                preds_auto, logits_auto, preds_tf, logits_tf = encode_reconstruct_auto_and_tf(VAE_model,
                                                                                              batch_inputs_bert,
                                                                                              batch_labels_gpt2,
                                                                                              use_cache=USE_CACHE_AUTOREGRESS)

                logits_auto, logits_tf = logits_auto[:, :-1, :].cpu(), logits_tf[:, :-1, :].cpu()

                batch_labels_gpt2 = batch_labels_gpt2[:, 1:].cpu()  # cut-off BOS token
                preds_auto, preds_tf = preds_auto[:, :-1].cpu(), preds_tf[:,:-1].cpu()  # cut-off last token to make same size
                mask = (batch_labels_gpt2 < 50257).float().cpu()  # where is not the padding or EOS token, put 1

                accs_auto = get_masked_accuracy(preds_auto, batch_labels_gpt2, mask)
                accs_tf = get_masked_accuracy(preds_tf, batch_labels_gpt2, mask)

                ce_loss_prop_auto, ppl_prop_auto = get_cross_entropy_perplexity(logits_auto, batch_labels_gpt2, mask)
                ce_loss_prop_tf, ppl_prop_tf = get_cross_entropy_perplexity(logits_tf, batch_labels_gpt2, mask)

                if batch_i == (N_BATCHES - 1):
                    break

                all_labels.append(batch_labels_gpt2)
                all_preds_auto.append(preds_auto)
                all_preds_tf.append(preds_tf)
                all_accs_auto.append(accs_auto)
                all_accs_tf.append(accs_tf)
                all_ce_losses_auto.append(ce_loss_prop_auto)
                all_ce_losses_tf.append(ce_loss_prop_tf)
                all_prop_ppls_auto.append(ppl_prop_auto)
                all_prop_ppls_tf.append(ppl_prop_tf)

            settings_result = {
                'all_sentences': sentences,
                'all_labels': all_labels,
                'all_preds_auto': all_preds_auto,
                'all_preds_tf' : all_preds_tf,
                'all_accs_auto': torch.cat(all_accs_auto),
                'all_accs_tf': torch.cat(all_accs_tf),
                'all_ce_losses_auto': torch.cat(all_ce_losses_auto),
                'all_ce_losses_tf': torch.cat(all_ce_losses_tf),
                'all_prop_ppls_auto': torch.cat(all_prop_ppls_auto),
                'all_prop_ppls_tf': torch.cat(all_prop_ppls_tf)
            }

            # Save results of setting
            pickle.dump(settings_result, open(setting_dir + 'results.pickle', 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)