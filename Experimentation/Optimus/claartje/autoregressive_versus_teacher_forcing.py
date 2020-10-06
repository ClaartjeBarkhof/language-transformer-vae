# IMPORTS
import torch
import numpy as np
import os
import csv
from argparse import Namespace
import torch.nn.functional as F
from datetime import datetime
import pickle

import sys
sys.path.append('../code')
sys.path.append('../code/examples/big_ae')

from relevant_optimus_code.configuration_bert import BertConfig
from relevant_optimus_code.configuration_gpt2 import GPT2Config
from relevant_optimus_code.tokenization_bert import BertTokenizer
from relevant_optimus_code.tokenization_gpt2 import GPT2Tokenizer
from relevant_optimus_code.modeling_bert import BertForLatentConnector
from relevant_optimus_code.modeling_gpt2 import GPT2ForLatentConnector
# Other functions that might be relevant: interpolate, latent_code_from_text, text_from_latent_code, top_k_top_p_filtering
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

# Random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Model settings
LATENT_SIZES = {'snli': 768, 'wikipedia': 32}
DO_LOWERCASE = False

# Paths to models
PREFIX_PATH = '/Users/claartje/Dropbox (Persoonlijk)/Studie/Master AI/Thesis/code-thesis/Experimentation/Optimus/'
SNLI_MODEL_BASE_PATH = 'output/LM/Snli/768/philly_vae_snli_b{}_d5_r00.5_ra0.25_length_weighted/'
WIKIPEDIA_MODEL_BASE_PATH = 'output/pretrain/philly_rr3_vc4_g8_base_vae_wikipedia_pretraining_beta_schedule_beta{' \
                            '}_d1.0_ro0.5_ra0.25_32_v2/'

# Data
SENTENCES_FILE = "sample_text.txt"
MAX_SENTENCES = True
MAX_N_SENTENCES = 200

# Batch size
BATCH_SIZE = 20
N_BATCHES = 2  # -1 for all

# Decode sequence length
MAX_DEC_SEQ_LEN = 10  # -1 for max_len

# Results
RESULT_DIR = 'evaluation-results'

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
    Returns a dict to all model paths.
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
    # Load a trained Encoder model and vocabulary that you have fine-tuned
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES[ENCODER_MODEL_TYPE]
    model_encoder = encoder_model_class.from_pretrained(path, latent_size=LATENT_SIZES[snli_wikipedia])
    tokenizer_encoder = encoder_tokenizer_class.from_pretrained(ENCODER_MODEL_NAME, do_lower_case=DO_LOWERCASE)
    return model_encoder, tokenizer_encoder


def get_model_tokenizer_decoder(path, snli_wikipedia):
    # Load a trained Decoder model and vocabulary that you have fine-tuned
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[DECODER_MODEL_TYPE]
    model_decoder = decoder_model_class.from_pretrained(path, latent_size=LATENT_SIZES[snli_wikipedia])
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained(DECODER_MODEL_NAME, do_lower_case=DO_LOWERCASE)
    model_decoder, tokenizer_decoder = add_special_tokens_to_decoder(model_decoder, tokenizer_decoder)
    return model_decoder, tokenizer_decoder


def get_model_vae(path, model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, snli_wikipedia):
    checkpoint_full = torch.load(os.path.join(path, 'training.bin'),
                                 map_location=torch.device(DEVICE))
    args = {'latent_size': LATENT_SIZES[snli_wikipedia], 'device': DEVICE}
    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, Namespace(**args))
    model_vae.load_state_dict(checkpoint_full['model_state_dict'])
    model_vae.eval()
    return model_vae


def load_all_models(model_paths):
    """
    Loads VAE models with all settings (snli, wikipedia) & all beta values.
    """

    vae_models = {'snli': {}, 'wikipedia': {}}

    for snli_wikipedia in ['wikipedia', 'snli']:
        for beta, paths in model_paths[snli_wikipedia].items():

            model_encoder, tokenizer_encoder = get_model_tokenizer_encoder(paths['encoder_path'], snli_wikipedia)
            model_decoder, tokenizer_decoder = get_model_tokenizer_decoder(paths['decoder_path'], snli_wikipedia)

            vae_models[snli_wikipedia][beta] = get_model_vae(paths['full_path'], model_encoder, model_decoder,
                                                             tokenizer_encoder, tokenizer_decoder, snli_wikipedia)
    return vae_models


def load_sentences(sentences_txt_file, max_N_sentences=False, N_sentences=200):
    sentences = []
    with open(sentences_txt_file, 'r') as fd:
        reader = csv.reader(fd, delimiter='\t')
        for i, row in enumerate(reader):
            sentences.append(row[1])
            sentences.append(row[2])
            if max_N_sentences:
                if (len(sentences) >= N_sentences):
                    break
    return sentences


def tokenise_pad_sentences(sentences, tokenizer_encoder, tokenizer_decoder):
    sent_batch = sentences  # [:20]
    tok_sent_batch_enc = []
    tok_sent_batch_dec = []

    sent_lens = []
    tok_sent_lens_enc = []
    tok_sent_lens_dec = []

    for i, s in enumerate(sent_batch):
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

    print("Padded decoder GPT2 input/laebl sequence batch shape:", pad_tok_sent_dec.shape)
    print("Padded encoder BERT input sequence batch shape:", pad_tok_sent_enc.shape)

    return pad_tok_sent_dec, pad_tok_sent_enc #, sent_lens, tok_sent_lens_enc, tok_sent_lens_dec, tok_sent_batch_enc, tok_sent_batch_dec


def teacher_force_decode(VAE_model, latent_z, labels_GPT2, train=False):
    with torch.set_grad_enabled(train):
        # Calculate a mask of outputs that should be taken into account (shifted padding)
        reconstruction_mask = (labels_GPT2 != 50257).float()
        shift_reconstruction_mask = reconstruction_mask[:, 1:].contiguous()

        # Decode
        outputs = VAE_model.decoder(input_ids=labels_GPT2, past=latent_z, labels=labels_GPT2,
                                    label_ignore=VAE_model.pad_token_id)
        lm_logits = outputs[1]  # element 2 is transformer_outputs[1:], don't need that

        # Align labels and outputs
        shift_labels = labels_GPT2[:, 1:].contiguous()  # Shift labels to left, so that align with outputs
        shift_lm_logits = lm_logits[:, :-1, :].contiguous()  # Skip last output, so that aligns with labels

        # Functions
        softmax_fn = torch.nn.Softmax(dim=2)
        ce_fn = torch.nn.CrossEntropyLoss(ignore_index=VAE_model.pad_token_id, reduction='none')

        # Convert output to probability to token id prediction
        probs = softmax_fn(shift_lm_logits)
        pred = torch.argmax(probs, dim=2)  # take max over the vocab dimension (2)

        # CE loss expects B x Vocab, so force sequence dim into B dim.
        probs_resize = shift_lm_logits.view(-1, shift_lm_logits.size(-1))
        ce_loss_resize = ce_fn(probs_resize, shift_labels.view(-1))
        ce_loss = torch.sum(ce_loss_resize.view(-1, shift_labels.shape[-1]), -1)  # put seq. dim. back

        # Mask predictions
        masked_pred = (pred * shift_reconstruction_mask).int()

        return masked_pred, ce_loss


def autoregressive_decode(VAE_model, latent_z, labels_GPT2, max_sent_len=-1, train=False):
    if (max_sent_len == -1):
        max_sent_len = labels_GPT2.shape[1] - 1  # -1 for BOS token

    # Skip beginning of sentence token and respect max length
    shift_labels = labels_GPT2[:, 1:max_sent_len + 1].contiguous()

    # Functions
    softmax_fn = torch.nn.Softmax(dim=2)
    ce_fn = torch.nn.CrossEntropyLoss(ignore_index=VAE_model.pad_token_id, reduction='none')

    with torch.set_grad_enabled(train):
        # Start with BOS token
        generated_so_far = torch.tensor(VAE_model.tokenizer_decoder.added_tokens_encoder['<BOS>'],
                                        dtype=torch.long, device=DEVICE)
        # Make a batch of BOS tokens
        generated_so_far = generated_so_far.unsqueeze(0).repeat(latent_z.shape[0], 1)

        logits_lm_so_far = []

        # Generate for the whole batch token per token (auto regressively)
        for i in range(max_sent_len):
            # Outputs is a tuple (hidden, present)
            outputs = VAE_model.decoder(generated_so_far, past=latent_z)
            logits_lm = outputs[0]

            # logits_lm is of dim B x N input_tokens x vocab size
            # we want the output for the last hidden state
            next_token_logits = logits_lm[:, -1, :]

            logits_lm_so_far.append(next_token_logits)

            # No top_k, top_p filtering,
            # just apply softmax and sample woth that distribution
            next_token_probs = F.softmax(next_token_logits, dim=1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            # The generated output can be concatted to the previous ouput
            generated_so_far = torch.cat((generated_so_far, next_token), dim=1)

        logits_sequence = torch.stack(logits_lm_so_far, dim=1)

        probs_resize = logits_sequence.view(-1, logits_sequence.size(-1))
        ce_loss_resize = ce_fn(probs_resize, shift_labels.view(-1))
        ce_loss = torch.sum(ce_loss_resize.view(-1, shift_labels.shape[-1]), -1)  # put seq. dim. back

        return generated_so_far[:, 1:], ce_loss


def encode_reconstruct_auto_and_tf(VAE_model, inputs_BERT, labels_GPT2):
    # Mask tokens that are 0 (PAD in BERT)
    attention_mask = (inputs_BERT > 0).float()

    # ENCODE
    _, pooled_hidden_fea = VAE_model.encoder(inputs_BERT, attention_mask)

    # Connect ENCODE - DECODE
    # Now implemented as taking the mean of the std, log var output,
    # but can also use the connect function of the VAE model
    # latent_z, loss_kl = self.connect(pooled_hidden_fea); latent_z = latent_z.squeeze(1)
    mean, _ = VAE_model.encoder.linear(pooled_hidden_fea).chunk(2, -1)
    latent_z = mean.squeeze(1)

    # DECODE AUTO REGRESSIVE
    gen_pred_auto, ce_loss_auto = autoregressive_decode(VAE_model, latent_z, labels_GPT2)

    # DECODE TEACHER-FORCE
    masked_pred_tf, ce_loss_tf = teacher_force_decode(VAE_model, latent_z, labels_GPT2)

    return gen_pred_auto, ce_loss_auto, masked_pred_tf, ce_loss_tf


if __name__ == "__main__":
    model_paths = get_all_model_paths()
    VAE_models = load_all_models(model_paths)

    # Tokenizers should be the same for all models? Not sure actually
    tokenizer_encoder = VAE_models['snli'][1.0].tokenizer_encoder
    tokenizer_decoder = VAE_models['snli'][1.0].tokenizer_decoder

    sentences = load_sentences(SENTENCES_FILE, max_N_sentences=MAX_SENTENCES, N_sentences=MAX_N_SENTENCES)
    pad_tok_sent_dec, pad_tok_sent_enc = tokenise_pad_sentences(sentences, tokenizer_encoder, tokenizer_decoder)

    n_batches = int(np.ceil(pad_tok_sent_enc.shape[0] / BATCH_SIZE))
    N_BATCHES = n_batches if (N_BATCHES == - 1) else N_BATCHES

    if (MAX_DEC_SEQ_LEN != -1):
        pad_tok_sent_enc = pad_tok_sent_enc[:, :MAX_DEC_SEQ_LEN]
        pad_tok_sent_dec = pad_tok_sent_dec[:, :MAX_DEC_SEQ_LEN]

    bert_sequence_batches = pad_tok_sent_enc.chunk(n_batches)
    gpt2_sequence_batches = pad_tok_sent_dec.chunk(n_batches)



    results_all_models = {}

    for snli_wikipedia in ['snli', 'wikipedia']:
        results_all_models[snli_wikipedia] = {}

        for beta, VAE_model in VAE_models[snli_wikipedia].items():
            print("-" * 30)
            print("Configuration: {} - beta: {}".format(snli_wikipedia, beta))

            pred_auto, ce_losses_auto, pred_tf, ce_losses_tf = [], [], [], []

            for batch_i in range(n_batches):
                print("Batch {}".format(batch_i), end='\r')
                p_au, c_au, p_tf, c_tf = encode_reconstruct_auto_and_tf(VAE_model,
                                                                        bert_sequence_batches[batch_i],
                                                                        gpt2_sequence_batches[batch_i])

                pred_auto.append(p_au.cpu())
                ce_losses_auto.append(c_au.cpu())
                pred_tf.append(p_tf.cpu())
                ce_losses_tf.append(c_tf.cpu())

                if batch_i == (N_BATCHES - 1):
                    break

            # Add results to dict with all configurations
            results_all_models[snli_wikipedia][beta] = {'pred_auto_all': torch.cat(pred_auto),
                                                        'ce_losses_auto_all': torch.cat(ce_losses_auto),
                                                        'pred_tf_all': torch.cat(pred_tf),
                                                        'ce_losses_tf_all': torch.cat(ce_losses_tf)}

    # Save results
    os.makedirs(RESULT_DIR, exist_ok=True)
    now = datetime.now().strftime("%d-%m-%Y--%H:%M:%S")
    with open('{}/results-{}.pickle'.format(RESULT_DIR, now), 'wb') as handle:
        pickle.dump(results_all_models, handle, protocol=pickle.HIGHEST_PROTOCOL)