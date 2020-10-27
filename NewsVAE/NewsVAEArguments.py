import argparse


def preprare_parser():
    parser = argparse.ArgumentParser()

    # DATA & TOKENISATION
    parser.add_argument("--tokenizer_name", default='roberta', type=str,
                        help="The name of the tokenizer, 'roberta' by default.")
    parser.add_argument("--dataset_name", default='cnn_dailymail', type=str,
                        help="The name of the dataset, 'cnn_dailymail' by default.")
    parser.add_argument("--num_workers", default=8, type=int,
                        help="Num workers for data loading.")

    # TRAIN
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size for data loading and training.")
    parser.add_argument("--learning_rate", default=0.05, type=int,
                        help="Learning rate (default: 0.05).")
    parser.add_argument("--linear_lr_warmup_n_steps", default=40e3, type=int,
                        help="Number of steps it takes to linearly "
                             "increase to the standard learning rate.")
    parser.add_argument("--accumulate_n_batches_grad", default=1, type=int,
                        help="Number of batches to accumulate gradients over."
                             "Default is no accumulation: 1.")
    parser.add_argument("--n_train_steps", default=6e5, type=int,
                        help="Number of train steps. Overrules n_epochs.")
    parser.add_argument("--n_epochs", default=None, type=int,
                        help="Number of epochs, overruled by n_train_steps if not None.")

    # LOSS
    parser.add_argument("--hinge_loss_lambda", default=0.5, type=float,
                        help="The KL loss is capped at this value (hing loss).")
    parser.add_argument("--beta", default=0.5, type=float,
                        help="The balancing beta term between the reconstruction loss"
                             " and KL-divergence term.")

    # MODEL
    parser.add_argument("--latent_size", default=768, type=int,
                        help="The size of the latent space. The output from the "
                             "encoder is now 768 x 2 (first and last token). The last projection should be 2 x the "
                             "size of the latent space, because it contains mean and logvar with"
                             "both the dimensionality of the latent space.")
    parser.add_argument("--hidden_size", default=768, type=int,
                        help="The size of hidden representations (default: 768).")
    parser.add_argument("--n_layers", default=12, type=int,
                        help="The number of transformer layers in the encoder and decoder (default: 12).")
    parser.add_argument("--deterministic_connect", default=False, type=bool,
                        help="Whether or not to connect the encoder and decoder deterministically. "
                             "If deterministically, the mean vector is taken to be the latent, otherwise"
                             "the latent vector is sampled.")
    parser.add_argument("--add_latent_via_memory", default=True, type=bool,
                        help="Add the latent to the decoding process by the memory mechanism"
                             "as descrbed in the Optimus paper (default: True)")
    parser.add_argument("--add_latent_via_embeddings", default=True, type=bool,
                        help="Add the latent to the decoding process by adding it to the"
                             "embeddings (initial hidden states). (default: True)")
    # TODO: share latent with memory mechanism (not 1 latent per layer, but shared over all layers)

    args = parser.parse_args()

    return args
