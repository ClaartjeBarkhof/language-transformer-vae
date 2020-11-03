import argparse
import utils


def preprare_parser():
    parser = argparse.ArgumentParser()

    # DATA & TOKENISATION
    parser.add_argument("--tokenizer_name", default='roberta', type=str,
                        help="The name of the tokenizer, 'roberta' by default.")
    parser.add_argument("--dataset_name", default='cnn_dailymail', type=str,
                        help="The name of the dataset, 'cnn_dailymail' by default.")
    parser.add_argument("--num_workers", default=6, type=int,
                        help="Num workers for data loading.")

    # TRAIN / VALIDATION
    parser.add_argument("--batch_size", default=3, type=int,
                        help="Batch size for data loading and training.")
    parser.add_argument("--learning_rate", default=0.001, type=int,
                        help="Learning rate (default: 0.001).")
    parser.add_argument("--epsilon", default=1e-8, type=int,
                        help="Epsilon (default: 1e-8).")
    parser.add_argument("--linear_lr_warmup_n_steps", default=500, type=int,
                        help="Number of steps it takes to linearly "
                             "increase to the standard learning rate.")
    parser.add_argument("--do_linear_warmup", default=False, type=bool,
                        help="Whether or not to do linear warmup.")
    parser.add_argument("--max_train_steps", default=3000, type=int,
                        help="Maximum number of train steps (in total). Used if earlier than max_epochs.")
    parser.add_argument("--max_valid_steps", default=50, type=int,
                        help="Maximum number of validation steps (per epoch).")

    n_gpus_default = 4 if utils.get_platform()[0] == 'lisa' else None

    # GPU
    parser.add_argument("--accumulate_n_batches_grad", default=10, type=int,
                        help="Number of batches to accumulate gradients over."
                             "Default is no accumulation: 1.")
    parser.add_argument("--n_gpus", default=n_gpus_default, type=int,
                        help="Number GPUs to use (default: None).")
    parser.add_argument("--n_nodes", default=1, type=int,
                        help="Number nodes to use (default: 1).")

    # LOGGING
    parser.add_argument("--logging", default=True, type=bool,
                        help="Whether or not to log the process of the model (default: True).")
    parser.add_argument("--log_every_n_steps", default=20, type=int,
                        help="Every how many steps to log (default: 20).")
    parser.add_argument("--wandb_project", default='thesis', type=str,
                        help="The name of the W&B project to store runs to.")
    parser.add_argument("--run_name_prefix", default='test3000', type=str,
                        help="Prefix of the run name (to give it a marker or hparam settins) (default: '').")

    # CHECKPOINTING
    parser.add_argument("--checkpoint", default=True, type=bool,
                        help="Whether or not to checkpoint (save) the model. (default: False).")
    parser.add_argument("--checkpoint_every_n_steps", default=1000, type=int,
                        help="Every how many steps to checkpoint (default: 1000).")

    # SEED
    parser.add_argument("--seed", default=0, type=int,
                        help="Seed for deterministic runs.")
    parser.add_argument("--deterministic", default=True, type=bool,
                        help="Whether or not to set seed and run everything deterministically.")

    # LOSS
    parser.add_argument("--hinge_loss_lambda", default=0.5, type=float,
                        help="The KL loss is capped at this value (hing loss).")
    parser.add_argument("--beta", default=0.5, type=float,
                        help="The balancing beta term between the reconstruction loss"
                             " and KL-divergence term.")

    # MODEL
    parser.add_argument("--base_checkpoint_name", default="roberta-base", type=str,
                        help="The name of the checkpoint to use to initialise the EncoderDecoderVAE.")
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
    parser.add_argument("--do_tie_weights", default=True, type=bool,
                        help="Whether or not to tie the weights of the encoder and decoder"
                             "(default: True).")
    # TODO: share latent with memory mechanism (not 1 latent per layer, but shared over all layers)

    args = parser.parse_args()

    return args
