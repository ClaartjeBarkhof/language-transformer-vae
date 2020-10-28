import argparse
import utils


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
    parser.add_argument("--linear_lr_warmup_n_steps", default=20, type=int,
                        help="Number of steps it takes to linearly "
                             "increase to the standard learning rate.")
    parser.add_argument("--max_train_steps", default=100, type=int,
                        help="Maximum number of train steps. Used if earlier than max_epochs.")
    parser.add_argument("--max_epochs", default=None, type=int,
                        help="Maximum number of epochs. Used if earlier than max_train_steps.")
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int,
                        help="Every how many epochs should the validation be executed (default: 1).")

    n_gpus_default = 1 if utils.get_platform()[0] == 'lisa' else None
    log_gpu_memory_default = None if utils.get_platform()[0] == 'local' else 'all'

    # GPU
    parser.add_argument("--accumulate_n_batches_grad", default=1, type=int,
                        help="Number of batches to accumulate gradients over."
                             "Default is no accumulation: 1.")
    parser.add_argument("--n_gpus", default=n_gpus_default, type=int,
                        help="Number GPUs to use (default: None).")
    parser.add_argument("--n_nodes", default=1, type=int,
                        help="Number nodes to use (default: 1).")
    parser.add_argument("--distributed_backend", default=None, type=int,
                        help="Accelerator backend to use:"
                             "  - dp:       DataParallel: multi-GPU"
                             "  - ddp:      DistributedDataParaellel: multi-node GPU"
                             "  - ddp_cpu:  DistributedDataParallel on CPU (only speed-up with multi-node)"
                             "  - ddp2:     dp on node, ddp acrros nodes")
    parser.add_argument("--log_gpu_memory", default=log_gpu_memory_default, type=str,
                        help="Options:"
                             "  - None: no logging"
                             "  - all: log all GPUs on master node"
                             "  - min_max: log the min_max memory in master node.")

    # LOGGING
    parser.add_argument("--log_every_n_steps", default=10, type=int,
                        help="Every how many steps to log (default: 10).")

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
