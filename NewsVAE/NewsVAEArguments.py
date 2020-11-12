import argparse
import utils


def preprare_parser(jupyter=False):
    parser = argparse.ArgumentParser()

    # DATA & TOKENISATION
    parser.add_argument("--tokenizer_name", default='roberta', type=str,
                        help="The name of the tokenizer, 'roberta' by default.")
    parser.add_argument("--dataset_name", default='cnn_dailymail', type=str,
                        help="The name of the dataset, 'cnn_dailymail' by default.")
    parser.add_argument("--num_workers", default=8, type=int,
                        help="Num workers for data loading.")
    parser.add_argument("--debug_data", default=False, type=bool,
                        help="Whether or not to use debug data (default: False).")
    parser.add_argument("--debug_data_len", default=1000, type=int,
                        help="How much data to take for debugging per set. (default: 2000).")
    parser.add_argument("--max_seq_len", default=64, type=int,
                        help="What the maximum sequence length the model accepts is (default: 128).")

    # TRAIN / VALIDATION
    parser.add_argument("--batch_size", default=3, type=int,
                        help="Batch size for data loading and training.")
    parser.add_argument("--max_train_steps_epoch", default=5000, type=int,
                        help="Maximum number of train steps (per epoch / phase) (for all set to -1).")
    parser.add_argument("--max_valid_steps_epoch", default=1000, type=int,
                        help="Maximum number of validation steps (per epoch / phase) (for all set to -1).")
    parser.add_argument("--max_global_train_steps", default=50000, type=int,
                        help="Maximum number of train steps in total.")

    # GRADIENT ACCUMULATION
    parser.add_argument("--accumulate_n_batches_grad", default=6, type=int,
                        help="Number of batches to accumulate gradients over."
                             "Default is no accumulation: 1.")

    # OPTIMISER + SCHEDULER
    parser.add_argument("--lr", default=0.05, type=float,
                        help="Learning rate (default: 0.05).")
    parser.add_argument("--warmup_updates", default=1000, type=int,
                        help="Warm-up updates, how many updates in take to "
                             "take the initial learning rate (default: 1, no warm-up).")

    # GRADIENT CHECKPOINTING
    parser.add_argument("--gradient_checkpointing", default=True, type=bool,
                        help="Whether or not to use gradient checkpointing (default: True).")

    # AUTOMATIC MIXED PRECISION
    parser.add_argument("--use_amp", default=False, type=bool,
                        help="Whether or not to use automatic mixed precision (default: True).")

    # DISTRIBUTED TRAINING
    n_gpus_default = 2 if utils.get_platform()[0] == 'lisa' else None
    ddp_default = True if type(n_gpus_default) == int else False
    parser.add_argument("--n_gpus", default=n_gpus_default, type=int,
                        help="Number GPUs to use (default: None).")
    parser.add_argument("--n_nodes", default=1, type=int,
                        help="Number nodes to use (default: 1).")
    parser.add_argument("--ddp", default=ddp_default, type=bool,
                        help="Whether or not to use Distributed Data Parallel (DDP) "
                             "(default: True if n_gpus > 1, else: False).")

    # LOGGING
    parser.add_argument("--logging", default=False, type=bool,
                        help="Whether or not to log the process of the model (default: True).")
    parser.add_argument("--log_every_n_steps", default=1, type=int,
                        help="Every how many steps to log (default: 20).")
    parser.add_argument("--wandb_project", default='thesis', type=str,
                        help="The name of the W&B project to store runs to.")
    parser.add_argument("--run_name_prefix", default='', type=str,
                        help="Prefix of the run name (to give it a marker or hparam settins) (default: '').")

    # PRINTING
    parser.add_argument("--print_stats", default=True, type=bool,
                        help="Whether or not print stats.")
    parser.add_argument("--print_every_n_steps", default=1, type=int,
                        help="Every how many steps to print.")

    # TIME STEPS
    parser.add_argument("--time_batch", default=True, type=bool,
                        help="Whether or not to log the process of the model (default: True).")

    # CHECKPOINTING
    parser.add_argument("--checkpoint", default=False, type=bool,
                        help="Whether or not to checkpoint (save) the model. (default: False).")
    parser.add_argument("--checkpoint_every_n_steps", default=5, type=int,
                        help="Every how many (training) steps to checkpoint (default: 1000).")
    parser.add_argument("--load_from_checkpoint", default=False, type=bool,
                        help="Load from checkpoint given by checkpoint_file (default: False).")
    parser.add_argument("--checkpoint_file", default="", type=str,
                        help="File name of a checkpoint to load in (default: '').")

    # SEED
    parser.add_argument("--seed", default=0, type=int,
                        help="Seed for deterministic runs.")
    parser.add_argument("--deterministic", default=True, type=bool,
                        help="Whether or not to set seed and run everything deterministically.")

    # LOSS
    parser.add_argument("--hinge_loss_lambda", default=1, type=float,
                        help="The KL loss below this value is not taken into account.")
    parser.add_argument("--beta", default=0.5, type=float,
                        help="The balancing beta term between the reconstruction loss"
                             " and KL-divergence term.")
    parser.add_argument("--KL_annealing", default=True, type=bool,
                        help="Whether or not to perform (cyclic) KL annealing from 0 to 1 in "
                             "KL_annealing_steps.")
    parser.add_argument("--KL_annealing_steps", default=1250, type=int,
                        help="How many steps a full KL-annealing cycle from 0->1 should take"
                             "Be careful it does not increase linearly. It increases linearly"
                             "for half a cycle and then stays 1 for half a cycle.")
    parser.add_argument("--objective", default='mmd-vae', type=str,
                        help="Which objective to use, options:"
                             "  - beta-vae"
                             "  - mmd-vae")
    parser.add_argument("--mmd_lambda", default=10000, type=float,
                        help="How much to weight the mmd loss.")

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
    parser.add_argument("--do_tie_weights", default=False, type=bool,
                        help="Whether or not to tie the weights of the encoder and decoder"
                             "(default: True).")
    prefix_NewsVAE_path = utils.get_code_dir()
    parser.add_argument("--prefix_NewsVAE_path", default=prefix_NewsVAE_path, type=str,
                        help="Path to NewsVAE folder (default: depends on machine).")

    if jupyter:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    if args.objective == "mmd-vae":
        print("--> Note to self: Objective is mmd-vae, setting KL-annealing to False and Beta to 1.0.")
        args.KL_annealing = False
        args.beta = 1.0

    # To quickly set some of the more important parameters
    args.n_gpus = 4
    args.max_seq_len = 64
    args.do_tie_weights = True
    args.gradient_checkpointing = False
    args.ddp = True
    args.accumulate_n_batches_grad = 2
    args.batch_size = 32
    args.run_name_prefix = "BETA-VAE-TESTRUN"
    args.lr = 0.05
    args.logging = True
    args.objective = "beta-vae"

    args.checkpoint = True
    args.checkpoint_every_n_steps = 4000

    args.max_valid_steps_epoch = 500
    args.max_train_steps_epoch = 5000

    args.KL_annealing = True
    args.KL_annealing_steps = args.max_train_steps_epoch

    print("-"*100)
    print("SOME IMPORTANT ARGUMENTS")
    print("-"*100)

    print("MAX SEQ LEN:", args.max_seq_len)
    print("OBJECTIVE:", args.objective)

    if args.objective == 'beta-vae':
        if args.KL_annealing:
            print("BETA-VAE OBJECTIVE with KL ANNEALING:")
            print("KL-ANNEALING (step per effective batch size):", args.KL_annealing)
        else:
            print("BETA-VAE without KL ANNEALING:")
            print("STATIC BETA AT: ", args.beta)
        print("HINGE TARGET KL:", args.hinge_loss_lambda)
    elif args.objective == 'mmd-vae':
        print("MMD-VAE OBJECTIVE")
        print("Lambda to weight the MMD objective:", args.mmd_lambda)

    print('-' * 30)
    print("N_GPUS:", args.n_gpus)
    print("DDP:", args.ddp)
    print("BATCH SIZE:", args.batch_size)
    print("GRADIENT ACCUMULATION (N BATCHES):", args.accumulate_n_batches_grad)
    print("EFFECTIVE BATCHSIZE PER GRAD STEP:", args.n_gpus * args.accumulate_n_batches_grad * args.batch_size)
    print('-'*30)
    print("LR:", args.lr)
    print("TIE WEIGHTS:", args.do_tie_weights)
    print("GRAD CHECKPOINT:", args.gradient_checkpointing)
    print('-' * 30)
    print("CHECKPOINTING:", args.checkpoint)
    print("LOGGING:", args.logging)
    print("RUN PREFIX:", args.run_name_prefix)
    print('-' * 30)
    print("TRAIN (GLOBAL) STEPS:", args.max_global_train_steps)
    print("TRAIN EPOCH LEN:", args.max_train_steps_epoch)
    print("VALID EPOCH LEN:", args.max_valid_steps_epoch)

    if args.debug_data:
        print('-' * 30)
        print("DATA DEBUG (LEN {}): {}".format(args.debug_data_len, args.debug_data))

    print("-" * 100)

    return args
