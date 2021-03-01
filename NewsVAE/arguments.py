import argparse
import distutils
import utils_train


def preprare_parser(jupyter=False, print_settings=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name_prefix", default='TEST-1FEB', type=str,
                        help="Prefix of the run name (to give it a marker or hparam settins) (default: '').")

    # TRAIN / VALIDATION
    parser.add_argument("--batch_size", default=3, type=int,
                        help="Batch size for data loading and training.")
    parser.add_argument("--max_train_steps_epoch_per_rank", default=20, type=int,
                        help="Maximum number of train steps (per epoch / phase) (for all set to -1).") # max 192246
    parser.add_argument("--max_valid_steps_epoch_per_rank", default=20, type=int,
                        help="Maximum number of validation steps (per epoch / phase) (for all set to -1).") # max 1220
    parser.add_argument("--max_global_train_steps", default=50000, type=int,
                        help="Maximum number of train steps in total. Careful this is NOT the "
                             "number of gradient steps performed. That will be / accumulate_n_batches_grad."
                             "So to account for that multiply max_global_train_steps by accumulate_n_batches_grad.")

    # GRADIENT ACCUMULATION
    parser.add_argument("--accumulate_n_batches_grad", default=2, type=int,
                        help="Number of batches to accumulate gradients over."
                             "Default is no accumulation: 1.")

    # OPTIMISER + SCHEDULER
    parser.add_argument("--lr", default=0.00002, type=float,
                        help="Learning rate (default: 0.00002).")
    parser.add_argument("--lr_scheduler", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to use a lr scheduler (default: True).")
    parser.add_argument("--lr_warmup_updates", default=4000, type=int,
                        help="Warm-up updates (gradient steps), how many updates in take to "
                             "take the initial learning rate (default: 1, no warm-up).")
    parser.add_argument("--linear_lr_sched_grad_total_steps", default=27500, type=int,
                        help="Number of training gradient steps to linear decrease "
                             "the learning rate (default: 27500).")
    parser.add_argument("--lr_scheduler_type", default="vaswani", type=str,
                        help="Which learning rate schedule to use (if lr_scheduler is True), "
                             "options: 'vaswani', other things will automatically be interpreted as linear"
                             "warmup with linear decrease (default: vaswani).")
    parser.add_argument("--gradient_clipping", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to use a gradient clipping above norm 1.0 (default: True).")


    # GRADIENT CHECKPOINTING
    parser.add_argument("--gradient_checkpointing", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to use gradient checkpointing (default: True).")

    # AUTOMATIC MIXED PRECISION
    parser.add_argument("--use_amp", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to use automatic mixed precision (default: True).")

    # DISTRIBUTED TRAINING
    parser.add_argument("--n_gpus", default=1, type=int,
                        help="Number GPUs to use (default: None).")
    parser.add_argument("--n_nodes", default=1, type=int,
                        help="Number nodes to use (default: 1).")
    parser.add_argument("--ddp", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to use Distributed Data Parallel (DDP) "
                             "(default: True if n_gpus > 1, else: False).")

    # LOGGING
    parser.add_argument("--logging", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to log the process of the model (default: True).")
    parser.add_argument("--log_every_n_steps", default=1, type=int,
                        help="Every how many steps to log (default: 20).")
    parser.add_argument("--wandb_project", default='thesis', type=str,
                        help="The name of the W&B project to store runs to.")

    # DATA & TOKENISATION
    parser.add_argument("--tokenizer_name", default='roberta', type=str,
                        help="The name of the tokenizer, 'roberta' by default.")
    parser.add_argument("--dataset_name", default='ptb_text_only', type=str,
                        help="The name of the dataset, 'cnn_dailymail' by default, else: ptb_text_only.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Num workers for data loading.")
    parser.add_argument("--max_seq_len", default=64, type=int,
                        help="What the maximum sequence length the model accepts is (default: 128).")

    # PRINTING
    parser.add_argument("--print_stats", default=True, type=bool,
                        help="Whether or not print stats.")
    parser.add_argument("--print_every_n_steps", default=10, type=int,
                        help="Every how many steps to print.")

    # CHECKPOINTING
    parser.add_argument("--checkpoint", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to checkpoint (save) the model. (default: False).")
    parser.add_argument("--checkpoint_every_n_steps", default=1000, type=int,
                        help="Every how many (training) steps to checkpoint (default: 1000).")
    parser.add_argument("--load_from_checkpoint", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Load from checkpoint given by checkpoint_file (default: False).")
    parser.add_argument("--checkpoint_file", default="", type=str,
                        help="File name of a checkpoint to load in (default: '').")
    parser.add_argument("--continue_train_after_checkpoint_loading", default=True,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help="If false, the epoch and best validation etc. are set to "
                             "their initial values again. as if training from scratch.")

    # SEED
    parser.add_argument("--seed", default=0, type=int,
                        help="Seed for deterministic runs.")
    parser.add_argument("--deterministic", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to set seed and run everything deterministically.")

    # LOSS
    parser.add_argument("--hinge_loss_lambda", default=0.25, type=float,
                        help="The KL loss below this value is not taken into account.")
    parser.add_argument("--beta", default=0.5, type=float,
                        help="The balancing beta term between the reconstruction loss"
                             " and KL-divergence term.")
    # EMBEDDING SPACE LOSS
    parser.add_argument("--return_embedding_loss", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to log the embedding space loss L2.")
    parser.add_argument("--reduce_seq_dim_embedding_loss", default="mean", type=str,
                        help="How to reduce the embedding space loss along the sequence dimension (default: sum).")
    parser.add_argument("--reduce_batch_dim_embedding_loss", default="mean", type=str,
                        help="How to reduce the embedding space loss along the batch dimension (default: mean).")

    # LINEAR KL ANNEALING
    parser.add_argument("--kl_linear_annealing", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to perform (linear) KL annealing from 0 to 1 in "
                             "KL_annealing_grad_steps_linear.")
    parser.add_argument("--kl_annealing_grad_steps_linear", default=1000, type=int,
                        help="How many steps to linearly anneal beta from 0 to 1 as a warmup. (default: 1000)")

    # CYCLICAL KL ANNEALING
    parser.add_argument("--kl_cyclical_annealing", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to perform (cyclic) KL annealing from 0 to 1 in "
                             "KL_annealing_grad_steps_per_cycle.")
    parser.add_argument("--kl_annealing_grad_steps_per_cycle", default=9000, type=int,
                        help="How many gradient steps to perform per cycle (default: 9000).")

    parser.add_argument("--objective", default='mmd-vae', type=str,
                        help="Which objective to use, options:"
                             "  - beta-vae"
                             "  - mmd-vae")
    parser.add_argument("--mmd_lambda", default=10000, type=float,
                        help="How much to weight the mmd loss.")

    # MODEL
    parser.add_argument("--latent_size", default=768, type=int,
                        help="The size of the latent space. The output from the "
                             "encoder is now 768 x 2 (first and last token). The last projection should be 2 x the "
                             "size of the latent space, because it contains mean and logvar with"
                             "both the dimensionality of the latent space.")
    parser.add_argument("--add_latent_via_memory", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Add the latent to the decoding process by the memory mechanism"
                             "as descrbed in the Optimus paper (default: True)")
    parser.add_argument("--add_latent_via_embeddings", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Add the latent to the decoding process by adding it to the"
                             "embeddings (initial hidden states). (default: True)")
    parser.add_argument("--do_tie_weights", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to tie the weights of the encoder and decoder"
                             "(default: True).")
    parser.add_argument("--do_tie_embedding_spaces", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to tie the weights of all embedding matrices: encoder input embeddings,"
                             "decoder input embeddings, decoder output embeddings (default: True).")
    parser.add_argument("--add_decoder_output_embedding_bias", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to add decoder output embedding bias, which makes the"
                             "embedding space different from the input embedding spaces (default: True).")

    code_dir_path = utils_train.get_code_dir()
    parser.add_argument("--code_dir_path", default=code_dir_path, type=str,
                        help="Path to NewsVAE folder (default: depends on machine).")

    # Hacky thing to be able to run in jupyter
    if jupyter:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    # Turn off annealing if MMD vae
    if args.objective == "mmd-vae":
        if print_settings:
            print("--> Note to self: Objective is mmd-vae, setting KL-annealing to False and Beta to 1.0.")
        args.KL_annealing = False
        args.beta = 1.0

    assert not (args.kl_cyclical_annealing == args.kl_linear_annealing == True), \
        "Choose one of the two annealing schedules, can't do both at the same time."

    # PRINT SOME SETTINGS
    if print_settings:
        print("-" * 71)
        print("-" * 30, "ARGUMENTS", "-" * 30)
        print("-" * 71)

        for k, v in vars(args).items():
            print(k, ":", v)

        print("-" * 70)
        print("-" * 70)

    return args
