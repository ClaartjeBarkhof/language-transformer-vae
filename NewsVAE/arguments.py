import argparse
import distutils
import utils_train


def preprare_parser(jupyter=False, print_settings=True):
    parser = argparse.ArgumentParser()

    # RUN NAME
    parser.add_argument("--run_name_prefix", default='TEST-ATT-TO-LATENT', type=str,
                        help="Prefix of the run name (to give it a marker or hparam settins) (default: '').")

    # TRAIN / VALIDATION
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for data loading and training.")
    parser.add_argument("--max_train_steps_epoch_per_rank", default=3, type=int,
                        help="Maximum number of train steps (per epoch / phase) (for all set to -1).")  # max 192246
    parser.add_argument("--max_valid_steps_epoch_per_rank", default=10, type=int,
                        help="Maximum number of validation steps (per epoch / phase) (for all set to -1).")  # max 1220
    parser.add_argument("--max_global_train_steps", default=50000, type=int,
                        help="Maximum number of train steps in total. Careful this is NOT the "
                             "number of gradient steps performed. That will be / accumulate_n_batches_grad."
                             "So to account for that multiply max_global_train_steps by accumulate_n_batches_grad.")

    # GRADIENT ACCUMULATION
    parser.add_argument("--accumulate_n_batches_grad", default=1, type=int,
                        help="Number of batches to accumulate gradients over."
                             "Default is no accumulation: 1.")

    # OPTIMISER + SCHEDULER (for the total loss)
    parser.add_argument("--lr", default=0.00005, type=float,
                        help="Learning rate (default: 0.00002).")
    parser.add_argument("--lr_scheduler", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
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
                        help="Whether or not to use gradient checkpointing (default: False).")

    # AUTOMATIC MIXED PRECISION
    parser.add_argument("--use_amp", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
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
    parser.add_argument("--num_workers", default=8, type=int,
                        help="Num workers for data loading.")
    parser.add_argument("--max_seq_len", default=64, type=int,
                        help="What the maximum sequence length the model accepts is (default: 128).")

    # PRINTING
    parser.add_argument("--print_stats", default=True, type=bool,
                        help="Whether or not print stats.")
    parser.add_argument("--print_every_n_steps", default=1, type=int,
                        help="Every how many steps to print.")

    # CHECKPOINTING
    parser.add_argument("--checkpoint", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to checkpoint (save) the model. (default: False).")
    parser.add_argument("--checkpoint_every_n_steps", default=10, type=int,
                        help="Every how many (training) steps to checkpoint (default: 1000).")
    # parser.add_argument("--load_from_checkpoint", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
    #                     help="Load from checkpoint given by checkpoint_file (default: False).")
    # parser.add_argument("--checkpoint_file", default="", type=str,
    #                     help="File name of a checkpoint to load in (default: '').")
    # parser.add_argument("--continue_train_after_checkpoint_loading", default=True,
    #                     type=lambda x: bool(distutils.util.strtobool(x)),
    #                     help="If false, the epoch and best validation etc. are set to "
    #                          "their initial values again. as if training from scratch.")

    # SEED
    parser.add_argument("--seed", default=0, type=int,
                        help="Seed for deterministic runs.")
    parser.add_argument("--deterministic", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to set seed and run everything deterministically.")

    # EMBEDDING SPACE LOSS
    parser.add_argument("--return_embedding_loss", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to log the embedding space loss L2.")
    parser.add_argument("--reduce_seq_dim_embedding_loss", default="mean", type=str,
                        help="How to reduce the embedding space loss along the sequence dimension (default: sum).")
    parser.add_argument("--reduce_batch_dim_embedding_loss", default="mean", type=str,
                        help="How to reduce the embedding space loss along the batch dimension (default: mean).")

    # MODEL
    parser.add_argument("--latent_size", default=32, type=int,
                        help="The size of the latent space. The output from the "
                             "encoder is now 32 x 2 (first and last token). The last projection should be 2 x the "
                             "size of the latent space, because it contains mean and logvar with"
                             "both the dimensionality of the space.")
    parser.add_argument("--add_latent_via_memory", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Add the latent to the decoding process by the memory mechanism"
                             "as descrbed in the Optimus paper (default: True)")
    parser.add_argument("--add_latent_via_embeddings", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Add the latent to the decoding process by adding it to the"
                             "embeddings (initial hidden states). (default: True)")
    parser.add_argument("--add_latent_via_cross_attention", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Add the latent to the decoding process by adding it via the cross attention mechanism"
                             "(Latent Decoder Attention). (default: True)")
    parser.add_argument("--add_latent_via_gating", default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Add the latent to the decoding process using a gating mechanism based on "
                             "self-attention scores. (default: True)")
    parser.add_argument("--do_tie_weights", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to tie the weights of the encoder and decoder"
                             "(default: True).")
    parser.add_argument("--do_tie_embedding_spaces", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to tie the weights of all embedding matrices: encoder input embeddings,"
                             "decoder input embeddings, decoder output embeddings (default: True).")
    parser.add_argument("--add_decoder_output_embedding_bias", default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to add decoder output embedding bias, which makes the"
                             "embedding space different from the input embedding spaces (default: True).")
    parser.add_argument("--decoder_only", default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to run decoder_only mode (not a VAE). This is meant as a baseline mode.")

    # NOISY TEACHER-FORCED
    parser.add_argument("--drop_inputs_decoder", default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to drop input embeddings at the decoder (default: True).")
    parser.add_argument("--drop_inputs_decoder_prob", default=0.2, type=float,
                        help="The probability with which dropping input embeddings at the decoder should happen"
                             "(default: 0.2).")

    code_dir_path = utils_train.get_code_dir()
    parser.add_argument("--code_dir_path", default=code_dir_path, type=str,
                        help="Path to NewsVAE folder (default: depends on machine).")

    #################################################
    # Per objective parameters                      #
    #################################################

    # Objective overview:
    #
    #   1. evaluation: no objective
    #
    #   2. autoencoder
    #
    #   3. vae
    #
    #   4. beta-vae:
    #       - beta * KL
    #          - beta constant
    #          - beta linear sched.
    #          - beta lagrangian (minimum desired rate)
    #
    #   5. free-bits-beta-vae:
    #       = beta-vae + free bits (can't be combined with MDR)
    #
    #   6. beta-tc-vae:
    #       - alpha * MI:
    #           - alpha constant
    #           - alpha linear sched.
    #           - alpha lagrangian (minimum or maximum desired MI)
    #       - beta * TC:
    #           - beta constant
    #           - beta linear sched.
    #       - gamma * Dim. KL
    #           - gamma constant
    #           - gamma linear sched.
    #           - gamma lagrangian (minimum desired Dim. KL)
    #
    #   7. mmd-vae:
    #        - lambda * mmd
    #           - lambda constant

    parser.add_argument("--objective", default='vae', type=str,
                        help="Which objective to use, options:"
                             "  - 1. evaluation (eval)"
                             "  - 2. autoencoder (ae)"
                             "  - 3. vae (vae)"
                             "  - 4. beta-vae (b_vae)"
                             "  - 5. free-bits-beta-vae (fb_b_vae)"
                             "  - 6. beta-tc-vae (b_tc_vae)"
                             "  - 7. mmd-vae (mmd_vae)"
                             "  - 8. hoffman (hoffman_vae)")

    # ---------------------
    # BETA - VAE          #
    # ---------------------

    # ----------------------------------------------------
    # beta-vae:
    # How is the KL term managed:
    parser.add_argument("--b_vae_beta_constant_linear_lagrangian", default="constant", type=str,
                        help="What kind of 'annealing' is used for beta in beta-vae objective mode, options:"
                             "  - constant"
                             "  - linear"
                             "  - lagrangian")

    # If a schedule is used (constant or linear ramp)
    parser.add_argument("--b_vae_beta", default=1.0, type=float,
                        help="The balancing beta term between the reconstruction loss"
                             " and KL-divergence term.")
    parser.add_argument("--b_vae_beta_ramp_len_grad_steps", default=1000, type=int,
                        help="For beta schedule set the length of the ramp in grad steps (default: 1000).")
    parser.add_argument("--b_vae_beta_ramp_type", default="increase", type=str,
                        help="For the parameter scheduler, set whether its orientation is 'decrease' or 'increase'.")

    # If Lagrangian optimisation is used (= Minimum Desired Rate)
    parser.add_argument("--b_vae_kl_lagrangian_target_pd", default=0.5, type=float,
                        help="Per dimension target rate for the MDR constrained optimisation. (default: 0.5)")
    parser.add_argument("--b_vae_kl_lagrangian_alpha", default=0.5, type=float,
                        help="alpha of lagrangian moving average, as in https://arxiv.org/abs/1810.00597. "
                             "If alpha=0, no moving average is used. (default: 0.5)")
    parser.add_argument("--b_vae_kl_lagrangian_lr", default=0.00005, type=float,
                        help="Learning rate for the MDR constraint optimiser. (default: 5e-5)")

    # ---------------------
    # FREE BITS BETA-VAE  #
    # ---------------------

    # free-bits-beta-vae (extends beta_vae parameters), can not be used in combination with Lagrangian optimisation
    parser.add_argument("--fb_b_vae_free_bits_pd", default=0.0, type=float,
                        help="The KL loss per dimension below this value is not taken into account.")

    # ---------------------
    # BETA - TC - VAE     #
    # ---------------------

    # ----------------------------------------------------
    # alpha * MI
    # How is the MI term managed
    parser.add_argument("--b_tc_vae_alpha_constant_linear_lagrangian", default="constant", type=str,
                        help="What kind of 'annealing' is used for alpha in beta-tc-vae objective mode, options:"
                             "  - constant"
                             "  - linear"
                             "  - lagrangian")

    # If a schedule is used (constant or linear ramp) for alpha
    parser.add_argument("--b_tc_vae_alpha", default=1.0, type=float,
                        help="Weighting the MI term in Beta-TCVAE (default: alpha = 1.0)")
    parser.add_argument("--b_tc_vae_alpha_ramp_len_grad_steps", default=1000, type=int,
                        help="For beta schedule set the length of the ramp in grad steps (default: 1000).")
    parser.add_argument("--b_tc_vae_alpha_ramp_type", default="increase", type=str,
                        help="For the parameter scheduler, set whether its orientation is 'decrease' or 'increase'.")

    # If Lagrangian optimisation is used (Minimum or Maximum Desired MI)
    parser.add_argument("--b_tc_vae_MI_lagrangian_target", default=15.0, type=float,
                        help="Per dimension target rate for the MDR constrained optimisation. (default: 0.5)")
    parser.add_argument("--b_tc_vae_MI_lagrangian_alpha", default=0.5, type=float,
                        help="alpha of lagrangian moving average, as in https://arxiv.org/abs/1810.00597. "
                             "If alpha=0, no moving average is used. (default: 0.5)")
    parser.add_argument("--b_tc_vae_MI_lagrangian_lr", default=0.00005, type=float,
                        help="Learning rate for the MDR constraint optimiser. (default: 5e-5)")
    parser.add_argument("--b_tc_vae_MI_lagrangian_relation", default='ge', type=str,
                        help="Whether the MI constraint is 'ge' to the target or 'le' to the target. (default: 'ge')")

    # ----------------------------------------------------
    # beta * TC
    # How is the TC term managed
    parser.add_argument("--b_tc_vae_beta_constant_linear_lagrangian", default="constant", type=str,
                        help="What kind of 'annealing' is used for beta in beta-tc-vae objective mode, options:"
                             "  - constant"
                             "  - linear")

    # If a schedule is used (constant or linear ramp) for beta
    parser.add_argument("--b_tc_vae_beta", default=1.0, type=float,
                        help="Weighting the TC term in Beta-TCVAE (default: beta = 1.0)")
    parser.add_argument("--b_tc_vae_beta_ramp_len_grad_steps", default=1000, type=int,
                        help="For beta schedule set the length of the ramp in grad steps (default: 1000).")
    parser.add_argument("--b_tc_vae_beta_ramp_type", default="increase", type=str,
                        help="For the parameter scheduler, set whether its orientation is 'decrease' or 'increase'.")

    # ----------------------------------------------------
    # gamma * Dimension wise KL
    # How is the Dim KL term managed
    parser.add_argument("--b_tc_vae_gamma_constant_linear_lagrangian", default="constant", type=str,
                        help="What kind of 'annealing' is used for gamma in beta-tc-vae objective mode, options:"
                             "  - constant"
                             "  - linear"
                             "  - lagrangian")

    # If a schedule is used (constant or linear ramp) for gamma
    parser.add_argument("--b_tc_vae_gamma", default=1.0, type=float,
                        help="Weighting the dimension wise KL term in Beta-TCVAE (default: gamma = 1.0)")
    parser.add_argument("--b_tc_vae_gamma_ramp_len_grad_steps", default=1000, type=int,
                        help="For beta schedule set the length of the ramp in grad steps (default: 1000).")
    parser.add_argument("--b_tc_vae_gamma_ramp_type", default="increase", type=str,
                        help="For the parameter scheduler, set whether its orientation is 'decrease' or 'increase'.")

    # If Lagrangian optimisation is used  (Minimum Desired Dimension-wise KL)
    parser.add_argument("--b_tc_vae_Dim_KL_lagrangian_target_pd", default=0.5, type=float,
                        help="Per dimension target rate for the MDR constrained optimisation. (default: 0.5)")
    parser.add_argument("--b_tc_vae_Dim_KL_lagrangian_alpha", default=0.5, type=float,
                        help="alpha of lagrangian moving average, as in https://arxiv.org/abs/1810.00597. "
                             "If alpha=0, no moving average is used. (default: 0.5)")
    parser.add_argument("--b_tc_vae_Dim_KL_lagrangian_lr", default=0.00005, type=float,
                        help="Learning rate for the MDR constraint optimiser. (default: 5e-5)")

    # ---------------------
    # MMD - VAE           #
    # ---------------------

    parser.add_argument("--mmd_vae_lambda", default=10000, type=float,
                        help="Lambda to weight the MMD loss (default: lambda = 10e3).")

    # ---------------------
    # HOFFMAN -VAE        #
    # ---------------------

    # ----------------------------------------------------
    # alpha * MI
    # How is the MI term managed
    parser.add_argument("--hoffman_vae_alpha_constant_linear_lagrangian", default="constant", type=str,
                        help="What kind of 'annealing' is used for gamma in beta-tc-vae objective mode, options:"
                             "  - constant"
                             "  - linear"
                             "  - lagrangian")

    # If a schedule is used (constant or linear ramp) for alpha
    parser.add_argument("--hoffman_vae_alpha", default=1.0, type=float,
                        help="Weighting the MI term in hoffman VAE (default: alpha = 1.0)")
    parser.add_argument("--hoffman_vae_alpha_ramp_len_grad_steps", default=1000, type=int,
                        help="For alpha schedule set the length of the ramp in grad steps (default: 1000).")
    parser.add_argument("--hoffman_vae_alpha_ramp_type", default="increase", type=str,
                        help="For the parameter scheduler, set whether its orientation is 'decrease' or 'increase'.")

    # If Lagrangian optimisation is used
    parser.add_argument("--hoffman_vae_MI_lagrangian_target", default=15.0, type=float,
                        help="Target MI for constrained optimisation in Hoffman VAE. (default: 15.0)")
    parser.add_argument("--hoffman_vae_MI_lagrangian_alpha", default=0.5, type=float,
                        help="alpha of lagrangian moving average, as in https://arxiv.org/abs/1810.00597. "
                             "If alpha=0, no moving average is used. (default: 0.5)")
    parser.add_argument("--hoffman_vae_MI_lagrangian_lr", default=0.00005, type=float,
                        help="Learning rate for the MDR constraint optimiser. (default: 5e-5)")
    parser.add_argument("--hoffman_vae_MI_lagrangian_relation", default='ge', type=str,
                        help="Whether the MI constraint is 'ge' to the target or 'le' to the target. (default: 'ge')")

    # ----------------------------------------------------
    # beta * marginal KL
    # How is the MI term managed
    parser.add_argument("--hoffman_vae_beta_constant_linear_lagrangian", default="constant", type=str,
                        help="What kind of 'annealing' is used for gamma in beta-tc-vae objective mode, options:"
                             "  - constant"
                             "  - linear"
                             "  - lagrangian")

    # If a schedule is used (constant or linear ramp) for alpha
    parser.add_argument("--hoffman_vae_beta", default=1.0, type=float,
                        help="Weighting the marginal KL term in hoffman VAE (default: alpha = 1.0)")
    parser.add_argument("--hoffman_vae_beta_ramp_len_grad_steps", default=1000, type=int,
                        help="For alpha schedule set the length of the ramp in grad steps (default: 1000).")
    parser.add_argument("--hoffman_vae_beta_ramp_type", default="increase", type=str,
                        help="For the parameter scheduler, set whether its orientation is 'decrease' or 'increase'.")

    # If Lagrangian optimisation is used
    parser.add_argument("--hoffman_vae_marg_KL_lagrangian_target", default=0.1, type=float,
                        help="Per dimension target marginal KL for the lagrangian optimisation of marginal KL. (default: 0.5)")
    parser.add_argument("--hoffman_vae_marg_KL_lagrangian_alpha", default=0.5, type=float,
                        help="alpha of lagrangian moving average, as in https://arxiv.org/abs/1810.00597. "
                             "If alpha=0, no moving average is used. (default: 0.5)")
    parser.add_argument("--hoffman_vae_marg_KL_lagrangian_lr", default=0.00005, type=float,
                        help="Learning rate for the MDR constraint optimiser. (default: 5e-5)")
    parser.add_argument("--hoffman_vae_marg_KL_lagrangian_relation", default='le', type=str,
                        help="Whether the MI constraint is 'ge' to the target or 'le' to the target. (default: 'ge')")

    #################################################

    # Hacky thing to be able to run in jupyter
    if jupyter:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    # Objective overview:
    #
    #   1. evaluation: no objective
    #
    #   2. autoencoder
    #
    #   3. vae
    #
    #   4. beta-vae:
    #       - beta * KL
    #          - beta constant
    #          - beta linear sched.
    #          - beta lagrangian (minimum desired rate)
    #
    #   5. free-bits-beta-vae:
    #       = beta-vae + free bits (can't be combined with MDR)
    #
    #   6. beta-tc-vae:
    #       - alpha * MI:
    #           - alpha constant
    #           - alpha linear sched.
    #           - alpha lagrangian (minimum or maximum desired MI)
    #       - beta * TC:
    #           - beta constant
    #           - beta linear sched.
    #       - gamma * Dim. KL
    #           - gamma constant
    #           - gamma linear sched.
    #           - gamma lagrangian (minimum desired Dim. KL)
    #
    #   7. mmd-vae:
    #        - lambda * mmd
    #           - lambda constant

    assert args.objective in ["evaluation", "autoencoder", "vae", "beta-vae", "hoffman",
                              "free-bits-beta-vae", "beta-tc-vae", "mmd-vae"], "Invalid objective, see options. Quit!"

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
