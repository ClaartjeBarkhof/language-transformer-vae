import utils_train
from arguments import preprare_parser
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
from pytorch_lightning import seed_everything
from modules.vae import NewsVAE
from dataset_wrappper import NewsData
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import multiprocessing

import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from modules.decoder import DecoderNewsVAE
from modules.encoder import EncoderNewsVAE


def get_optimizer(vae_model, learning_rate=2e-5):
    """
    Return a simple AdamW optimiser.

    Args:
        vae_model: nn.module
        learning_rate: float

    Returns:
        optimizer: torch.optizer
    """

    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=learning_rate)

    return optimizer


def get_scheduler(optimizer, warmup_updates=4000, lr_scheduler=True, lr=2e-5):
    """
    Returns a learning rate scheduler with linear warm-up (if lr_scheduler).
    Otherwise it returns the unscaled learning rate.

    Args:
        optimizer: torch.optim
        warmup_updates: int:
            How many steps to warm-up (linearly) to the initial learning rate
        lr_scheduler: bool:
            Whether or not to actually apply the scheduler.
    Returns:
        scheduler: LambdaLR scheduler
            This is the learning rate scheduler that may be updated during training by calling .step()
    """

    def get_lr_scale(step):
        step += 1  # to avoid zero division

        # If using a scheduler, calculate the scaling coefficient
        if lr_scheduler:
            # Square root decay as described in Vaswani et al. (2017), section 5.3
            # performs linear warm-up
            arg1 = 1 / np.sqrt(step)
            arg2 = step * (warmup_updates ** (-1.5))
            d_model = 768
            lr_scale = (1 / np.sqrt(d_model)) * np.min([arg1, arg2])

        # Scale to the set learning rate (fixed)
        else:
            lr_scale = lr

        return lr_scale

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=get_lr_scale
    )

    return scheduler


def get_dataloader(phases, ddp=False, batch_size=12, num_workers=8, debug_data=False,
                   debug_data_len=200, max_seq_len=64, world_size=4,
                   dataset_name="cnn_dailymail", tokenizer_name="roberta",
                   device_name="cuda:0", world_master=True, gpu_rank=0):
    """
    Get data loaders for distributed sampling for different phases (eg. train and validation).

    Args:
        phases: list:
            List of phases to make dataloaders for (train, validation, test)
        ddp: bool:
            Whether or not to use Distributed Data Parallel
        batch_size: int
        num_workers: int
        debug_data: bool
            Whether or not to use a fraction of the data for debugging
        debug_data_len: int:
            How many data points to use for debugging if debug mode is on.
        max_seq_len: int
        world_size: int
            Number of nodes x number of GPUs
        dataset_name: str:
            The name of the dataset ("cnn_dailymail")
        tokenizer_name: str:
            The name of the tokenizer ("roberta")
        device_name: str:
            Which device is active.
        world_master: bool:
            Whether or not to print.
        gpu_rank: int:
            If cuda, what the rank of the GPU is.

    Returns:
        loaders: dict[str, torch.utils.data.DataLoader]
            A dict with dataloaders for every named phase.
        data:
            NewsData object with the data # TODO: change this
        samplers: dict[str. torch.utils.data.distributed.DistributedSampler]
            A dict with samplers for the GPU (if ddp).
    """

    if world_master: print("Get dataloaders...")

    pin_memory = True if "cuda" in device_name else False

    loaders = {}
    samplers = {}

    # Get data
    data = NewsData(dataset_name, tokenizer_name,
                    batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, debug=debug_data,
                    debug_data_len=debug_data_len, max_seq_len=max_seq_len,
                    device=device_name)

    for phase in phases:
        # DDP
        if ddp:
            sampler = DistributedSampler(data.datasets[phase], num_replicas=world_size,
                                         shuffle=True, rank=gpu_rank)
            samplers[phase] = sampler

            # With distributed sampling, shuffle must be false, the sampler takes care of that
            loaders[phase] = DataLoader(data.datasets[phase], batch_size=batch_size,
                                        sampler=sampler, pin_memory=True, shuffle=False,
                                        num_workers=num_workers, collate_fn=data.collate_fn)
        # Single GPU and CPU
        else:
            # Without distributed sampling
            loaders[phase] = DataLoader(data.datasets[phase], batch_size=batch_size,
                                        shuffle=True, pin_memory=pin_memory,
                                        num_workers=num_workers, collate_fn=data.collate_fn)

    if world_master: print("Done getting dataloaders...")

    return loaders, data, samplers


def get_model_on_device(device_name="cuda:0", latent_size=768, gradient_checkpointing=False,
                        add_latent_via_memory=True, add_latent_via_embeddings=True,
                        do_tie_weights=True, world_master=True):
    """
    Get a VAE model on correct device.

    Args:
        device_name: str
        latent_size: int
        gradient_checkpointing: bool
        add_latent_via_embeddings: bool
        add_latent_via_memory: bool
        do_tie_weights: bool
        world_master: bool:
    Returns:
        vae_model: NewsVAE object
    """

    if world_master: print("Loading model...")

    decoder = DecoderNewsVAE(gradient_checkpointing=gradient_checkpointing)
    encoder = EncoderNewsVAE(gradient_checkpointing=gradient_checkpointing, latent_size=latent_size)

    vae_model = NewsVAE(encoder, decoder, latent_size=latent_size,
                        add_latent_via_memory=add_latent_via_memory,
                        add_latent_via_embeddings=add_latent_via_embeddings,
                        do_tie_weights=do_tie_weights)

    vae_model = vae_model.to(device_name)

    if world_master: print("Done model...")

    return vae_model


def determine_beta(global_grad_step, config_beta=0.5, KL_cyclical_annealing=False, KL_linear_annealing=True,
                   KL_annealing_grad_steps_linear=1000, KL_annealing_grad_steps_per_cycle=9000):
    """
    Determine beta given the current global gradient step. Cycles consists
    for one half of annealing from 0 to 1 and the other half of keeping beta
    constant at one. To then start a new cycle.

    Args:
        global_grad_step: int
            How many gradient steps are performed so far.
        KL_cyclical_annealing: bool:
            Whether or not to anneal the KL divergence cyclically
        KL_annealing_grad_steps_per_cycle: int
            How many gradient steps to perform a full cycle.
        KL_linear_annealing: bool:
            Whether or not to anneal the KL divergence linearly (warm-up annealing)
        KL_annealing_grad_steps_linear: int
            How many gradient steps to use to anneal from 0 to 1 linearly
    Returns:
        beta: float
            Current beta weight to use.
    """

    # CYCLICAL ANNEALING
    if KL_cyclical_annealing:
        cycle_step = global_grad_step % KL_annealing_grad_steps_per_cycle

        # First half of cycle grow linearly from 0 -> 1
        if cycle_step < (KL_annealing_grad_steps_per_cycle / 2):
            if global_grad_step > 0:
                beta = cycle_step / (KL_annealing_grad_steps_per_cycle / 2)
            else:
                beta = 0.0

        # Second half of the cycle, beta = 1.0
        else:
            beta = 1.0

    # LINEAR / WARMUP ANNEALING
    elif KL_linear_annealing:
        if global_grad_step < KL_annealing_grad_steps_linear:
            beta = (global_grad_step + 1) / KL_annealing_grad_steps_linear
        else:
            beta = 1.0

    # NO ANNEALING
    else:
        beta = config_beta

    return float(beta)


def do_valid_step(vae_model, batch, beta, hinge_kl_loss_lambda):
    """
    Perform a validation step (no grads, eval mode, no autocast?)

    Args:
        vae_model: nn.module
        batch: Dict[str, Tensor]
        beta: float
            The current beta to balance the KL divergence in the loss.
        hinge_kl_loss_lambda: float
            What the minimal value of KL should be capped at per dimension
    Returns:
        losses: Dict[str, Union[float, Tensor]]
            statistics accumulated in a dict
    """

    vae_model.eval()

    with torch.set_grad_enabled(False):
        losses = vae_model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           beta=beta,
                           return_predictions=False,
                           return_exact_match_acc=True,
                           return_attention_probs=False,
                           hinge_kl_loss_lambda=hinge_kl_loss_lambda)
        losses['total_loss'] = losses['total_loss'].item()

    return losses


def do_train_step(vae_model, batch, optimizer, scheduler, scaler, global_step, beta, hinge_kl_loss_lambda,
                  use_amp=False, accumulate_n_batches_grad=1):
    """
    Perform a train step with autocast, gradients enabled and gradient accumulated backward.

    Args:
        vae_model: nn.Module
        batch: Dict[str, Tensor]
            Input data to the model
        optimizer: torch.optim
        scheduler:
            Learning rate scheduler
        scaler:
            Grad scaler (amp)
        global_step: int
        use_amp: bool:
            Whether or not to use automatic mixed precision.
        accumulate_n_batches_grad: int
        beta: float
    """

    # TODO: grad clipping see use in combo with amp:
    # https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
    vae_model.train()

    with torch.set_grad_enabled(True):
        with autocast(enabled=use_amp):
            losses = vae_model(input_ids=batch['input_ids'],
                               attention_mask=batch['attention_mask'],
                               beta=beta,
                               return_predictions=False,
                               return_exact_match_acc=True,
                               return_attention_probs=False,
                               hinge_kl_loss_lambda=hinge_kl_loss_lambda)

            loss = losses['total_loss'] / accumulate_n_batches_grad

    # Gradient accumulate in here
    scaler.scale(loss).backward()

    # If gradient accumulated long enough: set a step
    if (global_step + 1) % accumulate_n_batches_grad == 0:
        scaler.step(optimizer)  # instead of optimizer.step()

        scaler.update()

        # Advance the learning rate scheduler by 1
        scheduler.step()

        # Zero the gradients only here
        optimizer.zero_grad()

    # Detach now (this is not divided by args.accumulate_n_batches_grad)
    # but since we are not summing I think that's fine
    losses['total_loss'] = losses['total_loss'].item()

    return vae_model, optimizer, scheduler, losses


def train(device_rank, config, run_name):
    # Device
    device_name = utils_train.set_device(device_rank)

    # Determine world size and whether this device is world master
    world_master, world_size = utils_train.get_world_specs(config.n_gpus, config.n_nodes, device_name)

    # Determine the maximum number of steps for this device
    global_max_steps, global_max_grad_steps = utils_train.determine_global_max_steps(config.max_global_train_steps,
                                                                                     config.batch_size, world_size,
                                                                                     config.accumulate_n_batches_grad)

    # Initiate process group and specify backend configurations
    if config.ddp:
        if world_master: print("Init process group...")
        if world_master: print(f"--> CPU count {multiprocessing.cpu_count()}")
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=int(config.n_gpus * config.n_nodes), rank=device_rank)

    # Seed everything
    seed_everything(config.seed)

    # Get model
    vae_model = get_model_on_device(device_name=device_name, latent_size=config.latent_size,
                                    gradient_checkpointing=config.gradient_checkpointing,
                                    add_latent_via_memory=config.add_latent_via_memory,
                                    add_latent_via_embeddings=config.add_latent_via_embeddings,
                                    do_tie_weights=config.do_tie_weights, world_master=world_master)

    # Initalise logging
    if config.logging and world_master: utils_train.init_logging(vae_model, run_name, config.code_dir_path,
                                                                 config.wandb_project, config)

    # Data loaders / data set / samplers (if ddp)
    data_loaders, data, samplers = get_dataloader(["train", "validation"], ddp=config.ddp, batch_size=config.batch_size,
                                                  num_workers=config.num_workers, debug_data=config.debug_data,
                                                  debug_data_len=config.debug_data_len, max_seq_len=config.max_seq_len,
                                                  world_size=world_size, dataset_name=config.dataset_name,
                                                  tokenizer_name=config.tokenizer_name,
                                                  device_name=device_name, world_master=world_master,
                                                  gpu_rank=device_rank)

    # Optimizer, learning rate scheduler and amp scaler
    optimizer = get_optimizer(vae_model, 1.0)  # lr is set to 1.0 because the scheduler takes care of the actual learning rate
    scheduler = get_scheduler(optimizer, config.lr_warmup_updates, config.lr_scheduler, config.lr)
    scaler = GradScaler(enabled=config.use_amp)

    # Set-up DDP
    if config.ddp:
        vae_model = torch.nn.parallel.DistributedDataParallel(vae_model, device_ids=[device_rank],
                                                              find_unused_parameters=True)
        print(f"-> Turned on DDP for device rank {device_rank}")

    # Zero grads
    vae_model.zero_grad()

    # Initialise the stats to keep track of
    stats = utils_train.make_nested_dict()
    finished_training = False

    epoch, global_step, global_grad_step, best_valid_loss = 0, 0, 0, 1000

    # These are actual steps, not gradient steps, so they work in combination with global step
    max_train_steps_epoch_per_rank, max_valid_steps_epoch_per_rank = utils_train.determine_max_epoch_steps_per_rank(
        config.max_train_steps_epoch_per_rank, config.max_valid_steps_epoch_per_rank, data.datasets,
        config.batch_size, world_size=world_size, world_master=world_master)

    if config.load_from_checkpoint:
        # Load model and all relevant states and counters
        if config.continue_train_after_checkpoint_loading:
            optimizer, scheduler, VAE_model, scaler, global_step, \
            epoch, best_valid_loss = utils_train.load_from_checkpoint(vae_model, config.checkpoint_file,
                                                                      optimizer=optimizer, scheduler=scheduler,
                                                                      scaler=scaler, world_master=world_master,
                                                                      ddp=config.ddp, use_amp=config.use_amp)
        # Only load the model
        else:
            _, _, VAE_model, _, _, _, _ = utils_train.load_from_checkpoint(vae_model, config.checkpoint_file,
                                                                           world_master=world_master, ddp=config.ddp)

    if world_master: print("Start or resume training!")

    # ----------------------------------------------------------------------------------------------------
    # TRAINING!
    # ----------------------------------------------------------------------------------------------------
    while not finished_training:
        # TRAIN, VALID
        for phase in data_loaders.keys():

            if finished_training: break

            if config.ddp:
                print(f"-> Setting epoch explicitly to {epoch} on device {device_name}")
                samplers[phase].set_epoch(epoch)  # needed to explicitly shuffle

            max_steps = max_train_steps_epoch_per_rank if phase == 'train' else max_valid_steps_epoch_per_rank

            # TODO: add profiler to keep track of timing
            for batch_i, batch in enumerate(data_loaders[phase]):

                print(
                    f"{device_name} | batch_i {batch_i} | global_step {global_step} | global_grad_step {global_grad_step}")

                # ----------------------------------------------------------------------------------------------------
                # TRAIN / VALIDATION STEPS
                # ----------------------------------------------------------------------------------------------------

                # SET DEVICE
                batch = utils_train.transfer_batch_to_device(batch, device_name)

                # Determine beta (annealed or fixed)
                beta = determine_beta(global_grad_step, config_beta=config.beta,
                                      KL_cyclical_annealing=config.KL_cyclical_annealing,
                                      KL_annealing_grad_steps_per_cycle=config.KL_annealing_grad_steps_per_cycle,
                                      KL_linear_annealing=config.KL_linear_annealing,
                                      KL_annealing_grad_steps_linear=config.KL_annealing_grad_steps_linear)

                # PERFORM TRAIN / VALIDATION
                if phase == 'train':
                    vae_model, optimizer, scheduler, losses = do_train_step(vae_model, batch, optimizer,
                                                                            scheduler, scaler, global_step, beta,
                                                                            config.hinge_loss_lambda)
                else:
                    losses = do_valid_step(vae_model, batch, beta, config.hinge_loss_lambda)

                # ----------------------------------------------------------------------------------------------------
                # INSERT STATISTICS, PRINT, LOG, CHECKPOINT
                # ----------------------------------------------------------------------------------------------------

                # INSERT STATISTICS
                stats = utils_train.insert_stats(stats, losses, epoch, phase)

                # PRINT
                if world_master and global_step % config.print_every_n_steps == 0 and config.print_stats:
                    utils_train.print_stats(stats, epoch, phase, global_step, global_max_steps,
                                            global_grad_step, global_max_grad_steps, batch_i, max_steps,
                                            beta, utils_train.get_lr(scheduler))

                # LOG STEP (only if world master)
                if batch_i % config.log_every_n_steps == 0 and config.logging and world_master:
                    utils_train.log_losses_step(losses, phase, epoch, config.log_every_n_steps, global_step,
                                                global_grad_step, utils_train.get_lr(scheduler), beta)

                # CHECKPOINT
                if (global_step % config.checkpoint_every_n_steps == 0) and phase == 'train' \
                        and config.checkpoint and device_rank == 0:
                    utils_train.save_checkpoint_model(vae_model, optimizer, scheduler, scaler, run_name,
                                                      config.code_dir_path, global_step, best_valid_loss, epoch)

                # ----------------------------------------------------------------------------------------------------
                # KEEP TRACK OF STEPS (IN PHASE AND GLOBALLY)
                # ----------------------------------------------------------------------------------------------------

                # ADVANCE STEP if in train mode
                if phase == "train":
                    global_step += 1
                    if global_step % config.accumulate_n_batches_grad == 0:
                        global_grad_step += 1

                # CHECK IF EPOCH PHASE IS OVER (after advancing one)
                if batch_i >= max_steps: break
                if global_step >= global_max_steps: finished_training = True; break

            # ----------------------------------------------------------------------------------------------------
            # END OF PHASE
            # ----------------------------------------------------------------------------------------------------

            # BEST MODEL CHECKPOINT
            if phase == 'validation' and world_master:
                mean_valid_loss = np.mean(stats[epoch]['validation']['total_loss'])
                if config.checkpoint and mean_valid_loss < best_valid_loss:
                    print(f"Found better (mean) validation loss (at this device): "
                          f"{mean_valid_loss:.4f}. Saving checkpoint!")
                    utils_train.save_checkpoint_model(vae_model, optimizer, scheduler, scaler, run_name,
                                                      config.code_dir_path, global_step,
                                                      best_valid_loss, epoch, best=True)
                    best_valid_loss = mean_valid_loss
            # TODO: evaluation loop

        # ----------------------------------------------------------------------------------------------------
        # END OF EPOCH
        # ----------------------------------------------------------------------------------------------------

        # LOG EPOCH STATS (if world master)
        if config.logging and world_master:
            utils_train.log_stats_epoch(stats, epoch, global_step, global_grad_step)

        epoch += 1


def main(config):
    # Init folders & get unique run name
    run_name = utils_train.get_run_name(config.run_name_prefix)
    utils_train.prepare_folders(run_name, config.code_dir_path)

    print("-" * 71)
    print("-" * 30, "IN MAIN", "-" * 30)
    print("-" * 71)

    # DDP
    if config.ddp:
        print(f"*** Using DDP, spawing {config.n_gpus * config.n_nodes} processes")
        utils_train.set_ddp_environment_vars()
        mp.spawn(train, nprocs=int(config.n_gpus * config.n_nodes), args=(config, run_name))

    # Single GPU
    elif not config.ddp and config.n_gpus > 0:
        print(f"*** Not using DDP, only using device: {torch.cuda.current_device()}")
        train(torch.cuda.current_device(), config, run_name)

    # CPU
    else:
        print("*** Using CPU, you're warned!")
        train("cpu", config, run_name)

    print("-" * 71)
    print("-" * 71)


if __name__ == "__main__":
    import warnings
    import arguments

    config = arguments.preprare_parser(jupyter=False, print_settings=True)
    # config.ddp = True
    # config.n_gpus = 1
    # config.max_global_train_steps = 30
    # config.print_every_n_steps = 1
    # config.num_workers = 2
    # config.accumulate_n_batches_grad = 4
    # config.logging = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(config)