import torch.distributed as dist
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
from pytorch_lightning import seed_everything

import multiprocessing
from constraintoptim.constraint import *

import utils_train




def do_valid_step(vae_model, batch, device_name="cuda:0"):
    """
    Perform a validation step.
    """
    vae_model.eval()

    with torch.set_grad_enabled(False):

        vae_outputs = vae_model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],

                                return_reconstruction_loss=True,  # ce summed, averaged

                                return_cross_entropy=True,  # ce per word
                                reduce_seq_dim_ce="mean",
                                reduce_batch_dim_ce="mean",

                                return_exact_match=True,  # interpretable to track with validation
                                reduce_seq_dim_exact_match="mean",
                                reduce_batch_dim_exact_match="mean",

                                device_name=device_name)

        vae_outputs['total_loss'] = vae_outputs['total_loss'].item()

    return vae_outputs


def do_train_step(vae_model, batch, global_step, use_amp=False, accumulate_n_batches_grad=1,
                  device_name="cuda:0", gradient_clipping=True):
    """
    Perform a train step with mixed precision auto cast, gradients enabled and gradient accumulated backward.
    """

    # ---------------------------------------------------------------
    # TOTAL LOSS, VAE NETWORK PARAMETERS
    # ---------------------------------------------------------------

    vae_model.train()

    with torch.set_grad_enabled(True):
        with autocast(enabled=use_amp):

            losses = vae_model(input_ids=batch['input_ids'],
                               attention_mask=batch['attention_mask'],
                               return_exact_match=False,
                               return_reconstruction_loss=True,
                               device_name=device_name)

            loss = losses['total_loss'] / accumulate_n_batches_grad

    # All steps below follow what is described on this page:
    # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping

    # Gradient accumulate in here
    vae_model.loss_term_manager.scaler.scale(loss).backward()

    # If gradient accumulated long enough: set a step
    if (global_step + 1) % accumulate_n_batches_grad == 0:

        # GRADIENT CLIPPING
        if gradient_clipping:
            # Unscales the gradients of optimizer's assigned params in-place
            vae_model.loss_term_manager.scaler.unscale_(vae_model.loss_term_manager.total_loss_optimiser)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), 1.0)

        # OPTIMISER STEP
        vae_model.loss_term_manager.scaler.step(vae_model.loss_term_manager.total_loss_optimiser)

        # AMP SCALER UPDATE
        vae_model.loss_term_manager.scaler.update()

        # Zero the gradients only here
        vae_model.loss_term_manager.total_loss_optimiser.zero_grad()

        # LR Scheduler
        # Advance the learning rate scheduler by 1
        vae_model.loss_term_manager.total_loss_scheduler.step()

    # ---------------------------------------------------------------
    # CONSTRAINT OPTIMISATION / LOSS TERM PARAMETER UPDATES
    # ---------------------------------------------------------------
    # Update all the parameters (alpha, beta, gamma, lambda, etc.)
    # and perform update step for constraint optimisers
    # No gradient accumulation do this update every train step
    for loss_term, manager in vae_model.loss_term_manager.manager.items():

        # If parameter scheduler, perform step (to update the parameter value)
        if isinstance(manager, utils_train.ParameterScheduler):
            manager.step()

        # If a constraint optimiser
        elif isinstance(manager, dict):
            manager["optimiser"].step()
            manager["optimiser"].zero_grad()

    # Detach now (this is not divided by args.accumulate_n_batches_grad)
    # but since we are not summing I think that's fine
    losses['total_loss'] = losses['total_loss'].item()
    losses["LR"] = utils_train.get_lr(vae_model.loss_term_manager.total_loss_scheduler)

    return vae_model, losses


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

    # Data loaders / data set / samplers (if ddp)
    data_loaders, data, samplers = get_dataloader(["train", "validation"], ddp=config.ddp, batch_size=config.batch_size,
                                                  num_workers=config.num_workers, max_seq_len=config.max_seq_len,
                                                  world_size=world_size, dataset_name=config.dataset_name,
                                                  tokenizer_name=config.tokenizer_name,
                                                  device_name=device_name, world_master=world_master,
                                                  gpu_rank=device_rank)

    # Get model
    dataset_size = data.datasets['train'].shape[0]
    vae_model = utils_train.get_model_on_device(config=config, dataset_size=dataset_size, device_name=device_name,
                                    world_master=world_master)

    # Initialise logging
    if config.logging and world_master:
        utils_train.init_logging(vae_model, run_name, config.code_dir_path, config.wandb_project, config)



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

            for batch_i, batch in enumerate(data_loaders[phase]):
                # ----------------------------------------------------------------------------------------------------
                # TRAIN / VALIDATION STEPS
                # ----------------------------------------------------------------------------------------------------

                # SET DEVICE
                batch = utils_train.transfer_batch_to_device(batch, device_name)

                # PERFORM TRAIN / VALIDATION STEP
                if phase == 'train':
                    vae_model, losses = do_train_step(
                                    vae_model, batch, global_step,
                                    use_amp=config.use_amp,
                                    accumulate_n_batches_grad=config.accumulate_n_batches_grad,
                                    device_name=device_name,
                                    gradient_clipping=config.gradient_clipping)
                else:
                    losses = do_valid_step(vae_model, batch, device_name=device_name)

                # ----------------------------------------------------------------------------------------------------
                # INSERT STATISTICS, PRINT, LOG, CHECKPOINT
                # ----------------------------------------------------------------------------------------------------

                # INSERT STATISTICS
                stats = utils_train.insert_stats(stats, losses, epoch, phase)

                # PRINT
                if world_master and global_step % config.print_every_n_steps == 0 and config.print_stats:
                    utils_train.print_stats(stats, epoch, phase, global_step, global_max_steps,
                                            global_grad_step, global_max_grad_steps, batch_i, max_steps)

                # LOG STEP (only if world master)
                if batch_i % config.log_every_n_steps == 0 and config.logging and world_master:
                    utils_train.log_losses_step(losses, phase, epoch, config.log_every_n_steps, global_step, global_grad_step)

                # CHECKPOINT
                if (global_step % config.checkpoint_every_n_steps == 0) and phase == 'train' \
                        and config.checkpoint and device_rank == 0:
                    utils_train.save_checkpoint_model(vae_model, run_name, config.code_dir_path, global_step,
                                                      best_valid_loss, epoch, config, best=False)

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
            # TODO: fix this
            # if phase == 'validation' and world_master:
            #     mean_valid_loss = np.mean(stats[epoch]['validation']['recon_loss'])
            #     if config.checkpoint and mean_valid_loss < best_valid_loss:
            #         print(f"Found better (mean) validation loss (at this device): "
            #               f"{mean_valid_loss:.4f}. Saving checkpoint!")
            #         utils_train.save_checkpoint_model(vae_model, optimizer, scheduler, scaler, run_name,
            #                                           config.code_dir_path, global_step,
            #                                           best_valid_loss, epoch, best=True)
            #         best_valid_loss = mean_valid_loss
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

    args = arguments.preprare_parser(jupyter=False, print_settings=True)

    # Warning filter for dataset loading warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(args)
