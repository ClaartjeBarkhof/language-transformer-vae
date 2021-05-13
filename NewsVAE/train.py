import torch.distributed as dist
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
from pytorch_lightning import seed_everything
import multiprocessing
from constraintoptim.constraint import *
import utils_train
import modules.vae as vae
from loss_and_optimisation import ParameterScheduler, LossTermManager
import numpy as np


def do_valid_step(loss_term_manager, batch, device_name="cuda:0", ddp=False, decoder_only=False, iw_ll_n_samples=50,
                  eval_iw_ll_x_gen=True, max_seq_len_x_gen=64):
    """
    Perform a validation step.
    """
    if ddp:
        loss_term_manager.module.vae_model.eval()
    else:
        loss_term_manager.vae_model.eval()

    with torch.set_grad_enabled(False):
        vae_outputs = loss_term_manager(input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask'],
                                        return_exact_match=True,  # interpretable to track with validation
                                        decoder_only=decoder_only,
                                        eval_iw_ll_x_gen=eval_iw_ll_x_gen,
                                        return_posterior_stats=True,
                                        device_name=device_name,
                                        iw_ll_n_samples=iw_ll_n_samples,
                                        return_attention_to_latent=True,
                                        train=True,  # not returning posteriors and latents etc.
                                        max_seq_len_x_gen=max_seq_len_x_gen)
        # Detach as no update being done
        vae_outputs['total_loss'] = vae_outputs['total_loss'].item()

    return vae_outputs


def do_train_step(loss_term_manager, batch, global_step, use_amp=False, accumulate_n_batches_grad=1,
                  device_name="cuda:0", gradient_clipping=True, ddp=False, decoder_only=False):
    """
    Perform a train step with mixed precision auto cast, gradients enabled and gradient accumulated backward.
    """

    # ---------------------------------------------------------------
    # TOTAL LOSS, VAE NETWORK PARAMETERS
    # ---------------------------------------------------------------
    if ddp:
        scaler = loss_term_manager.module.scaler
        total_loss_optim = loss_term_manager.module.total_loss_optimiser
        lr_scheduler = loss_term_manager.module.total_loss_scheduler
        loss_term_manager.module.vae_model.train()
        vae_model = loss_term_manager.module.vae_model
        manager = loss_term_manager.module.manager
    else:
        scaler = loss_term_manager.scaler
        total_loss_optim = loss_term_manager.total_loss_optimiser
        lr_scheduler = loss_term_manager.total_loss_scheduler
        loss_term_manager.vae_model.train()
        vae_model = loss_term_manager.vae_model
        manager = loss_term_manager.manager

    # FORWARD
    with torch.set_grad_enabled(True):
        with autocast(enabled=False):   # TODO: fix autocast
            # Forward through model happens within loss_term_manager
            losses = loss_term_manager(input_ids=batch['input_ids'],
                                       attention_mask=batch['attention_mask'],
                                       return_exact_match=True,
                                       return_posterior_stats=True,
                                       eval_iw_ll_x_gen=False,
                                       iw_ll_n_samples=1,
                                       train=True,
                                       decoder_only=decoder_only,
                                       max_seq_len_x_gen=1,  # no effect if eval_iw_ll_x_gen is False
                                       return_attention_to_latent=False,
                                       device_name=device_name)

            loss = losses['total_loss'] / accumulate_n_batches_grad

    # All steps below follow what is described on this page:
    # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping

    # Gradient accumulate in here (scaled by
    #scaler.scale(loss).backward()
    loss.backward()

    # If gradient accumulated long enough: set a step
    if (global_step + 1) % accumulate_n_batches_grad == 0:

        # GRADIENT CLIPPING
        if gradient_clipping:
            # Unscales the gradients of optimizer's assigned params in-place
            # scaler.unscale_(total_loss_optim)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), 1.0)

        # OPTIMISER STEP (first unscaled)
        # scaler.step(total_loss_optim)
        total_loss_optim.step()

        # AMP SCALER UPDATE
        #scaler.update()

        # Zero the gradients only here
        total_loss_optim.zero_grad()

        # LR Scheduler
        # Advance the learning rate scheduler by 1
        lr_scheduler.step()

    # ---------------------------------------------------------------
    # CONSTRAINT OPTIMISATION / LOSS TERM PARAMETER UPDATES
    # ---------------------------------------------------------------

    # Update all the parameters (alpha, beta, gamma, lambda, etc.)
    # and perform update step for constraint optimisers
    # No gradient accumulation do this update every train step
    for loss_term, m in manager.items():

        # If parameter scheduler, perform step (to update the parameter value)
        if isinstance(m, ParameterScheduler):
            m.step()

        # If a constraint optimiser
        elif isinstance(m, dict):
            # scaler.step(m["optimiser"])
            m["optimiser"].step()
            m["optimiser"].zero_grad()

    # Detach now (this is not divided by args.accumulate_n_batches_grad)
    # but since we are not summing I think that's fine
    losses['total_loss'] = losses['total_loss'].item()
    losses["LR"] = utils_train.get_lr(lr_scheduler)

    return loss_term_manager, losses


def train(device_rank, config, run_name):
    print("**** DEVICE: ", device_rank)

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
    data_loaders, data, samplers = utils_train.get_dataloader(["train", "validation"], ddp=config.ddp,
                                                              batch_size=config.batch_size,
                                                              num_workers=config.num_workers,
                                                              max_seq_len=config.max_seq_len,
                                                              world_size=world_size, dataset_name=config.dataset_name,
                                                              tokenizer_name=config.tokenizer_name,
                                                              device_name=device_name, world_master=world_master,
                                                              gpu_rank=device_rank)

    # Get model and loss term manager
    dataset_size = data.datasets['train'].shape[0]
    loss_term_manager = vae.get_loss_term_manager_with_model(config, world_master=world_master,
                                                             dataset_size=dataset_size, device_name=device_name)

    # Initialise logging
    if config.logging and world_master:
        utils_train.init_logging(loss_term_manager.vae_model, run_name, config.code_dir_path,
                                 config.wandb_project, config)

    # Set-up DDP
    if config.ddp:
        # Wrap both the model and constraints etc in a loss_term_manager nn.Module as suggested here:
        # https://discuss.pytorch.org/t/multiple-modules-with-distributed-data-parallel/115621
        loss_term_manager = torch.nn.parallel.DistributedDataParallel(loss_term_manager,
                                                                      device_ids=[device_rank],
                                                                      find_unused_parameters=True)
        print(f"-> Turned on DDP for device rank {device_rank}")

    # Zero grads TODO: fix this
    # loss_term_manager.zero_grad()

    # Initialise the stats to keep track of
    stats = utils_train.make_nested_dict()
    finished_training = False

    epoch, global_step, global_grad_step, not_improved_epochs = 0, 0, 0, 0
    epoch_pareto_effiency_dict = {"rate": [], "-distortion": [], "iw_ll_mean": [], "iw_ll_x_gen_mean": [], "-D_ks": []}
    best_valid_loss, best_valid_rate, best_valid_rec = 1000, -1000, 1000

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

            if finished_training:
                break

            if config.ddp:
                print(f"-> Setting epoch explicitly to {epoch} on device {device_name}")
                samplers[phase].set_epoch(epoch)  # needed to explicitly shuffle

            max_steps = max_train_steps_epoch_per_rank if phase == 'train' else max_valid_steps_epoch_per_rank
            atts_to_latent, masks = [], []
            for batch_i, batch in enumerate(data_loaders[phase]):
                # ----------------------------------------------------------------------------------------------------
                # TRAIN / VALIDATION STEPS
                # ----------------------------------------------------------------------------------------------------

                # SET DEVICE
                batch = utils_train.transfer_batch_to_device(batch, device_name)

                # PERFORM TRAIN / VALIDATION STEP
                if phase == 'train':
                    loss_term_manager, losses = do_train_step(
                        loss_term_manager,
                        batch, global_step,
                        use_amp=config.use_amp,
                        accumulate_n_batches_grad=config.accumulate_n_batches_grad,
                        device_name=device_name,
                        gradient_clipping=config.gradient_clipping,
                        decoder_only=config.decoder_only,
                        ddp=config.ddp)
                else:
                    losses = do_valid_step(loss_term_manager, batch,
                                           device_name=device_name, ddp=config.ddp, decoder_only=config.decoder_only,
                                           iw_ll_n_samples=config.iw_ll_n_samples, eval_iw_ll_x_gen=config.eval_iw_ll_x_gen,
                                           max_seq_len_x_gen=config.max_seq_len_x_gen)
                    if "attention_to_latent" in losses:
                        atts_to_latent.append(losses["attention_to_latent"].cpu())
                        masks.append(batch["attention_mask"][:, 1:].cpu())
                        del losses["attention_to_latent"]

                # ----------------------------------------------------------------------------------------------------
                # INSERT STATISTICS, PRINT, LOG, CHECKPOINT
                # ----------------------------------------------------------------------------------------------------

                # INSERT STATISTICS
                stats = utils_train.insert_stats(stats, losses, epoch, phase)

                # PRINT
                if world_master and global_step % config.print_every_n_steps == 0 and config.print_stats:
                    utils_train.print_stats(stats, epoch, phase, global_step, global_max_steps,
                                            global_grad_step, global_max_grad_steps, batch_i, max_steps,
                                            config.objective)

                # LOG STEP (only if world master)
                if batch_i % config.log_every_n_steps == 0 and config.logging and world_master:
                    if config.add_latent_w_matrix_influence:
                        utils_train.add_matrix_influence_weight_to_loss(loss_term_manager, global_step,
                                                                        global_grad_step, ddp=config.ddp)
                    utils_train.log_losses_step(losses, phase, epoch, config.log_every_n_steps, global_step,
                                                global_grad_step)


                # CHECKPOINT
                # if (global_step % config.checkpoint_every_n_steps == 0) and phase == 'train' \
                #         and config.checkpoint and device_rank == 0:
                #     vae_model = loss_term_manager.vae_model if config.ddp is False else loss_term_manager.module.vae_model
                #     utils_train.save_checkpoint_model(vae_model, run_name, config.code_dir_path, global_step,
                #                                       best_valid_loss, epoch, config, checkpoint_type="last")

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
            # END OF TRAIN / VALID PHASE
            # ----------------------------------------------------------------------------------------------------

            # BEST MODEL CHECKPOINT
            if phase == 'validation' and world_master:
                val_epoch_stats = stats[epoch]['validation']

                epoch_pareto_effiency_dict, efficient_epochs = utils_train.determine_checkpoint(val_epoch_stats,
                                                                                                epoch_pareto_effiency_dict,
                                                                                                epoch)

                if config.checkpoint:
                    vae_model = loss_term_manager.vae_model if config.ddp is False else loss_term_manager.module.vae_model
                    utils_train.save_checkpoint_model(vae_model, run_name, config.code_dir_path, global_step,
                                                      epoch, config, efficient_epochs, epoch_pareto_effiency_dict)

        # ----------------------------------------------------------------------------------------------------
        # END OF EPOCH
        # ----------------------------------------------------------------------------------------------------

        # LOG EPOCH STATS (if world master)
        if config.logging and world_master:
            utils_train.log_stats_epoch(stats, epoch, global_step, global_grad_step, atts_to_latent, masks)

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
        # train(2, config, run_name)

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
