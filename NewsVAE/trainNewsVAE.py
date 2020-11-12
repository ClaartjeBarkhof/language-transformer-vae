import os
import datetime
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import wandb
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
import NewsVAEArguments
from pytorch_lightning import seed_everything
from EncoderDecoderShareVAE import EncoderDecoderShareVAE
from NewsData import NewsData
from torch.utils.data import DataLoader
import numpy as np
import utils
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_DDP_environment_vars(port_nr=1234):
    """
    Set the environment variables for the DDP environment.

    :param port_nr
    """
    os.environ['MASTER_ADDR'] = utils.get_ip()
    os.environ['MASTER_PORT'] = str(port_nr)
    print("Setting MASTER_ADDR: {}, MASTER PORT: {}...".format(utils.get_ip(), port_nr))


def get_model_on_device(gpu_rank, args):
    """
    Initialise a VAE model model from checkpoints and set to CUDA.

    :param gpu_rank:
    :param args:
    :return:
    """
    if gpu_rank == 0: print("Loading model...")

    # Encoder Decoder model
    VAE_model = EncoderDecoderShareVAE(args, args.base_checkpoint_name,
                                       do_tie_weights=args.do_tie_weights)
    VAE_model = VAE_model.cuda(gpu_rank)

    return VAE_model


def get_optimizer(VAE_model, args):
    """
    Return a simple optimiser.

    :param VAE_model:
    :param args:
    :return:
    """

    optimizer = torch.optim.AdamW(VAE_model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    def get_lr_scale(step):
        step += 1

        # First linear warm-up
        if step < args.warmup_updates:
            lr_scale = step / args.warmup_updates
        # Then square root decay
        else:
            lr_scale = 1 / np.sqrt(step - args.warmup_updates)

        return lr_scale

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=get_lr_scale
    )

    return scheduler


def get_dataloader(args, phases, gpu_rank):
    """
    Get data loaders for distributed sampling for different phases (eg. train and validation).

    :param args:
    :param phases:
    :param gpu_rank:
    :return:
    """

    if gpu_rank == 0: print("Get dataloaders...")

    loaders = {}
    samplers = {}
    # Get data
    data = NewsData(args.dataset_name, args.tokenizer_name,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    pin_memory=True, debug=args.debug_data,
                    debug_data_len=args.debug_data_len, max_seq_len=args.max_seq_len)

    for phase in phases:
        if args.ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(data.datasets[phase], rank=gpu_rank, shuffle=True,
                                                                      num_replicas=int(args.n_gpus * args.n_nodes))
            samplers[phase] = sampler

            # With distributed sampling, shuffle must be false
            loaders[phase] = DataLoader(data.datasets[phase], batch_size=args.batch_size,
                                        sampler=sampler, pin_memory=True,
                                        num_workers=args.num_workers, collate_fn=data.collate_fn)
        else:
            # Without distributed sampling
            loaders[phase] = DataLoader(data.datasets[phase], batch_size=args.batch_size,
                                        shuffle=True, pin_memory=True,
                                        num_workers=args.num_workers, collate_fn=data.collate_fn)

            sampler = None

    return loaders, data, samplers


def transfer_batch_to_device(batch, gpu_rank):
    """
    Transfer an input batch to (CUDA) device.

    :param batch:
    :param device:
    :return:
    """

    for k in batch.keys():
        batch[k] = batch[k].cuda(gpu_rank)
    return batch


def insert_stats(stats, new_stats, epoch, phase):
    """
    Add losses to the statistics.

    :param stats:
    :param new_stats:
    :param epoch:
    :param phase:
    :return:
    """

    for stat_name, value in new_stats.items():
        # Make a list to save values to if first iteration of epoch
        if type(stats[epoch][phase][stat_name]) != list:
            stats[epoch][phase][stat_name] = []

        stats[epoch][phase][stat_name].append(value)

    return stats


def print_stats(stats, epoch, phase, global_step, max_global_train_steps,
                phase_step, max_steps, beta):
    """
    Print statistics to track process.

    :param stats:
    :param epoch:
    :param phase:
    :param total_steps:
    :param global_step:
    :return:
    """

    print_string = "EPOCH {:4} | STEP {:5}/{:5} | {} {:6}/{:6}".format(epoch, global_step, max_global_train_steps,
                                                                       phase, phase_step, max_steps)

    for s, v in stats[epoch][phase].items():
        print_string += " | {:8}: {:8.4f}".format(s, v[-1])

    print_string += " | Beta: {:.4f}".format(beta)

    print(print_string)


def do_valid_step(VAE_model, batch, global_step, args):
    """
    Perform a validation step (no grads, eval mode, no autocast?)

    :param VAE_model:
    :param batch:
    :param args:
    :return:
    """

    VAE_model.eval()

    with torch.set_grad_enabled(False):
        losses = VAE_model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           beta=determine_beta(global_step, args),
                           args=args)
        losses['total_loss'] = losses['total_loss'].item()

    return losses


def do_train_step(VAE_model, batch, batch_i, optimizer, scheduler, scaler, global_step, args):
    """
    Perform a train step with autocast, gradients enabled and gradient accumulated backward.

    :param VAE_model:
    :param batch:
    :param batch_i:
    :param optimizer:
    :param scheduler:
    :param scaler:
    :param args:
    :return:
    """

    # TODO: grad clipping see use in combo with amp:
    # https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
    VAE_model.train()

    # Zero gradients
    optimizer.zero_grad()
    VAE_model.zero_grad()

    with torch.set_grad_enabled(True):
        with autocast(enabled=args.use_amp):
            losses = VAE_model(input_ids=batch['input_ids'],
                               attention_mask=batch['attention_mask'],
                               beta=determine_beta(global_step, args),
                               args=args)

            loss = losses['total_loss'] / args.accumulate_n_batches_grad

    # Gradient accumulate in here
    scaler.scale(loss).backward()

    # If gradient accumulated long enough: set a step
    if (batch_i + 1) % args.accumulate_n_batches_grad == 0:
        scaler.step(optimizer)  # instead of optimizer.step()
        scaler.update()
        # Advance the learning rate scheduler by 1
        scheduler.step()

    # Detach now (this is not divided by args.accumulate_n_batches)
    # but since we are not summing I think that's fine
    losses['total_loss'] = losses['total_loss'].item()

    return VAE_model, optimizer, scheduler, losses


def get_lr(scheduler):
    return scheduler.get_last_lr()


def log_stats_step(losses, phase, epoch, log_every, global_step, lr, beta):
    """
    Log losses of step to W&B.

    :param losses:
    :param phase:
    :param epoch:
    :return:
    """

    logs = {"TRAIN-STEP-STATS-{}-{} (steps x {})".format(phase, stat_name, log_every): v for stat_name, v in
            losses.items()}
    logs["{}-epoch".format(phase)] = epoch
    logs["beta"] = beta
    logs["global_step"] = global_step
    logs["STEP-STATS-Learning rate (steps x {})".format(log_every)] = lr
    wandb.log(logs)


def log_stats_epoch(stats, epoch):
    """
    Log stats of epoch to W&B.

    :param stats:
    :param epoch:
    :return:
    """

    logs = {}
    for phase, phase_stats in stats[epoch].items():
        for stat_name, stat in phase_stats.items():
            log_name = "EPOCH-STATS-{}-{}".format(phase, stat_name)
            logs[log_name] = np.mean(stat)
    wandb.log(logs)


def load_from_checkpoint(VAE_model, optimizer, scheduler, scaler, args):
    print("Loading VAE_model, optimizer and scheduler from {}".format(args.checkpoint_file))
    assert os.path.isfile(args.checkpoint_file), "-> checkpoint file path must exist for it to be loaded!"

    checkpoint = torch.load(args.checkpoint_file)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    VAE_model.load_state_dict(checkpoint["VAE_model_state_dict"])
    global_step = checkpoint["global_step"]
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    epoch = checkpoint["epoch"]
    best_valid_loss = checkpoint["best_valid_loss"]

    print("Checkpoint global_step: {}, epoch: {}, best_valid_loss: {}".format(global_step,
                                                                              epoch,
                                                                              best_valid_loss))

    return optimizer, scheduler, VAE_model, scaler, global_step, epoch, best_valid_loss


def save_checkpoint_model(VAE_model, optimizer, scheduler, scaler, run_name,
                          global_step, best_valid_loss, epoch, args, best=False):
    """
    Save checkpoint for later use.

    :param VAE_model:
    :param train_step:
    :param optimizer:
    :param scheduler:
    :param run_name:
    :return:
    """

    # Put in the name that it is a 'best' model
    if best:
        global_step = "best"

    # If just a regular checkpoint, remove the previous checkpoint
    if not best:
        for ckpt in os.listdir('{}Runs/{}'.format(args.prefix_NewsVAE_path, run_name)):
            if ('best' not in ckpt) and ('checkpoint' in ckpt):
                print("Removing previously saved checkpoint: 'Runs/{}/{}'".format(run_name, ckpt))
                os.remove('{}Runs/{}/{}'.format(args.prefix_NewsVAE_path, run_name, ckpt))

    print(
        "Saving checkpoint at '{}Runs/{}/checkpoint-{}.pth'...".format(args.prefix_NewsVAE_path, run_name, global_step))

    checkpoint = {
        'VAE_model_state_dict': VAE_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_valid_loss': best_valid_loss,
        'epoch': epoch,
    }

    torch.save(checkpoint, '{}Runs/{}/checkpoint-{}.pth'.format(args.prefix_NewsVAE_path, run_name, global_step))


def init_logging(VAE_model, run_name, args):
    """
    Initialise W&B logging.

    :param VAE_model:
    :param run_name:
    :param args:
    :param gpu_rank:
    :return:
    """

    print("Initialising W&B logging...")
    wandb.init(name=run_name, project=args.wandb_project,
               dir='{}Runs/{}/wandb'.format(args.prefix_NewsVAE_path, run_name), config=args)
    wandb.watch(VAE_model)


def determine_beta(step, args):
    """
    Determine beta. If using KL.annealing, determine beta based on global step
    and global step compared to KL_annealing_steps. If not using annealing
    use the pre-defined beta or if using another objective (mmd-vae) set beta to 1.

    :param step:
    :param args:
    :return:
    """

    # Consider only every accumulate_n_batches to be a new 'step'
    step = int(np.floor(step / args.accumulate_n_batches_grad))

    # Annealed beta
    if args.KL_annealing:
        cycle_step = step % args.KL_annealing_steps
        # First half of cycle grow linearly from 0 -> 1
        if cycle_step < (args.KL_annealing_steps / 2):
            beta = cycle_step / (args.KL_annealing_steps / 2)
        # Second half of the cycle, beta = 1.0
        else:
            beta = 1.0
    # Static beta
    else:
        beta = args.beta
    return float(beta)


def get_run_name(args):
    """
    Make a unique run name, with the current type and possibly some named prefix.

    :param args:
    :param gpu_rank:
    :return:
    """

    datetime_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    run_name = "{}-run-{}".format(args.run_name_prefix, datetime_stamp)
    print("Run is called: {}...".format(run_name))
    return run_name


def prepare_folders(run_name, args):
    """
    Make folders to save stuff to.

    :param run_name:
    :return:
    """

    os.makedirs('{}Runs/{}'.format(args.prefix_NewsVAE_path, run_name), exist_ok=True)
    os.makedirs('{}Runs/{}/wandb'.format(args.prefix_NewsVAE_path, run_name), exist_ok=True)


def determine_max_epoch_steps(args, data):
    if args.max_valid_steps_epoch == -1:
        max_valid_steps_epoch = len(data.datasets['validation'])
    else:
        max_valid_steps_epoch = args.max_valid_steps_epoch

    if args.max_train_steps_epoch == -1:
        max_train_steps_epoch = len(data.datasets['train'])
    else:
        max_train_steps_epoch = args.max_train_steps_epoch

    return max_train_steps_epoch, max_valid_steps_epoch


def train(gpu_rank, args, run_name):
    """
    Train loop and preparation.

    :param gpu_rank:
    :param args:
    :param run_name:
    :return:
    """
    # Set GPU device
    torch.cuda.set_device(gpu_rank)
    cudnn.benchmark = False  # optimise backend algo

    # Initiate process group and specify backend configurations
    if args.ddp:
        if gpu_rank == 0: print("Init process group...")
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=int(args.n_gpus * args.n_nodes), rank=gpu_rank)

    # Seed everything
    seed_everything(args.seed)

    # Get model and set it to the correct device
    VAE_model = get_model_on_device(gpu_rank, args)

    # Init logging (once)
    if args.logging and gpu_rank == 0: init_logging(VAE_model, run_name, args)

    # Data loader
    data_loaders, data, samplers = get_dataloader(args, ['train', 'validation'], gpu_rank)

    # Optimizer
    optimizer = get_optimizer(VAE_model, args)
    scheduler = get_scheduler(optimizer, args)
    scaler = GradScaler(enabled=args.use_amp)

    # Set-up DDP
    if args.ddp:
        # find_unused_params does not have effect, must be set in file
        VAE_model = torch.nn.parallel.DistributedDataParallel(VAE_model, device_ids=[gpu_rank])

    # Zero grads
    VAE_model.zero_grad()

    # Initialise the stats to keep track of
    stats = utils.make_nested_dict()
    finished_training = False

    epoch, global_step = 0, 0
    best_valid_loss = 1000
    max_train_steps_epoch, max_valid_steps_epoch = determine_max_epoch_steps(args, data)

    if args.load_from_checkpoint:
        optimizer, scheduler, VAE_model, scaler, global_step, epoch, best_valid_loss = load_from_checkpoint(VAE_model,
                                                                                                            optimizer,
                                                                                                            scheduler,
                                                                                                            scaler,
                                                                                                            args)

    if gpu_rank == 0: print("Start or resume training!")

    while not finished_training:
        # TRAIN, VALID
        for phase in data_loaders.keys():
            if args.ddp:
                samplers[phase].set_epoch(epoch)

            phase_step = 0
            max_steps = max_train_steps_epoch if phase == 'train' else max_valid_steps_epoch

            for batch_i, batch in enumerate(data_loaders[phase]):
                if args.time_batch:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                # SET DEVICE
                batch = transfer_batch_to_device(batch, gpu_rank)

                # PERFORM TRAIN / VALIDATION
                if phase == 'train':
                    VAE_model, optimizer, scheduler, losses = do_train_step(
                        VAE_model, batch, batch_i, optimizer, scheduler, scaler, global_step, args)
                else:
                    losses = do_valid_step(VAE_model, batch, global_step, args)

                # PROCESS, PRINT & LOG STATISTICS
                if args.time_batch:
                    end.record()
                    torch.cuda.synchronize()
                    time = start.elapsed_time(end)
                    losses['time'] = time

                # INSERT
                if gpu_rank == 0:
                    stats = insert_stats(stats, losses, epoch, phase)

                # PRINT
                if gpu_rank == 0 and phase_step % args.print_every_n_steps == 0 and args.print_stats:
                    print_stats(stats, epoch, phase, global_step, args.max_global_train_steps,
                                phase_step, max_steps, determine_beta(global_step, args))

                # LOG STEP (ONLY FOR TRAIN)
                if phase_step % args.log_every_n_steps == 0 and args.logging and phase == 'train' and gpu_rank == 0:
                    log_stats_step(losses, phase, epoch, args.log_every_n_steps, global_step,
                                   get_lr(scheduler), determine_beta(global_step, args))

                # CHECK POINTING
                if (global_step % args.checkpoint_every_n_steps == 0) and args.checkpoint and gpu_rank == 0:
                    save_checkpoint_model(VAE_model, optimizer, scheduler, scaler, run_name, global_step,
                                          best_valid_loss, epoch, args)

                # CHECK IF FINISHED (EPOCH & GLOBALLY)
                if phase_step >= max_steps: break
                if global_step >= args.max_global_train_steps: finished_training = True; break

                # ADVANCE A STEP
                phase_step += 1
                if phase == "train": global_step += 1

            if finished_training: break

        # LOG EPOCH STATS
        if args.logging and gpu_rank == 0:
            log_stats_epoch(stats, epoch)

        # BEST MODEL CHECKPOINT
        if phase == 'valid' and args.checkpoint and stats[epoch]['valid']['total_loss'] < best_valid_loss:
            print(
                "Found better validation loss: {:.4f}. Saving checkpoint!".format(stats[epoch]['valid']['total_loss']))
            save_checkpoint_model(VAE_model, optimizer, scheduler, scaler, run_name, global_step, best_valid_loss,
                                  epoch, args, best=True)

        epoch += 1

    print("Average time of passing a batch: {} +- {}".format(np.mean(stats[0]['train']['time']),
                                                             np.std(stats[0]['train']['time'])))


def main(args):
    # Init folders & get unique run name
    run_name = get_run_name(args)
    prepare_folders(run_name, args)

    # Start distributed training
    if args.ddp:
        print("Using DDP")
        set_DDP_environment_vars()
        mp.spawn(train, nprocs=int(args.n_gpus * args.n_nodes), args=(args, run_name))
    else:
        print("Not using DDP, only using device: {}".format(torch.cuda.current_device()))
        train(torch.cuda.current_device(), args, run_name)


if __name__ == "__main__":
    args = NewsVAEArguments.preprare_parser()
    main(args)
