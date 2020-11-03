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
from transformers import AdamW, get_linear_schedule_with_warmup
from NewsData import NewsData
from torch.utils.data import DataLoader
import numpy as np
import utils


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

    optimizer = AdamW(VAE_model.parameters(),
                      lr=args.learning_rate,
                      eps=args.epsilon)
    return optimizer


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

    # Get data
    data = NewsData(args.dataset_name, args.tokenizer_name,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    pin_memory=True, debug=False)

    for phase in phases:
        sampler = torch.utils.data.distributed.DistributedSampler(data.datasets[phase],
                                                                  num_replicas=int(args.n_gpus * args.n_nodes))
        # With distributed sampling, shuffle must be false
        loaders[phase] = DataLoader(data.datasets[phase], batch_size=args.batch_size,
                                    sampler=sampler, shuffle=False, pin_memory=True,
                                    num_workers=args.num_workers, collate_fn=data.collate_fn)

    return loaders, data


def transfer_batch_to_device(batch, device='cuda'):
    """
    Transfer an input batch to (CUDA) device.

    :param batch:
    :param device:
    :return:
    """

    for k in batch.keys():
        batch[k] = batch[k].to(device)
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


def print_stats(stats, epoch, phase, total_steps, global_step):
    """
    Print statistics to track process.

    :param stats:
    :param epoch:
    :param phase:
    :param total_steps:
    :param global_step:
    :return:
    """

    print_string = "EPOCH {:4} | STEP {:6}/{:6} | {}".format(epoch, global_step, total_steps, phase)

    for s, v in stats[epoch][phase].items():
        print_string += " | {:12}: {:8.4f}".format(s, v[-1])

    print(print_string)


def do_valid_step(VAE_model, batch, args):
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
                           args=args)

    return losses


def do_train_step(VAE_model, batch, batch_i, optimizer, scheduler, scaler, args):
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

    optimizer.zero_grad()

    with torch.set_grad_enabled(True):
        with autocast():
            losses = VAE_model(input_ids=batch['input_ids'],
                               attention_mask=batch['attention_mask'],
                               args=args)
            loss = losses['total_loss'] / args.accumulate_n_batches_grad

    scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    # Gradient accumulation
    if (batch_i + 1) % args.accumulate_n_batches_grad == 0:
        # instead of optimizer.step()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        VAE_model.zero_grad()

    # Detach now (this is not divided by args.accumulate_n_batches)
    # but since we are not summing I think that's fine
    losses['total_loss'] = losses['total_loss'].item()

    return VAE_model, optimizer, scheduler, losses


def log_stats_step(losses, phase, epoch):
    """
    Log losses of step to W&B.

    :param losses:
    :param phase:
    :param epoch:
    :return:
    """


    logs = {"STEP-STATS-{}-{}".format(phase, stat_name): v for stat_name, v in losses.items()}
    logs["{}-epoch".format(phase)] = epoch
    wandb.log(logs)


def log_stats_epoch(stats, epoch):
    """
    Log stats of epoch to W&B.

    :param stats:
    :param epoch:
    :return:
    """

    for phase, phase_stats in stats[epoch].items():
        logs = {"EPOCH-STATS-{}-{}".format(phase, stat_name): np.mean(stats) for stat_name, v in phase_stats.items()}
        wandb.log(logs)


def checkpoint_model(VAE_model, train_step, optimizer, scheduler, run_name):
    """
    Save checkpoint for later use.

    :param VAE_model:
    :param train_step:
    :param optimizer:
    :param scheduler:
    :param run_name:
    :return:
    """

    print("Saving checkpoint at 'Runs/{}/checkpoint.pth'...".format(run_name))
    checkpoint = {
        'train_step': train_step,
        'VAE_model': VAE_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler}

    torch.save(checkpoint, 'Runs/{}/checkpoint.pth'.format(run_name))


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
    wandb.init(name=run_name, project=args.wandb_project, dir='Runs/{}/wandb'.format(run_name), config=args)
    wandb.watch(VAE_model)


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


def prepare_folders(run_name):
    """
    Make folders to save stuff to.

    :param run_name:
    :return:
    """

    os.makedirs('Runs/{}'.format(run_name), exist_ok=True)
    os.makedirs('Runs/{}/wandb'.format(run_name), exist_ok=True)


def train(gpu_rank, args, run_name):
    """
    Train loop and preparation.

    :param gpu_rank:
    :param args:
    :param run_name:
    :return:
    """

    world_size = int(args.n_gpus * args.n_nodes)

    # Set GPU device
    torch.cuda.set_device(gpu_rank)

    # Initiate process group and specify backend configurations
    if gpu_rank == 0: print("Init process group...")
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=gpu_rank)

    # Seed
    seed_everything(args.seed)

    # Get model and set it to the correct device
    VAE_model = get_model_on_device(gpu_rank, args)

    # Init logging
    if args.logging and gpu_rank == 0: init_logging(VAE_model, run_name, args)

    # Data loader
    data_loaders, data = get_dataloader(args, ['train', 'validation'], gpu_rank)

    # Optimizer
    optimizer = get_optimizer(VAE_model, args)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=args.max_train_steps)
    scaler = GradScaler(enabled=True)

    # Set-up DDP
    # TODO: not sure what 'find_unused_parameters' means?
    VAE_model = torch.nn.parallel.DistributedDataParallel(VAE_model, device_ids=[gpu_rank],
                                                          find_unused_parameters=True)

    # Zero grads
    VAE_model.zero_grad()

    # Initialise the stats to keep track of
    stats = utils.make_nested_dict()

    finished_training = False
    epoch, train_step, valid_step = 0, 0, 0

    if gpu_rank == 0: print("Start training!")
    while not finished_training:
        for phase in data_loaders.keys():
            for batch_i, batch in enumerate(data_loaders[phase]):
                # SET DEVICE
                batch = transfer_batch_to_device(batch)

                # PERFORM TRAIN / VALIDATION
                if phase == 'train':
                    VAE_model, optimizer, scheduler, losses = do_train_step(
                        VAE_model, batch, batch_i, optimizer, scheduler, scaler, args)
                    train_step += 1
                else:
                    losses = do_valid_step(VAE_model, batch, args)
                    valid_step += 1

                # PROCESS, PRINT & LOG STATISTICS
                step = train_step if phase == 'train' else valid_step
                max_steps = args.max_train_steps if phase == 'train' else args.max_valid_steps

                # INSERT
                if gpu_rank == 0:
                    stats = insert_stats(stats, losses, epoch, phase)

                # PRINT
                if gpu_rank == 0: print_stats(stats, epoch, phase, max_steps, step)

                # LOG
                if step % args.log_every_n_steps == 0 and args.logging and gpu_rank == 0:
                    log_stats_step(losses, phase, epoch)

                # CHECK POINTING
                # TODO: also save when better validation loss
                if (train_step % args.checkpoint_every_n_steps == 0) and args.checkpointing and gpu_rank == 0:
                    checkpoint_model(VAE_model, train_step, optimizer, scheduler, run_name)

                # CHECK IF FINISHED
                if valid_step >= args.max_valid_steps: break
                if train_step >= args.max_train_steps:
                    finished_training = True
                    break

            if finished_training: break

        if args.logging and gpu_rank == 0:
            log_stats_epoch(stats, epoch)

        epoch += 1


def main(args):
    # Init folders & get unique run name
    run_name = get_run_name(args)
    prepare_folders(run_name)

    # Start distributed training
    set_DDP_environment_vars()
    cudnn.benchmark = True  # optimise backend algo
    mp.spawn(train, nprocs=int(args.n_gpus * args.n_nodes), args=(args, run_name))


if __name__ == "__main__":
    args = NewsVAEArguments.preprare_parser()
    main(args)
