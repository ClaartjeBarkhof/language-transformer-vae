import platform
import socket
import collections
import datetime
import os
import wandb
import arguments
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils_external import tie_weights
import modules.vae as vae

from dataset_wrappper import NewsData

# ----------------------------------------------------------------------------------------------------
# INITIALISATION STUFF
# ----------------------------------------------------------------------------------------------------

def set_device(device_rank):
    """
    Set the device in case of cuda and give the device a name for .to(...) operations.

    Args:
        device_rank: Union[str, int]
            GPU rank if it is an integer, else "cpu" string.
    Returns:
        device_name: str:
            A device name that can be used with .to(device)
    """
    # GPU
    if type(device_rank) == int:
        print(f"-> Setting device {device_rank}")
        torch.cuda.set_device(device_rank)
        cudnn.benchmark = True  # optimise backend algo
        device_name = f"cuda:{device_rank}"

    # CPU
    else:
        device_name = "cpu"
    return device_name


def set_ddp_environment_vars(port_nr=1234):
    """
    Set the environment variables for the DDP environment.

    Args:
        port_nr: int: the port to use (default: 1234)
    """
    os.environ['MASTER_ADDR'] = get_ip()
    os.environ['MASTER_PORT'] = str(port_nr)
    print("Setting MASTER_ADDR: {}, MASTER PORT: {}...".format(get_ip(), port_nr))


def get_run_name(run_name_prefix=""):
    """
    Make a unique run name, with the current type and possibly some named prefix.

    Args:
        run_name_prefix: str
    """

    datetime_stamp = datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
    date, time = datetime_stamp.split("--")[0], datetime_stamp.split("--")[1]
    run_name = "{}-{}-run-{}".format(date, run_name_prefix, time)
    print("Run is called: {}...".format(run_name))
    return run_name


def prepare_folders(run_name, code_path):
    """
    Make folders to save stuff in.

    Args:
        run_name: str
        code_path: str:
            The path to the NewsVAE code directory (dependent on the machine)
    """

    os.makedirs('{}/Runs/{}'.format(code_path, run_name), exist_ok=True)
    os.makedirs('{}/Runs/{}/wandb'.format(code_path, run_name), exist_ok=True)


def determine_global_max_steps(max_global_train_steps, batch_size, world_size, accumulate_n_batches_grad):
    effective_batch_size = world_size * accumulate_n_batches_grad * batch_size
    max_global_grad_steps_rank = int(max_global_train_steps / accumulate_n_batches_grad)

    print(f"Max global steps on this rank: {max_global_train_steps}")
    print(f"This amounts to {max_global_grad_steps_rank} max global gradient steps on this rank.")
    print(f"The effective batch size is {effective_batch_size} (world_size x grad_acc_steps x batch_size = "
          f"{world_size} x {accumulate_n_batches_grad} x {batch_size} = {effective_batch_size}).")

    return max_global_train_steps, max_global_grad_steps_rank


def get_world_specs(n_gpus, n_nodes, device_name):
    """
    Determine the world size and whether the current device is world master.

    Args:
        n_gpus: int
        n_nodes: int
        device_name: str

    Returns:
        world_size: int
            How many devices are active
        world_master: bool
            Whether the current device is world master (GPU 0 or CPU)
    """

    world_size = int(n_gpus * n_nodes) if "cuda" in device_name else 1
    world_master = True if device_name in ["cuda:0", "cpu"] else False

    return world_master, world_size


def determine_max_epoch_steps_per_rank(max_train_steps_epoch_per_rank, max_valid_steps_epoch_per_rank, datasets_dict,
                                       batch_size, world_size=1, world_master=True):
    """
    Determine epoch length by applying logic to the configuration and
    the size of the dataset.

    First determine how many batches are present in the dataset (steps).
    Then see how many there are per rank (/ world size).
    Then determine whether that is more than set in the settings. If so, set to maximum.
    If set to -1, set to maximum as well.

    Args:

    """
    assert "train" in datasets_dict and "validation" in datasets_dict, \
        "-> Expects a dataset with both train and validation keys present."

    # How many batches are there in the dataset per rank
    n_train_batches_per_rank = int(len(datasets_dict['train']) / (world_size * batch_size))
    n_valid_batches_per_rank = int(len(datasets_dict['validation']) / (world_size * batch_size))

    if world_master:
        print(f"Length of the train set: {len(datasets_dict['train'])}, which amounts to max. "
              f"{n_train_batches_per_rank} batches of size {batch_size} per rank.")
        print(f"Length of the validation set: {len(datasets_dict['validation'])}, which amounts to max. "
              f"{n_valid_batches_per_rank} batches of size {batch_size} per rank.")
        print(f"max_train_steps_epoch_per_rank in config set to: "
              f"{max_train_steps_epoch_per_rank}")
        print(f"max_valid_steps_epoch_per_rank in config set to: "
              f"{max_valid_steps_epoch_per_rank}")

    # Train
    if max_train_steps_epoch_per_rank == -1:
        max_train_steps_epoch_per_rank = n_train_batches_per_rank
        if world_master: print(f"Setting max_train_steps_epoch to {n_train_batches_per_rank} "
                               f"given a world size of {world_size}")
    else:
        if max_train_steps_epoch_per_rank > n_train_batches_per_rank:
            if world_master:
                print("Warning: multiple passes over the TRAIN data are "
                      "contained in the number of steps per epoch set. "
                      "Setting max steps to epoch length.")
            max_train_steps_epoch_per_rank = n_train_batches_per_rank

    # Validation
    if max_valid_steps_epoch_per_rank == -1:
        if world_master: print(f"Setting max_train_steps_epoch to {n_train_batches_per_rank} "
                               f"given a world size of {world_size}")
        max_valid_steps_epoch_per_rank = n_valid_batches_per_rank
    else:
        if max_valid_steps_epoch_per_rank > n_valid_batches_per_rank:
            print("Warning: multiple passes over the VALIDATION data are "
                  "contained in the number of steps per epoch set. "
                  "Setting max steps to epoch length.")
            max_valid_steps_epoch_per_rank = n_valid_batches_per_rank

    return max_train_steps_epoch_per_rank, max_valid_steps_epoch_per_rank

# ----------------------------------------------------------------------------------------------------
# DATA LOADERS
# ----------------------------------------------------------------------------------------------------

def get_dataloader(phases, ddp=False, batch_size=12, num_workers=8, max_seq_len=64, world_size=4,
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
                    pin_memory=True, max_seq_len=max_seq_len,
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


# ----------------------------------------------------------------------------------------------------
# LOAD + SAVE MODEL & CHECKPOINTING
# ----------------------------------------------------------------------------------------------------

def load_from_checkpoint(path, world_master=True, ddp=False, device_name="cuda:0", latent_size=64,
                         do_tie_embeddings=True, do_tie_weights=True, add_latent_via_memory=True,
                         add_latent_via_embeddings=True, do_tie_embedding_spaces=True, dataset_size=3370,
                         add_decoder_output_embedding_bias=False, objective="evaluation", evaluation=True):
    # DETERMINE / CHECK PATH
    assert os.path.isfile(path), f"-> checkpoint file path ({path}) must exist for it to be loaded!"
    if world_master: print("Loading model from checkpoint: {}".format(path))

    # LOAD CHECKPOINT
    checkpoint = torch.load(path, map_location='cpu')

    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = arguments.preprare_parser(jupyter=True, print_settings=False)
        config.do_tie_weights = do_tie_weights
        config.objective = objective
        config.latent_size = latent_size
        config.add_latent_via_memory = add_latent_via_memory
        config.add_latent_via_embeddings = add_latent_via_embeddings
        config.do_tie_embedding_spaces = do_tie_embedding_spaces
        config.add_decoder_output_embedding_bias = add_decoder_output_embedding_bias

    vae_model = vae.get_model_on_device(config, dataset_size=dataset_size, device_name=device_name, world_master=True)
    # Bring to CPU, as state_dict loading needs to happen in CPU (strange memory errors occur otherwise)
    vae_model = vae_model.cpu()

    # DDP vs no DDP
    parameter_state_dict = checkpoint["VAE_model_state_dict"]
    # MODEL
    if "module." in list(checkpoint["VAE_model_state_dict"].keys())[0] and not ddp:
        print("Removing module string from state dict from checkpoint")
        parameter_state_dict = add_remove_module_from_state_dict(parameter_state_dict, remove=True)

    elif "module." not in list(checkpoint["VAE_model_state_dict"].keys())[0] and ddp:
        print("Adding module string to state dict from checkpoint")
        parameter_state_dict = add_remove_module_from_state_dict(remove=False)

    # in place procedure
    vae_model.load_state_dict(parameter_state_dict)
    vae_model = vae_model.to(device_name)

    # GLOBAL STEP, EPOCH, BEST VALIDATION LOSS
    global_step = checkpoint["global_step"]
    epoch = checkpoint["epoch"]
    best_valid_loss = checkpoint["best_valid_loss"]

    print("Checkpoint global_step: {}, epoch: {}, best_valid_loss: {}".format(global_step,
                                                                              epoch,
                                                                              best_valid_loss))

    # Do this again after loading checkpoint, because I am not sure if that is saved correctly
    # Weight tying / sharing between encoder and decoder RoBERTa part
    if do_tie_weights:
        print("Tying encoder decoder RoBERTa checkpoint weights!")
        base_model_prefix = vae_model.decoder.model.base_model_prefix
        tie_weights(vae_model.encoder.model, vae_model.decoder.model._modules[base_model_prefix], base_model_prefix)

    # Make all embedding spaces the same (encoder input, decoder input, decoder output)
    if do_tie_embeddings:
        print("Tying embedding spaces!")
        vae_model.tie_all_embeddings()

    if evaluation is True:
        print("Setting to eval mode.")
        vae_model.eval()

    return vae_model


def change_checkpoint_to_after_refactor(state_dict):
    """
    Before the refactor, the model looked slightly different. If you're trying
    to load a checkpoint from before the refactor it needs to be adjusted a bit.
    """

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "decoder.latent" in k:
            name = "latent_to_decoder." + k[8:]
        elif "encoder" in k[:8]:
            name = k.replace("encoder", "encoder.model", 1)
        elif "decoder" in k[:8]:
            name = k.replace("decoder", "decoder.model", 1)
        new_state_dict[name] = v
    return new_state_dict


def change_checkpoint_to_after_refactor_2(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "latent_to_decoder" in k[:17]:
            name = k.replace("latent_to_decoder", "decoder.latent_to_decoder")
        elif "decoder.decoder.latent_to_decoder." in k:
            name = k[8:]
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def save_checkpoint_model(vae_model, run_name, code_dir_path, global_step, best_valid_loss, epoch, config, best=False):
    """
    Save checkpoint for later use.
    """

    # Put in the name that it is a 'best' model
    if best:
        global_step = "best"

    # If just a regular checkpoint, remove the previous checkpoint
    if not best:
        for ckpt in os.listdir('{}/Runs/{}'.format(code_dir_path, run_name)):
            if ('best' not in ckpt) and ('checkpoint' in ckpt):
                print("Removing previously saved checkpoint: 'Runs/{}/{}'".format(run_name, ckpt))
                os.remove('{}/Runs/{}/{}'.format(code_dir_path, run_name, ckpt))

    print(
        "Saving checkpoint at '{}/Runs/{}/checkpoint-{}.pth'...".format(code_dir_path, run_name, global_step))

    # TODO: save scaler, scheduler, optimisers for continue training
    checkpoint = {
        'VAE_model_state_dict': vae_model.state_dict(),
        "config": config,
        # 'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        # 'scheduler_state_dict': scheduler.state_dict(),
        # 'scaler_state_dict': scaler.state_dict(),
        'best_valid_loss': best_valid_loss,
        'epoch': epoch,
    }

    torch.save(checkpoint, '{}/Runs/{}/checkpoint-{}.pth'.format(code_dir_path, run_name, global_step))


def add_remove_module_from_state_dict(state_dict, remove=True):
    """
    Adds or removes the 'module' string from keys in state dict that
    appears after saving in ddp mode.

    Args:
        state_dict: Dict[str, Tensor]
            parameter state dict
        remove: bool
            whether to remove (True) or add (False) the module string
    Returns:
        new_state_dict: Dict[str, Tensor]
            parameter state dict with module strings removed from
            or added from keys
    """

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if remove:
            name = k[7:]  # remove `module.`
        else:
            name = "module." + k
        new_state_dict[name] = v

    return new_state_dict


# ----------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------

def init_logging(vae_model, run_name, code_dir_path, wandb_project, config):
    """
    Initialise W&B logging.

    Args:
        vae_model: NewsVAE: the model object
        run_name: str: name of the run
        code_dir_path: str:
            Path to NewsVAE code dir
        wandb_project: str
            Name of the W&B project.
        config: Namespace: configuration to save
    """

    print("Initialising W&B logging...")
    wandb.init(name=run_name, project=wandb_project,
               dir='{}/Runs/{}/wandb'.format(code_dir_path, run_name), config=config)
    wandb.watch(vae_model)


def log_stats_epoch(stats, epoch, global_step, global_grad_step):
    """
    Log mean stats of epoch to W&B.

    Args:
        stats: defaultDict
            Nested dict containing all the statistics that are kept track of.
        epoch: int
        global_step: int
        global_grad_step: int
    """

    logs = {}
    for phase, phase_stats in stats[epoch].items():
        for stat_name, stat in phase_stats.items():
            log_name = "Epoch mean ({}) {}".format(phase, stat_name)
            logs[log_name] = np.mean(stat)

    logs['epoch'] = epoch
    logs['global step'] = global_step
    logs['global grad step'] = global_grad_step
    wandb.log(logs)


def log_losses_step(losses, phase, epoch, log_every_n_steps, global_step, global_grad_step):
    """
    Log losses of step to W&B.

    Args:
        losses: Dict[str, float]
            The dictionary holding the statistics of one step, such as the losses.
        phase: str
            Train or validation phase.
        epoch: int
        log_every_n_steps: int
            Every how many steps this function is called.
        global_step: int
        global_grad_step: int
        lr: float
            Current learning rate.
        beta: float
    """

    logs = {"Step log ({}) {}".format(phase, stat_name, log_every_n_steps): v for stat_name, v in
            losses.items()}
    logs["epoch"] = epoch
    logs["global step"] = global_step
    logs["global gradient step"] = global_grad_step
    wandb.log(logs)


def insert_stats(stats, new_stats, epoch, phase):
    """
    Add losses to the statistics.

    Args:
        stats: defaultDict
            Nested dictionary with all the statistics so far.
        new_stats: Dict[str, float]
            Dictionary with new stats to insert in all the stats.
        epoch: int
        phase: str
            Train or validation

    Returns:
        stats: defaultDict
            Nested dictionary with all the statistics plus the ones added.
    """

    for stat_name, value in new_stats.items():
        # Make a list to save values to if first iteration of epoch
        if type(stats[epoch][phase][stat_name]) != list:
            stats[epoch][phase][stat_name] = []
        if torch.is_tensor(value):
            stats[epoch][phase][stat_name].append(value.item())
        else:
            stats[epoch][phase][stat_name].append(value)

    return stats


# ----------------------------------------------------------------------------------------------------
# HELPER FUNCS
# ----------------------------------------------------------------------------------------------------
def get_code_dir():
    """
    Get the absolute path to the current working directory (NewsVAE directory).
    This differs per machine.

    Returns:
        code_dir_path: str
    """
    code_dir_path = os.path.dirname(os.path.realpath(__file__))

    return code_dir_path


def get_lr(scheduler):
    """
    Return the current learning rate, given the scheduler.

    Args:
        scheduler

    Returns:
        lr: float
            The current learning rate.
    """

    lr = scheduler.get_last_lr()[0]

    return lr


def print_stats(stats, epoch, phase, global_step, max_global_train_steps,
                global_grad_step, max_global_grad_train_steps,
                batch_i, phase_max_steps):
    """
    Print statistics to track process.

    Args:
        stats: defaultDict
            Nested dict with all stats in it.
        epoch: int
        phase: str
        global_step: int
        max_global_train_steps: int
        global_grad_step: int
        max_global_grad_train_steps: int
        batch_i: int
        phase_max_steps: int
        beta: float
        lr: float
    """
    print_string = "EPOCH {:4} | STEP {:5}/{:5} | GRAD STEP {:5}/{:5} | {:5} {:4}/{:4}".format(
        epoch, global_step + 1, max_global_train_steps, global_grad_step + 1,
        max_global_grad_train_steps, phase, batch_i + 1, phase_max_steps)

    print_string += "\n** GENERAL STATS"
    stat_dict = stats[epoch][phase]
    for s, v in stat_dict.items():
        if s not in ["alpha_MI", "beta_TC", "gamma_dim_KL", "alpha", "beta",
                     "gamma", "beta_KL", "KL", "TC", "MI", "dim_KL"]:
            print_string += " | {}: {:8.2f}".format(s, v[-1])

    # Beta-VAE
    if "beta_KL" in stat_dict:
        print_string += f"\n** BETA-VAE | beta {stat_dict['beta'][-1]:.2f} x KL {stat_dict['KL'][-1]:.2f} = {stat_dict['beta_KL'][-1]:.2f}"

    # Beta-TC-VAE
    elif "alpha_MI" in stat_dict:
        print_string += f"\n** BETA-TC-VAE | alpha {stat_dict['alpha'][-1]:.2f} x MI {stat_dict['MI'][-1]:.2f} = {stat_dict['alpha_MI'][-1]:.2f}"
        print_string += f" | beta {stat_dict['beta'][-1]:.2f} x TC {stat_dict['TC'][-1]:.2f} = {stat_dict['beta_TC'][-1]:.2f}"
        print_string += f" | gamma {stat_dict['gamma'][-1]:.2f} x Dim. KL {stat_dict['dim_KL'][-1]:.2f} = {stat_dict['gamma_dim_KL'][-1]:.2f}"

    print(print_string)


def get_number_of_params(model, trainable=True):
    """
    Count the number of (trainable) parameters in a model.

    Args:
        model: torch.nn.module
        trainable: bool:
            Whether to include trainable parameters only.
    Returns:
        n_params: int
    """

    if trainable:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        n_params = sum(p.numel() for p in model.parameters())

    return n_params


def print_platform_code_dir():
    """
    Print info on the platform, node and code working directory.
    """

    platform_name, node = get_platform()
    print("Detected platform: {} ({})".format(platform_name, node))
    print("Code directory absolute path: {}".format(get_code_dir()))


def get_available_devices():
    """
    Determine which devices are available: GPU(s) or CPU only.

    Return:
        device: torch.device
        device_ids: list[int]
            a list of available GPU ids
    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_ids = list(range(torch.cuda.device_count()))
        print('{} GPU(s) detected'.format(len(device_ids)))
    else:
        device = torch.device("cpu")
        device_ids = []
        print('No GPU, available. Only CPU.')
    return device, device_ids


def make_nested_dict():
    """
    A function to initialise a nested default dict.
    """
    return collections.defaultdict(make_nested_dict)


def get_ip():
    """
    Returns the current IP.
    """
    IP = socket.gethostbyname(socket.gethostname())
    return IP


def transfer_batch_to_device(batch, device_name="cuda:0"):
    """
    Transfer an input batch to a device.

    Args:
        device_name: str:
            The name of the device to transfer the data to
        batch: Dict[str, Tensor]
            The data dict to put on device
    """
    for k in batch.keys():
        batch[k] = batch[k].to(device_name)
    return batch


def get_platform():
    """
    Deduce which platform we're on (lisa or local machine).

    Returns:
        platform_name: str:
            Name of the platform (lisa or local)
        platform: ... TODO: not sure what this is again

    :return:
    """
    platform_name = 'lisa' if 'lisa' in platform.node() else 'local'
    return platform_name, platform.node()
