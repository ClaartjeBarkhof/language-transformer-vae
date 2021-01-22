import platform
import socket
import torch
import collections
import datetime
import os
import torch.backends.cudnn as cudnn
import numpy as np
import wandb


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

    datetime_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    run_name = "{}-run-{}".format(run_name_prefix, datetime_stamp)
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
# CHECKPOINTING
# ----------------------------------------------------------------------------------------------------

def load_from_checkpoint(vae_model, path, optimizer=None, scheduler=None, scaler=None,
                         world_master=True, ddp=False, use_amp=True, device_name="cuda:0"):

    vae_model = vae_model.cpu()

    # DETERMINE / CHECK PATH
    assert os.path.isfile(path), f"-> checkpoint file path ({path}) must exist for it to be loaded!"

    if world_master: print("Loading VAE_model, optimizer and scheduler from {}".format(path))

    # LOAD CHECKPOINT
    checkpoint = torch.load(path, map_location='cpu')

    # OPTIMIZER
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # SCHEDULER
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    parameter_state_dict = checkpoint["VAE_model_state_dict"]

    # MODEL
    if "module." in list(checkpoint["VAE_model_state_dict"].keys())[0] and not ddp:
        print("Removing module string from state dict from checkpoint")
        parameter_state_dict = add_remove_module_from_state_dict(parameter_state_dict, remove=True)

    elif "module." not in list(checkpoint["VAE_model_state_dict"].keys())[0] and ddp:
        print("Adding module string to state dict from checkpoint")
        parameter_state_dict = add_remove_module_from_state_dict(remove=False)

    # Adapt if checkpoint i from before refactor 1
    from_before_refactor = False
    for k in parameter_state_dict.keys():
        if "encoder.encoder" in k:
            from_before_refactor = True
            break

    if from_before_refactor:
        print("Changing checkpoint to match after refactor.")
        parameter_state_dict = change_checkpoint_to_after_refactor(parameter_state_dict)

    # refactor 2
    parameter_state_dict = change_checkpoint_to_after_refactor_2(parameter_state_dict)

    # in place procedure
    vae_model.load_state_dict(parameter_state_dict)
    vae_model = vae_model.to(device_name)

    # AMP SCALER
    # only load if using amp and a scaler is provided (when continue training from some point on).
    if scaler is not None and use_amp:
        if len(checkpoint["scaler_state_dict"]) == 0:
            print("--> Warning! You are trying to load weights into a scaler that comes from a "
                  "run with AMP disabled. Not loading any weights.")
        else:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # GLOBAL STEP, EPOCH, BEST VALIDATION LOSS
    global_step = checkpoint["global_step"]
    epoch = checkpoint["epoch"]
    best_valid_loss = checkpoint["best_valid_loss"]

    print("Checkpoint global_step: {}, epoch: {}, best_valid_loss: {}".format(global_step,
                                                                              epoch,
                                                                              best_valid_loss))

    return optimizer, scheduler, vae_model, scaler, global_step, epoch, best_valid_loss


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


def save_checkpoint_model(vae_model, optimizer, scheduler, scaler, run_name, code_dir_path,
                          global_step, best_valid_loss, epoch, best=False):
    """
    Save checkpoint for later use.

    Args:
        vae_model: nn.Module
        optimizer: torch.optim
        scheduler
        scaler
        run_name: str
        code_dir_path: str
        global_step: int
        best_valid_loss: float
        epoch: int
        best: bool
            Whether or not to save as a 'best' checkpoint or a regular checkpoint.
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

    checkpoint = {
        'VAE_model_state_dict': vae_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
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
            print(log_name)
            logs[log_name] = np.mean(stat)

    logs['epoch'] = epoch
    logs['global step'] = global_step
    logs['global grad step'] = global_grad_step
    wandb.log(logs)


def log_losses_step(losses, phase, epoch, log_every_n_steps, global_step, global_grad_step, lr, beta):
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
    logs["beta"] = beta
    logs["global step"] = global_step
    logs["global gradient step"] = global_grad_step
    logs["Step log ({}) learning rate".format(phase)] = lr
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
                batch_i, phase_max_steps, beta, lr):
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
    print_string = "EPOCH {:4} | STEP {:5}/{:5} | GRAD STEP {:5}/{:5} | {:5} {:6}/{:6}".format(
        epoch, global_step + 1, max_global_train_steps, global_grad_step + 1,
        max_global_grad_train_steps, phase, batch_i + 1, phase_max_steps)

    for s, v in stats[epoch][phase].items():
        print_string += " | {:8}: {:8.4f}".format(s, v[-1])

    print_string += " | Beta: {:.4f}".format(beta)
    print_string += " | LR: {:.6f}".format(lr)

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
