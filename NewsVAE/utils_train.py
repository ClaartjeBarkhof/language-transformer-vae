import platform
import socket
import torch
import collections
import datetime
import os
import wandb
import pathlib
import torch.backends.cudnn as cudnn


def set_verbosity(device_name):
    """
    Determine whether to do prints for this device.

    Args:
        device_name: str:
            A device name that can be used with .to(device)
    Returns:
        verbose: bool:
            Whether or not to be verbose for this machine.
    """

    # Only be verbose if GPU rank 0 or CPU training
    verbose = True if "cpu" or "0" in device_name else False
    return verbose


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
        torch.cuda.set_device(device_rank)
        cudnn.benchmark = True  # optimise backend algo
        device_name = f"cuda:{device_rank}"
    # CPU
    else:
        device_name = "cpu"
    return device_name


def init_logging(vae_model, run_name, args):
    """
    Initialise W&B logging.

    Args:
        vae_model: NewsVAE: the model object
        run_name: str: name of the run
        args: Namespace: configuration to save
    """

    print("Initialising W&B logging...")
    wandb.init(name=run_name, project=args.wandb_project,
               dir='{}Runs/{}/wandb'.format(args.prefix_NewsVAE_path, run_name), config=args)
    wandb.watch(vae_model)


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

    os.makedirs('{}Runs/{}'.format(code_path, run_name), exist_ok=True)
    os.makedirs('{}Runs/{}/wandb'.format(code_path, run_name), exist_ok=True)


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


def get_code_dir():
    """
    Get the absolute path to the current working directory (NewsVAE directory).
    This differs per machine.

    Returns:
        code_dir_path: str
    """
    code_dir_path = pathlib.Path().absolute()

    return code_dir_path


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
