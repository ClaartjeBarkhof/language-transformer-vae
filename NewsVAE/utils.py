import platform
import socket
import torch
import collections


def get_platform():
    PLATFORM = 'lisa' if 'lisa' in platform.node() else 'local'
    return PLATFORM, platform.node()


def get_code_dir():
    if get_platform()[0] == 'local':
        CODE_DIR = '/Users/claartje/Dropbox (Persoonlijk)/Studie/Master AI/Thesis/code-thesis/NewsVAE/'
    else:
        CODE_DIR = '/home/cbarkhof/code-thesis/NewsVAE/'

    return CODE_DIR


def get_number_of_params(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_platform_codedir():
    platform, node = get_platform()
    print("Detected platform: {} ({})".format(platform, node))
    print("Code directory absolute path: {}".format(get_code_dir()))

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_ids = list(range(torch.cuda.device_count()))
        print('{} GPU(s) detected'.format(len(device_ids)))
    else:
        device = torch.device("cpu")
        device_ids = []
        print('No GPU. switching to CPU')
    return device, device_ids


def make_nested_dict():
    return collections.defaultdict(make_nested_dict)


def get_ip():
    IP = socket.gethostbyname(socket.gethostname())
    return IP
