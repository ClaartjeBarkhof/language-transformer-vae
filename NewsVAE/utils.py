import platform


def get_platform():
    PLATFORM = 'local' if platform.node() == 'MacBook-Pro-van-Claartje.local' else 'lisa'
    return PLATFORM, platform.node()


def get_code_dir():
    if get_platform()[0] == 'local':
        CODE_DIR = '/Users/claartje/Dropbox (Persoonlijk)/Studie/Master AI/Thesis/code-thesis/NewsVAE/'
    else:
        CODE_DIR = '/home/cbarkhof/code-thesis/NewsVAE/'

    return CODE_DIR


def print_platform_codedir():
    platform, node = get_platform()
    print("Detected platform: {} ({})".format(platform, node))
    print("Code directory absolute path: {}".format(get_code_dir()))
