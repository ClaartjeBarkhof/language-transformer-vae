import platform


def get_platform():
    PLATFORM = 'local' if platform.node() == 'MacBook-Pro-van-Claartje.local' else 'lisa'
    return PLATFORM, platform.node()


def get_code_dir():
    if get_platform() == 'local':
        CODE_DIR = '/Users/claartje/Dropbox (Persoonlijk)/Studie/Master AI/Thesis/code-thesis/NewsVAE/'
    else:
        CODE_DIR = '/home/cbarkhof/code-thesis/NewsVAE/'
    return CODE_DIR
