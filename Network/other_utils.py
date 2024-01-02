import os
import sys

import numpy as np
import torch
import random
from pathlib import Path
import glob
import re


join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir


def path_str(url):
    """
    :Introduction
        this function is used to get the same path format under different systems
        like: root_head/a/v/, root_head is empty in linux
    :param str:
    :return:
    """
    if 'win32' in sys.platform:
        url = url.replace('\\', '/')
    return url


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def maybe_makedirs(directory):
    """
    :Introduction
        to write
    :param directory:
    :return:
    """
    directory = os.path.abspath(directory)
    if 'win32' in sys.platform:
        directory = path_str(directory)
    splits = directory.split("/")[1:]
    sys_head = directory.split("/")[0]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join(sys_head+"/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join(sys_head+"/", *splits[:i+1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']