import pathlib
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter

from model import ImageSiren

torch.manual_seed(2)


# Compare different actiavtion fucntions 


init_functions = {
    "ones": torch.nn.init.ones_,
    "eye": torch.nn.init.eye_,
    "default": partial(torch.nn.init.kaiming_uniform, a=5 ** (1/2)),
    "paper": None,
}

#for fname, func in init_functions.items():
