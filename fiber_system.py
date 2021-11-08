import torch
import torch.optim as optim
import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import config
from Transmitter import Tx
from Receiver import Rx
from Fiber import Amplifier, Fiber
import os
import torch.nn as nn
from collections import OrderedDict

# channel model
tx = Tx()                 # transmitter
rx = Rx()                 # receiver


if config.EDFA:
    fiber = Fiber(tx.lam_set, config.fiber_length, config.alphaB, config.n2, config.disp, config.dz, config.Nch, generate_noise=False) 
    amplifier = Amplifier(config.fiber_length, config.gerbio, config.generate_noise, config.noise_level)
else:
    fiber = Fiber(tx.lam_set,config.fiber_length, config.alphaB, config.n2, config.disp, config.dz, config.Nch, config.generate_noise) 
    amplifier = Amplifier(config.fiber_length, 0, False, config.noise_level)

fiber_block = nn.Sequential(OrderedDict([('fiber',fiber),('amp',amplifier)]))
channel_model = nn.Sequential(OrderedDict([(f'Block {i}',fiber_block) for i in range(config.span)]))

