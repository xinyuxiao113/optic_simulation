import torch
import torch.optim as optim
import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import config
from Transmitter import Tx
from Receiver import Rx
from Fiber import Fiber,Amplifier
from train_model import train, test_model

# model name: 'normal', 'scale', 'plus', 'scale+plus'
name = 'plus'
save_path = f'ckpt-set/ckpt-debug/'
out_path = f'out-set/out-debug/'
train(Epochs=100,batch=64,lr=0.001,name=name,save_path=save_path,out_path=out_path,power_range=[50,60],width=40,depth=2)