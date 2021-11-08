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

# model name: 'Meta-1' (plus term), 'Meta-2' (scale), 'Meta-3'(shared), NN-DBP
save_path = 'ckpt-set/ckpt-EDFA/'
out_path = 'out-set/out-EDFA/'
train(Epochs=600,batch=64,lr=1e-3,model_name='NN-DBP',save_path=save_path,out_path=out_path,power_range=[1,1],width=80,depth=3)
train(Epochs=600,batch=64,lr=1e-3,model_name='Meta-1',save_path=save_path,out_path=out_path,power_range=[1,1],width=80,depth=3)
train(Epochs=600,batch=64,lr=1e-3,model_name='Meta-2',save_path=save_path,out_path=out_path,power_range=[1,1],width=80,depth=3)
train(Epochs=600,batch=64,lr=1e-3,model_name='Meta-3',save_path=save_path,out_path=out_path,power_range=[1,1],width=80,depth=3)