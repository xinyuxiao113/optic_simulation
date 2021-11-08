## 导入模型和设置
import os
model_path = 'ckpt-set/ckpt-n/'
os.system(f'cp {model_path}config.py config.py')
import config
import torch
import numpy as np
import matplotlib.pyplot as plt
import config
from fiber_system import tx,rx,channel_model
from train_model import test_model, get_model
fig_path =  '/Users/xinyu/Desktop/WDM_Code/report/img/expriment/'  # 储存结果的路径


############################ setting parameters ############################
test_power = 50

############################ Initializing the system ############################
k = config.k              # number of central channel
tx.set_power([test_power]*config.Nch)
rx.set_power([test_power]*config.Nch)
comp = {}
z = {}
I = {}

############################ load model     ############################
model_names = ['no comp','full DBP', 'DO-DBP', 'SC-DBP','NN-DBP','Meta-1','Meta-2','Meta-3']

for name in model_names:
    comp[name] = get_model(name, model_path)



# 计算BER
acc = {}
for name in model_names:
    acc[name] = test_model(channel_model, comp[name], tx,rx,N=100,power=1)

for key in acc.keys():
    print('%10s  &   %g \\\\' % (key, acc[key]))

