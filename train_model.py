import torch
import torch.optim as optim
import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import config
from Transmitter import Tx
from Receiver import Rx
from Fiber import Amplifier, Fiber, WSS
import os
import torch.nn as nn
from collections import OrderedDict


def MSE(a,b):
    return torch.abs(a - b).pow(2).mean()

def Acc(a,b):
    return torch.sum(a==b).item() / a.numel()

def test_model(channel_model, comp, tx, rx, N, power=50):
    '''
    Test model Acc
    '''
    k = int((config.Nch - 1)/2)  # number of central channel
    tx.power = torch.ones(config.Nch) * power
    rx.power = torch.ones(config.Nch) * power
    data_batch, symbol_batch, bit_batch = tx.data(batch=N)
    y = channel_model(data_batch)
    z = comp(y)       # x: batch x 1 x Nfft
    if z.shape[1] > 1:
        z = z[:,config.k,:]
    else:
        z = z[:,0,:]
    bit_predict = rx.receiver(z, Nch=k)
    acc = Acc(bit_batch[:,k,:], bit_predict)
    return acc


def train(model_name, Epochs=config.Epochs,batch=config.batch,lr=config.lr, power_range=config.power_range,power_diverge = config.power_diverge,
save_path='ckpt-set/', out_path='out-set/', test_num=config.test_num, width=config.meta_width,depth=config.meta_depth):
    '''
    Traing and save a model
    '''

    ## Initializing the system
    
    k = config.k              # number of central channel
    tx = config.tx            # transmitter
    rx = config.rx            # receiver
    # fiber channel
    channel_model = config.channel_model

    ## Load model or create model
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path) 
    
    os.system(f'cp config.py {save_path}config.py')

    model_file = save_path + model_name + '_best.pt'   # model path
    loss_file = save_path + model_name +'_losspath.pt' # loss path
    print_file = out_path + model_name + 'print.txt'
    try:
        comp = torch.load(model_file)['model']
        start_num = torch.load(model_file)['epoch'] + 1
    except:
        start_num = 0
        ## 如果采用放大器，就不需要在DBP处做alphaB的补偿
        def dbp_block():
            return Fiber(tx.lam_set[k:k+1],config.fiber_length,1e-8,-config.n2,-config.disp,dz=config.DBP_dz,Nch=1,
        meta=model_name,is_trained=True,meta_width=width,meta_depth=depth)

        # Sequential model
        comp = nn.Sequential(OrderedDict([(f'DBP Block {i}', dbp_block()) for i in range(config.span)]))
    
    ## use gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        comp = comp.cuda()
        channel_model = channel_model.cuda()

    ## Training
    t0 = time.time()
    with open(print_file, 'a') as f:
        print('Training Start!', file=f) 
    optimizer = optim.Adam(params=comp.parameters(),lr=lr)
    train_loss_path = []

    for epoch in range(start_num, Epochs):

        # Generate data
        optimizer.zero_grad()

        # random power range
        if power_diverge:
            tx.power = torch.ones(config.Nch)*power_range[0] + (power_range[1] - power_range[0]) * torch.rand(config.Nch)
        else:
            tx.power = torch.ones(config.Nch)*power_range[0] + (power_range[1] - power_range[0]) * torch.rand(1)
        
        x,_,_ = tx.data(batch=batch)   # sample a batch   x: batch x Nch x Nfft

        # Propagation and Back propagation
        y = channel_model(x)
        z = comp(y[:,k:k+1,:])
        loss = MSE(z, x[:,k:k+1,:])
        loss.backward()
        optimizer.step()


        with open(print_file, 'a') as f:
            print('epoch %d/%d: loss: %g' % (epoch, Epochs, loss.item()), file=f)
            f.flush()
        train_loss_path.append(loss.item())
        
    # 合并 loss path
    train_loss_path = torch.tensor(train_loss_path)
    try:
        loss_path = torch.load(loss_file)['train loss']
        train_loss_path = torch.cat([loss_path, train_loss_path])
    except:
        pass
    
    torch.save({'train loss':train_loss_path},loss_file)

    t1 = time.time()
    with open(print_file, 'a') as f:
        print('Training done! Time cost: %g' % (t1-t0), file=f)
        print('Testing!', file=f)
    
    acc = test_model(channel_model, comp, tx, rx, N=test_num)
    with open(print_file, 'a') as f:
        print('Current Acc: %g' % acc, file=f)
        print('Mission complete!', file=f)
    torch.save({'epoch':epoch, 'loss':loss.item(), 'model':comp, 'acc':acc},model_file)
    return


def get_model(name, model_path):
    train_model_names = ['NN-DBP','Meta-1','Meta-2','Meta-3']
    wss = WSS()
    k = config.k

    if config.EDFA:
        alpha_dbp = 1e-8
    else:
        alpha_dbp = - config.alphaB

    if name == 'no comp':
        return wss
    elif name == 'DO-DBP':
        comp = Fiber(config.lam_set[k:k+1], config.fiber_length, alpha_dbp, n2=0, disp=-config.disp, dz=config.fiber_length, Nch=1)
        model = nn.Sequential(OrderedDict([(f'CD {i}', comp) for i in range(config.span)]))
        return nn.Sequential(wss, model)
    elif name == 'SC-DBP':
        comp = Fiber(config.lam_set[k:k+1], config.fiber_length, alpha_dbp, n2=-config.n2, disp=-config.disp, dz=config.dz, Nch=1)
        model = nn.Sequential(OrderedDict([(f'SC-DBP {i}', comp) for i in range(config.span)]))
        return nn.Sequential(wss, model)
    elif name == 'full DBP':
        comp = Fiber(config.lam_set, config.fiber_length, alpha_dbp, n2=-config.n2, disp=-config.disp, dz=1e4, Nch=config.Nch)
        model = nn.Sequential(OrderedDict([(f'full DBP {i}', comp) for i in range(config.span)]))
        return nn.Sequential(model)
    elif name in train_model_names:
        model = torch.load(model_path + name + '_best.pt',map_location=torch.device('cpu'))['model']
        return nn.Sequential(wss, model)
    else:
        print(f'No model names {name}!')
        raise(ValueError)
    
