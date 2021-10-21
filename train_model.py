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
import os

def MSE(a,b):
    return torch.abs(a - b).pow(2).mean()

def Acc(a,b):
    return torch.sum(a==b).item() / a.numel()

def test_model(fiber, comp, tx, rx, N):
    '''
    Test model Acc
    '''
    k = int((config.Nch - 1)/2)  # number of central channel
    data_batch, symbol_batch, bit_batch = tx.data(batch=N)
    y = fiber(data_batch)        # y: batch x Nch x Nfft
    x = comp(y[:,k:k+1,:])       # x: batch x 1 x Nfft
    x = x.squeeze(1)
    bit_predict = rx.receiver(x, Nch=k)
    acc = Acc(bit_batch[:,k,:], bit_predict)
    return acc


def train(Epochs, batch, name, lr=0.0001, save_path='ckpt/',
 out_path='out/', test_num=100, power_range=[50,50],width=80,depth=3):
    '''
    Traing and save a model
    '''

    # Initializing the system
    k = int((config.Nch - 1)/2)  # number of central channel
    tx = Tx()                    # transmitter
    rx = Rx()                    # receiver
    fiber = Fiber(tx.lam_set,length=1e5,alphaB=0.2,n2=2.7e-20,disp=17,dz=100,Nch=config.Nch,generate_noise=True)


    # Load model or create model
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path) 

    model_file = save_path + name + '_best.pt'   # model path
    loss_file = save_path + name +'_losspath.pt' # loss path
    print_file = out_path + name + 'print.txt'
    try:
        comp = torch.load(model_file)['model']
        start_num = torch.load(model_file)['epoch'] + 1
    except:
        start_num = 0
        comp = Fiber(tx.lam_set[k:k+1],length=1e5,alphaB=-0.2,n2=-2.7e-20,disp=-17,dz=1e4,Nch=1,
        meta=name,is_trained=True,meta_width=width,meta_depth=depth)
    
    # use gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        comp = comp.cuda()
        fiber = fiber.cuda()

    # Training
    t0 = time.time()
    with open(print_file, 'a') as f:
        print('Training Start!', file=f) 
    optimizer = optim.Adam(params=comp.parameters(),lr=lr)
    train_loss_path = []

    for epoch in range(start_num, Epochs):

        # Generate data
        T1 = time.time()
        optimizer.zero_grad()

        # random power range
        tx.power = torch.ones(config.Nch)*power_range[0] + (power_range[1] - power_range[0]) * torch.rand(config.Nch)
        x,_,_ = tx.data(batch=batch)   # sample a batch   x: batch x Nch x Nfft
        T2 = time.time()
        
        # Propagation and Back propagation
        T3 = time.time()
        y = fiber(x)        # y: batch x Nch x Nfft
        z = comp(y[:,k:k+1,:])
        loss = MSE(z, x[:,k:k+1,:])
        loss.backward()
        optimizer.step()
        T4 = time.time()

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
    
    acc = test_model(fiber, comp, tx, rx, N=test_num)
    with open(print_file, 'a') as f:
        print('Current Acc: %g' % acc, file=f)
        print('Mission complete!', file=f)
    torch.save({'epoch':epoch, 'loss':loss.item(), 'model':comp, 'acc':acc},model_file)
    return
    
