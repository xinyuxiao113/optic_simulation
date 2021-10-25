import torch
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F
import numpy as np
import config
from torch.fft import fft, ifft

def complex_leakyRelu(input):
    return F.leaky_relu(input.real,negative_slope=0.01) + F.leaky_relu(input.imag,negative_slope=0.01)*(1j)


def apply_complex(fr, fi, input, dtype = torch.complex64):
    '''
    operation in complex form
    (fr + i fr) (x + iy) = fr(x) - fi(y) + [fr(x) + fi(y)]i
    '''
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


class complex_conv1d(nn.Module):
    '''
    complex con1d mapping
    y = conv1d(W, x) + b
    W \in C^{}
    '''
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 'same', padding_mode='circular',
                 dilation=1, groups=1, bias=True):
        super(complex_conv1d, self).__init__()
        self.conv_r = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,padding_mode=padding_mode)
        self.conv_i = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,padding_mode=padding_mode)
        
    def forward(self,input):    
        return apply_complex(self.conv_r, self.conv_i, input)

    
class complex_linear(nn.Module):
    '''
    Complex linear mapping:
    y = W x + b
    W \in C^{n x m}
    b \in C^{m}
    near zero initilization
    '''
    def __init__(self,input_dim,output_dim):
        super(complex_linear,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        w = torch.randn(self.output_dim, self.input_dim)/100
        v = torch.randn(self.output_dim, self.input_dim)/100
        b0 = torch.randn(self.output_dim)/100
        b1 = torch.randn(self.output_dim)/100
        self.weight = nn.Parameter(torch.complex(w,v))
        self.bias = nn.Parameter(torch.complex(b0,b1))
    
    def forward(self,x):
        return F.linear(x,self.weight,self.bias)


class FNN(nn.Module):
    '''
    Fully connected complex network:
    width: a positive integer
    '''
    def __init__(self, input_features,out_features, width=config.meta_width, depth=config.meta_depth, init_value=None, to_real=False):
        super(FNN,self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.width = width
        self.depth= depth
        self.init_value = init_value.to(self.device)
        self.to_real = to_real
        self.fc0 = complex_linear(input_features, self.width)
        self.net = nn.ModuleList([complex_linear(self.width,self.width) for i in range(self.depth)])
        self.fc1 = complex_linear(self.width,out_features)
        self.activation = complex_leakyRelu

    def forward(self,u):
        u = self.activation(self.fc0(u))
        for net in self.net:
            u = self.activation(net(u))
        u = self.fc1(u)
        if self.to_real:
            u = torch.abs(u)**2
        return u + self.init_value
        #return u

class RFNN(nn.Module):
    '''
    Fully connected complex network:
    width: a positive integer
    '''
    def __init__(self, input_features,out_features, width=config.meta_width, depth=config.meta_depth, init_value=None, to_real=False):
        super(RFNN,self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.width = width
        self.depth= depth
        self.init_value = init_value.to(self.device)
        self.to_real = to_real
        self.fc0 = complex_linear(input_features, self.width)
        self.net = nn.ModuleList([complex_linear(self.width,self.width) for i in range(self.depth)])
        self.fc1 = complex_linear(self.width,out_features)
        self.activation = complex_leakyRelu

    def forward(self,u):
        u = self.activation(self.fc0(u))
        for net in self.net:
            u = self.activation(net(u))
        u = self.fc1(u)
        if self.to_real:
            u = torch.abs(u)**2
        return u + self.init_value


class CNN(nn.Module):
    '''
    Two layer complex network:
    width: a positive integer
    meta_type: 'filter' or 'scale'

    '''
    def __init__(self, input_features,out_features, width=config.meta_width, depth=config.meta_depth, init_value=None, to_real=False):
        super(CNN,self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.width = width
        self.depth= depth
        self.init_value = init_value.to(self.device)
        self.to_real = to_real
        self.fc0 = complex_linear(input_features, self.width)
        self.net = nn.ModuleList([complex_linear(self.width,self.width) for i in range(self.depth)])
        self.fc1 = complex_linear(self.width,out_features)
        self.activation = complex_leakyRelu

    def forward(self,u):
        u = self.activation(self.fc0(u))
        for net in self.net:
            u = self.activation(net(u))
        u = self.fc1(u)
        if self.to_real:
            u = torch.abs(u)**2
        return u + self.init_value
        #return u


class RCNN(nn.Module):
    '''
    Two layer complex network:
    width: a positive integer
    meta_type: 'filter' or 'scale'

    '''
    def __init__(self, input_features,out_features, width=config.meta_width, depth=config.meta_depth, init_value=None, to_real=False):
        super(RCNN,self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.width = width
        self.depth= depth
        self.init_value = init_value.to(self.device)
        self.to_real = to_real
        self.fc0 = complex_linear(input_features, self.width)
        self.net = nn.ModuleList([complex_linear(self.width,self.width) for i in range(self.depth)])
        self.fc1 = complex_linear(self.width,out_features)
        self.activation = complex_leakyRelu

    def forward(self,u):
        u = self.activation(self.fc0(u))
        for net in self.net:
            u = self.activation(net(u))
        u = self.fc1(u)
        if self.to_real:
            u = torch.abs(u)**2
        return u + self.init_value
        #return u

Meta_block = FNN
# Meta_block = CNN
# Meta_block = RCNN