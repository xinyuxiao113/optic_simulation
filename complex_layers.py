import torch
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F
import numpy as np
import config
from torch.fft import fft, ifft


def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class complex_conv1d(nn.Module):
    '''
    complex con1d mapping
    y = conv1d(W, x) + b
    W \in C^{}
    '''
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0, padding_mode='circular',
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


class Meta_block(nn.Module):
    '''
    Two layer complex network:
    width: a positive integer
    meta_type: 'filter' or 'scale'

    '''
    def __init__(self, meta_type='filter', width=config.meta_width, depth=config.meta_depth, Hi=None):
        super(Meta_block,self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.meta_type = meta_type
        self.width = width
        self.depth= depth
        self.Hi = Hi.to(self.device)
        self.fc0 = complex_linear(config.Nfft, self.width)
        self.net = nn.ModuleList([complex_linear(self.width,self.width) for i in range(self.depth)])
        if meta_type == 'filter':
            self.fc1 = complex_linear(self.width,config.Nfft)
        else:
            self.fc1 = complex_linear(self.width,1)
    
    def activation(self,x):
        return F.leaky_relu(x.real,negative_slope=0.01) + F.leaky_relu(x.imag,negative_slope=0.01)*(1j)

    def forward(self,u):
        u = self.activation(self.fc0(u))
        for net in self.net:
            u = self.activation(net(u))
        u = self.fc1(u)
        if self.meta_type == 'scale':
            u = torch.abs(u)**2
        
        return u + self.Hi
        #return u