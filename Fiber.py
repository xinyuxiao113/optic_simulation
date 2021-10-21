import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from torch.fft import fft, ifft

    
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


class Fiber(nn.Module):

    def __init__(self,lam_set=None,length=1e5,alphaB=0.2,n2=2.7e-20,disp=17,dz=100,Nch=1,
    generate_noise=False,is_trained=False,meta='0',meta_width=60, meta_depth=2):
        super(Fiber,self).__init__()
        ## field parameter
        self.Nsymb = config.Nsymb  # number ofs symbols
        self.Nt = config.Nt        # number of samples every symbol
        self.Nfft = config.Nfft    # dims of input
        self.Nch = Nch             # number of channels
        
        self.generate_noise = generate_noise         # generate noise or not 
        self.noise_level = config.noise_level        # noise level
        self.meta = meta                             # meta or not
        self.is_trained = is_trained                 # trian or not
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        ## fiber parameters 
        self.length = length         # length [m]
        self.alphaB = alphaB         # attenuation [dB/km]
        self.aeff = 80               # effective area [um^2] (1 um = 1e-6 m)
        self.n2 = n2                 # nonlinear index [m^2/W]
        self.lamb  = 1550            # wavelength [nm]
        self.lam_set = lam_set
        self.disp = disp             # dispersion [ps/nm/km] 
        self.slope = 0               # slope [ps/nm^2/km] 
        self.dphimax = 3E-3          # maximum nonlinear phase rotation per step
        self.dzmax   = 2E4           # maximum SSFM step 
        
        
        ############################# CONVERSIONS #############################
        #  nonlinear index [1/mW/m]      gam = 2*pi*n2/(lambda * Aeff)
        self.gam = torch.ones(self.Nch, self.Nfft)
        for kch in range(self.Nch):
            self.gam[kch] = 2*np.pi*self.n2/(self.lam_set[kch] * self.aeff)*1e18 * torch.ones(self.Nfft)
        self.L_nl = 1/(self.gam[0,0] * config.power[0])

        # alphaB: [dB/km] alphaB = 10*log_{10}(P_{1000}/P_{0})   alphain [m^-1] = log(P1/P0)
        self.alphalin = self.alphaB * (np.log(10)*1e-4)

        # Frequancies
        self.FN = torch.fft.fftshift(torch.arange(self.Nfft) - 0.5*self.Nfft)/self.Nsymb
        
        # calculate effective length [m]
        if self.alphalin == 0:
            self.Leff = self.length
        else:
            self.Leff = (1 - np.exp(-self.alphalin * self.length)) / self.alphalin
        
        # calculate beta1 beta2 beta3       
        maxl = max(lam_set)              # max lambda [nm]
        minl = min(lam_set)              # min lambda [nm]
        lamc = 2*maxl*minl/(maxl+minl)   # central lambda  [nm]

        # beta2 [ns^2/m]
        b20 = - self.lamb**2 / (2 * np.pi * config.CLIGHT ) * self.disp * 1e-6 

        # beta3 [ns^3/m]
        b30 = (self.lamb/(2 * np.pi * config.CLIGHT ))**2 * (2*self.lamb*self.disp+self.lamb**2*self.slope)*1e-6 

        #  Domega_ik: [1/ns]. "i" -> at ch. i, "0" -> at lambda, "c" -> at lamc
        Domega_i0 = 2*np.pi*config.CLIGHT * (1/self.lam_set - 1/self.lamb);    
        Domega_ic = 2*np.pi*config.CLIGHT * (1/self.lam_set - 1/lamc);    
        Domega_c0 = 2*np.pi*config.CLIGHT * (1/lamc - 1/self.lamb);  

        b1 = b20 * Domega_ic + 0.5*b30*(Domega_i0**2 - Domega_c0**2)  
        beta1 = b1
        beta2 = b20 + b30 * Domega_i0
        Dch = self.disp + self.slope * (self.lam_set - self.lamb)

        self.L_disp = 1/(config.symbol_rate**2 * abs(self.lamb**2/2/np.pi/config.CLIGHT*Dch*1e-6))
        self.betat = torch.zeros(self.Nch, self.Nfft)
        omega =  2 * np.pi * config.symbol_rate * self.FN

        for kch in range(self.Nch):
            self.betat[kch,:] = omega*beta1[kch] + 0.5*omega**2 * beta2[kch] + omega**3 * b30 / 6
        
        self.dz = dz                                         # step size
        self.H = torch.exp(- (1j) * self.betat * self.dz)    # dispersion operator
        self.K = int(length / dz)                            # number of segments per span

        # transform to the right device: cpu or gpu
        self.gam = self.gam.to(self.device)
        self.H = self.H.to(self.device)

        if self.is_trained:
            if self.meta == 'scale':
                self.H_trained = nn.ModuleList([Meta_block(width=meta_width, depth=meta_depth, meta_type='filter',Hi=self.H) for i in range(self.K)])
                self.scales = nn.ModuleList([Meta_block(width=meta_width, depth=meta_depth, meta_type='scale',Hi=torch.ones(1)) for i in range(self.K)])
            elif self.meta == 'plus':
                self.H_trained = nn.ModuleList([Meta_block(width=meta_width, depth=meta_depth, meta_type='filter',Hi=self.H) for i in range(self.K)])
                self.other_channel = nn.ModuleList([Meta_block(width=meta_width, depth=meta_depth, meta_type='scale',Hi=torch.zeros(1)) for i in range(self.K)])
            elif self.meta == 'scale+plus':
                self.H_trained = nn.ModuleList([Meta_block(width=meta_width,depth=meta_depth,meta_type='filter',Hi=self.H) for i in range(self.K)])
                self.scales = nn.ModuleList([Meta_block(width=meta_width,depth=meta_depth,meta_type='scale',Hi=torch.ones(1)) for i in range(self.K)])
                self.other_channel = nn.ModuleList([Meta_block(width=meta_width,depth=meta_depth,meta_type='scale',Hi=torch.zeros(1)) for i in range(self.K)])
            elif self.meta == 'normal':
                self.H_trained = nn.ParameterList([nn.Parameter(self.H) for i in range(self.K)])     # trained filter in linear step
                self.scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for i in range(self.K)]) # trained scales in nl step
            elif self.meta == 'shared':
                self.H_trained = Meta_block(width=meta_width,depth=meta_depth,meta_type='filter',Hi=self.H)
                self.other_channel = Meta_block(width=meta_width,depth=meta_depth,meta_type='scale',Hi=torch.zeros(1))
            else:
                raise(ValueError)
        else:
            pass

    def lin_step(self,u,step=0):
        '''
        Linear step
        '''
        if self.is_trained:
            if (self.meta == 'scale') or (self.meta == 'plus') or (self.meta == 'scale+plus'):
                u = ifft(fft(u,dim=-1) * self.H_trained[step](u), dim=-1)
            elif self.meta == 'normal':
                u = ifft(fft(u,dim=-1) * self.H_trained[step], dim=-1)
            elif self.meta == 'shared':
                u = ifft(fft(u,dim=-1) * self.H_trained(u), dim=-1)
        else:
            u = ifft(fft(u,dim=-1) * self.H, dim=-1)
        return u
    
    def nl_step(self,u,step=0):
        '''
        Nonlinear step
        '''
        power = abs(u)**2
        power = 2*torch.sum(power,dim=-2).unsqueeze(-2) - power
        leff = (1 - np.exp(- self.alphalin * self.dz)) / self.alphalin

        if self.is_trained:
            if self.meta == 'scale':
                u = u * torch.exp(-(1j) * self.scales[step](u) * self.gam * power * leff)
            elif self.meta == 'plus':
                u = u * torch.exp(-(1j) *  self.gam * (power + self.other_channel[step](u)) * leff)
            elif self.meta == 'scale+plus':
                u = u * torch.exp(-(1j) * self.scales[step](u) *  self.gam * (power + self.other_channel[step](u)) * leff)
            elif self.meta == 'normal':
                u = u * torch.exp(-(1j) * self.scales[step] * self.gam * power * leff)  
            elif self.meta == 'shared':
                u = u * torch.exp(-(1j) *  self.gam * (power + self.other_channel(u)) * leff)
        else:
            u = u * torch.exp(-(1j) * self.gam * power * leff)
        return u
    

    def forward(self,u,step=0):
        '''
        SSFM algorithm
        '''
        for step in range(self.K):
            u = self.nl_step(u,step=step)
            u = self.lin_step(u,step=step)
            u = u * np.exp(-0.5 * self.alphalin * self.dz)
            if self.generate_noise:
                noise = self.noise_level/np.sqrt(2) * torch.randn(u.shape) + self.noise_level * torch.randn(u.shape)*(1j)/np.sqrt(2)
                noise = noise.to(self.device)
                u = u + noise
        return u

class Amplifier(nn.Module):

    def __init__(self,gerbio):
        super(Amplifier,self).__init__()
        self.gain = np.sqrt(10**(0.1*gerbio))
    
    def forward(self,u):
        return u*self.gain