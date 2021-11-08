import torch
from torch.cuda import init
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F
import numpy as np
from complexPyTorch.complexLayers import ComplexLinear
import config
from torch.fft import fft, ifft
from complex_layers import FNN,CNN,complex_linear


class Fiber(nn.Module):
    '''
    Fiber model. SSFM Algorithm.
    '''
    def __init__(self,lam_set=None,length=1e5,alphaB=0.2,n2=2.7e-20,disp=17,dz=100,Nch=1,
    generate_noise=False,noise_level=config.noise_level,is_trained=False,meta=False,meta_width=60, meta_depth=2):
        super(Fiber,self).__init__()
        ## field parameter
        self.Nsymb = config.Nsymb  # number ofs symbols
        self.Nt = config.Nt        # number of samples every symbol
        self.Nfft = config.Nfft    # dims of input
        self.Nch = Nch             # number of channels
        
        self.generate_noise = generate_noise         # generate noise or not 
        self.noise_level = noise_level        # noise level
        self.meta = meta                             # meta or not
        self.is_trained = is_trained                 # trian or not
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        ## fiber parameters 
        self.length = length         # length [m]
        self.alphaB = alphaB         # attenuation [dB/km]
        self.aeff = config.Aeff      # effective area [um^2] (1 um = 1e-6 m)
        self.n2 = n2                 # nonlinear index [m^2/W]
        self.lamb  = config.lam           # wavelength [nm]
        self.lam_set = lam_set
        self.disp = disp              # dispersion [ps/nm/km] 
        self.slope = config.slope     # slope [ps/nm^2/km] 
        self.dphimax = config.dphimax # maximum nonlinear phase rotation per step
        self.dzmax   = config.dzmax   # maximum SSFM step 
        
        
        ############################# CONVERSIONS #############################
        #  nonlinear index [1/mW/m]      gam = 2*pi*n2/(lambda * Aeff)
        self.gam = torch.ones(self.Nch, self.Nfft)
        for kch in range(self.Nch):
            self.gam[kch] = 2*np.pi*self.n2/(self.lam_set[kch] * self.aeff)*1e18 * torch.ones(self.Nfft)
        self.L_nl = 1/(self.gam[0,0] * config.power[0])

        # alphaB: [dB/km] alphaB = 10*log_{10}(P_{1000}/P_{0})   alphain [m^-1] = log(P1/P0)
        self.alphalin = self.alphaB * (np.log(10)*1e-4)

        # Frequancies
        self.FN = torch.fft.fftshift(torch.arange(self.Nfft) - 0.5*self.Nfft) / self.Nsymb
        
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
            if self.meta == 'Meta-1' :
                self.H_trained = nn.ModuleList([FNN(self.Nfft, self.Nfft, meta_width, meta_depth,init_value=self.H) for i in range(self.K)])
                self.other_channel = nn.ModuleList([FNN(self.Nfft,1,meta_width, meta_depth,to_real=True,init_value=torch.ones(1)) for i in range(self.K)])
            elif self.meta == 'Meta-2':
                self.H_trained = nn.ModuleList([FNN(self.Nfft, self.Nfft, meta_width, meta_depth,init_value=self.H) for i in range(self.K)])
                self.scales = nn.ModuleList([FNN(self.Nfft,1,meta_width, meta_depth,to_real=True,init_value=torch.ones(1)) for i in range(self.K)])
            elif self.meta == 'Meta-12':
                self.H_trained = nn.ModuleList([FNN(self.Nfft, self.Nfft, meta_width, meta_depth,init_value=self.H) for i in range(self.K)])
                self.scales = nn.ModuleList([FNN(self.Nfft,1,meta_width, meta_depth,to_real=True,init_value=torch.ones(1)) for i in range(self.K)])
                self.other_channel = nn.ModuleList([FNN(self.Nfft,1,meta_width, meta_depth,to_real=True,init_value=torch.ones(1)) for i in range(self.K)])
            elif self.meta == 'NN-DBP':
                self.H_trained = nn.ParameterList([nn.Parameter(self.H) for i in range(self.K)])     # trained filter in linear step
                self.scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for i in range(self.K)]) # trained scales in nl step
            elif self.meta == 'Meta-3':
                self.H_trained = FNN(self.Nfft, self.Nfft, meta_width, meta_depth,init_value=self.H)
                self.other_channel = FNN(self.Nfft,1,meta_width, meta_depth,to_real=True,init_value=torch.ones(1))
            elif self.meta == 'RFNN':
                self.H_trained = nn.ModuleList([FNN(2 * self.Nfft, self.Nfft, meta_width, meta_depth,init_value=self.H) for i in range(self.K)])
                self.scales = nn.ModuleList([FNN(2 * self.Nfft,1,meta_width, meta_depth,to_real=True,init_value=torch.ones(1)) for i in range(self.K)])
            else:
                print(f'No such meta type named {self.meta}')
                raise(ValueError)
        else:
            pass

    def lin_step(self,u,v,step=0):
        '''
        Linear step
        u: curent state
        v: current state + history state
        '''
        if self.is_trained:
            '''
            Trained DBP
            '''
            if (self.meta == 'Meta-1') or (self.meta == 'Meta-2') or (self.meta == 'Meta-12') or (self.meta == 'RFNN'):
                u = ifft(fft(u,dim=-1) * self.H_trained[step](v), dim=-1)
            elif self.meta == 'NN-DBP':
                u = ifft(fft(u,dim=-1) * self.H_trained[step], dim=-1)
            elif self.meta == 'Meta-3':
                u = ifft(fft(u,dim=-1) * self.H_trained(v), dim=-1)
            else:
                print(f'No such meta type named {self.meta}')
                raise(ValueError)
        else:
            '''
            Normal SSFM linear step
            '''
            u = ifft(fft(u,dim=-1) * self.H, dim=-1)
        return u
    
    def nl_step(self,u,v,step=0):
        '''
        Nonlinear step
        u: curent state
        v: current state + history state
        '''
        power = abs(u)**2
        power = 2*torch.sum(power,dim=-2).unsqueeze(-2) - power
        leff = (1 - np.exp(- self.alphalin * self.dz)) / self.alphalin

        if self.is_trained:
            '''
            Trained DBP
            '''
            if self.meta == 'Meta-2':
                u = u * torch.exp(-(1j) * self.scales[step](v) * self.gam * power * leff)
            elif self.meta == 'Meta-1':
                u = u * torch.exp(-(1j) *  self.gam * (power + self.other_channel[step](v)) * leff)
            elif self.meta == 'Meta-12':
                u = u * torch.exp(-(1j) * self.scales[step](v) *  self.gam * (power + self.other_channel[step](v)) * leff)
            elif self.meta == 'NN-DBP':
                u = u * torch.exp(-(1j) * self.scales[step] * self.gam * power * leff)  
            elif self.meta == 'Meta-3':
                u = u * torch.exp(-(1j) *  self.gam * (power + self.other_channel(v)) * leff)
            else:
                print(f'No such meta type named {self.meta}')
                raise(ValueError)
        else:
            '''
            Normal SSFM nonlinear step
            '''
            u = u * torch.exp(-(1j) * self.gam * power * leff)
        return u
    

    def forward(self,u,step=0):
        '''
        SSFM algorithm
        '''
        temp = torch.zeros_like(u) + 0j
        for step in range(self.K):

            if self.meta == 'RFNN':
                v = torch.cat([u,temp],axis=-1)
            else:
                v = u

            # 记录 u
            temp = u

            u = self.nl_step(u,v,step=step)
            u = self.lin_step(u,v,step=step)
            u = u * np.exp(-0.5 * self.alphalin * self.dz)

            if self.generate_noise == 'n':
                noise = 1e-2* self.dz * self.noise_level / np.sqrt(2) * (torch.randn(u.shape) + torch.randn(u.shape)*(1j))
                noise = noise.to(self.device)
                u = u + noise
            elif self.generate_noise == 'n*u':
                noise = 1e-2* self.dz * self.noise_level / np.sqrt(2) * (torch.randn(u.shape) + torch.randn(u.shape)*(1j))
                noise = noise.to(self.device)
                u = u + noise * u
            elif self.generate_noise == False:
                pass
            else:
                print(f'No such noise type named {self.generate_noise}')
                raise(ValueError)
        return u


class Amplifier(nn.Module):

    def __init__(self,length, gerbio, generate_noise, noise_level):
        super(Amplifier,self).__init__()
        self.gain = np.sqrt(10**(0.1*gerbio))
        self.generate_noise = generate_noise
        self.noise_level = noise_level
        self.length = length
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def forward(self,u):
        if self.generate_noise == 'n':
                noise = 1e-2* self.length * self.noise_level / np.sqrt(2) * (torch.randn(u.shape) + torch.randn(u.shape)*(1j))
                noise = noise.to(self.device)
                u = u + noise
        elif self.generate_noise == 'n*u':
            noise = 1e-2* self.length * self.noise_level / np.sqrt(2) * (torch.randn(u.shape) + torch.randn(u.shape)*(1j))
            noise = noise.to(self.device)
            u = u + noise * u
        elif self.generate_noise == False:
            noise = 0
        else:
            print(f'No such noise type named {self.generate_noise}')
            raise(ValueError)
            
        return u*self.gain + noise

class WSS(nn.Module):
    def __init__(self):
        super(WSS,self).__init__()
    
    def forward(self,u):
        return torch.index_select(u, dim=-2, index=torch.tensor([config.k]))
