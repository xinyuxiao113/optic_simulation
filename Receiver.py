import torch
import torch.nn as nn
import numpy as np
import config

class Rx(nn.Module):
    '''
    Receiver Module
    '''
    def __init__(self, Nbit=config.Nbit, Nsymb=config.Nsymb, Nt=config.Nt, Nch=config.Nch, sample_rate=config.sample_rate):
        super(Rx,self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
       ## filed parameter
        self.Nbit = Nbit              # number of bits
        self.Nsymb = Nsymb            # number of symbols
        self.Nt = Nt                  # number of samples / symbol
        self.Nch = Nch                # number of channels 
        self.Nfft = self.Nt * self.Nsymb      
        self.sample_rate = sample_rate
        self.modulation_formate = config.modulation_formate 

        self.symbol_rate = config.symbol_rate # symbol rate 10[Gb/s]
        self.power = config.power             # power of Tx [mW]
        self.lam   = config.lam               # cenrtal wavelength [nm]
        self.spac = config.channel_space      # channel spacing [nm]
        self.duty  = config.duty              # duty cycle    
        self.roll = config.roll               # pulse roll-off
        self.pulse = self.one_pulse()      # pulse shape

        self.fil_ker = self.generate_filter() # filter
        self.norm = torch.sum(self.pulse ** 2) 
        

        # symbol set
        if self.modulation_formate == '4QAM':
            self.symb = torch.complex(torch.zeros(4),torch.zeros(4))
            for i in range(4):
                x = int(i/2)
                y = i%2
                self.symb[i] = (2*x-1 + (2*y-1)*(1j))
            
            self.symb = self.symb.to(self.device)

        ## lambda set
        self.lam_set = torch.tensor([self.lam])  # channel wavelength set
        for i in range(1,self.Nch):
            self.lam_set = torch.cat([self.lam_set,torch.tensor([self.lam + self.spac * i])])
    
    def set_power(self,data):
        self.power = torch.tensor(data)

    def one_pulse(self):
        '''
        generate a pulse function: Raise Cosin
        '''
        nl = round(0.5 * (1 - self.roll) * self.duty * self.Nt)
        nr = round(self.duty * self.Nt) - nl
        ncos = torch.arange(nl, nr)
        elpulse = torch.zeros(2 * self.Nt)
        elpulse[self.Nt: self.Nt + nl] = 1
        hperiod = self.duty * self.Nt - 2 * nl
        elpulse[nl + self.Nt: nr + self.Nt] = 0.5*(1 + torch.cos(np.pi/(hperiod)*(ncos-nl+0.5)))
        elpulse[0:self.Nt] = torch.flip(elpulse[self.Nt:2*self.Nt+1],[0])
        h = int(self.Nt / self.sample_rate)
        elpulse = elpulse[::h]
        return elpulse + (0j)
    
    def generate_filter(self):
        '''
        Generate the filter kernel H: C^{ Nsymb x Nfft }
        '''
        Nt = self.sample_rate
        Nfft = self.Nsymb * Nt
        H = torch.zeros(self.Nsymb, Nfft) + (0j)

        H[0, Nfft-Nt:Nfft] = self.pulse[0:Nt].conj()
        H[0, 0:Nt] = self.pulse[Nt:2*Nt].conj()
        for k in range(1,self.Nsymb):
            nstart = (k-1)*Nt
            nend = (k+1)*Nt
            H[k,nstart:nend] =  self.pulse.conj()
        H = H + 0j
        H = H.to(self.device)
        return H


    def filter(self,x, Nch):
        '''
            translate the analog signal to a complex symbol sequence.
            x: (batch x Nch x Nfft) or (Nfft)
            fil_ker: Nsymb x Nfft
        '''

        return torch.tensordot(x, self.fil_ker, dims=([-1],[-1])) / self.norm / np.sqrt(self.power[Nch])
    

    def demaping(self,I):
        '''
            The inverse process of pulse shaping
            translate analog signal to a complex symbol stream.
            I : (batch x Nsymb) or Nsymb
            Use gaussian detection.(取离得最近的符号)
        '''
        # OOK
        if self.modulation_formate == 'OOK':
            return (torch.abs(I) > 0.5)*1
        
        # '4QAM'
        if self.modulation_formate == '4QAM':
            
            # Use BroadCasting 
            res = torch.abs(I.unsqueeze(-1) - self.symb)
            idx = torch.argmin(res,axis=-1)
            return self.symb[idx]

    def demodulation(self,symbol_stream):
        '''
        demodulation: OOK, QAM
        C^(batch x Nsymb) --> {0,1}^(batch x Nbit)
        '''
        if self.modulation_formate == 'OOK':
            return symbol_stream
        
        # '4QAM'
        if symbol_stream.dim() == 1:
            symbol_stream.unsqueeze(0)

        batch = symbol_stream.size(0)
        bit_stream = torch.zeros(batch, self.Nbit, device=self.device)
        if self.modulation_formate == '4QAM':
            bit_stream[:,0::2] = (symbol_stream.real + 1)/2
            bit_stream[:,1::2] = (symbol_stream.imag + 1)/2
            return bit_stream

    

    def receiver(self,x,Nch):
        '''
        reciever
        '''
        x_filtered = self.filter(x,Nch)
        x_demaped = self.demaping(x_filtered)
        x_bit = self.demodulation(x_demaped)
        return x_bit
    
    def show_symbol(self,I,symbol_stream,size=5):
        import matplotlib.pyplot as plt
        x = I.real
        y = I.imag
        symbol_set = [-1+1j, 1+1j, -1-1j, 1-1j]
        for s in symbol_set:
            idx = (symbol_stream == s).nonzero().view(-1)
            x_show = x[idx]
            y_show = y[idx]
            plt.scatter(x_show.data,y_show.data,s=size) 
        ax = plt.gca()
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["left"].set_position(("data", 0))



