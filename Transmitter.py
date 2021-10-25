import torch
import torch.nn as nn
import numpy as np
import config

class Tx(nn.Module):
    '''
    Transmmiter module
    '''
    def __init__(self):
        super(Tx,self).__init__()
        ## filed parameter
        self.Nbit = config.Nbit              # number of bits
        self.Nsymb = config.Nsymb            # number of symbols
        self.Nt = config.Nt                  # number of samples / symbol
        self.Nch = config.Nch                # number of channels 
        self.Nfft = self.Nt * self.Nsymb   
        self.modulation_formate = config.modulation_formate  

        self.symbol_rate = config.symbol_rate # symbol rate 10[Gb/s]
        self.power = config.power             # power of Tx [mW]
        self.lam   = config.lam               # cenrtal wavelength [nm]
        self.spac = config.channel_space      # channel spacing [nm]
        self.duty  = config.duty              # duty cycle    
        self.roll = config.roll               # pulse roll-off
        self.pulse = self.one_pulse()         # pulse shape
        

        ## lambda set
        self.lam_set = torch.tensor([]) 
        k = int((self.Nch - 1)/2) # central channels
        for i in range(-k,k+1):
            self.lam_set = torch.cat([self.lam_set,torch.tensor([self.lam + self.spac * i])])


    def set_power(self,data):
        self.power = torch.tensor(data)

    def pattern(self):
        '''
        generate a random 0-1 sequence
        '''
        return (torch.rand(self.Nbit)>0.5) * 1.0 
    

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
        return elpulse + (0j)

    def modulation(self,pat):
        '''
        digital modulation: OOK, QAM
        {0,1}^(Nbit) --> C^(Nsymb)
        '''
        # OOK modulation
        if self.modulation_formate == 'OOK':
            return pat

        # 4QAM modulation
        symbol_stream = torch.zeros(self.Nsymb) + (0j)
        if self.modulation_formate == '4QAM':
            for i in range(self.Nsymb):
                b1 = pat[2*i] 
                b2 = pat[2*i+1]
                symbol_stream[i] = b1*2 - 1 + (b2*2-1)*(1j)
            return symbol_stream

    def pulse_shaping(self, symbol_stream):
        '''
        translate a complex symbol stream to analog signal
        C^{Nsymb} --> C^{Nfft}
        '''
        Nt = self.Nt
        Nfft = self.Nsymb * Nt
        E = torch.zeros(Nfft) + (0j)
        E[Nfft-Nt:Nfft] = symbol_stream[0]*self.pulse[0:Nt]
        E[0:Nt] = symbol_stream[0]*self.pulse[Nt:2*Nt]
        for k in range(1,self.Nsymb):
            nstart = (k-1)*Nt
            nend = (k+1)*Nt
            E[nstart:nend] = E[nstart:nend] + symbol_stream[k]*self.pulse
        return E + 0j
    
    def wdm_signal_sample(self):
        '''
            sample a wdm signal
            Out put:
            E, symbol_stream, bit_stream
        '''
        Nfft = self.Nsymb*self.Nt
        Nch = self.Nch
        bit_stream = torch.zeros(Nch, self.Nbit)
        symbol_stream = torch.zeros(Nch, self.Nsymb)*(1j)
        E = torch.zeros(Nch,Nfft) + 0j
        for i in range(Nch):
            bit_stream[i] = self.pattern()
            symbol_stream[i] = self.modulation(bit_stream[i])
            E[i] = self.pulse_shaping(symbol_stream[i])*np.sqrt(self.power[i])
        return E,symbol_stream,bit_stream

    def data(self,batch):
        '''
            sample a batch of wdm signal
            data_batch: batch x Nch x Nfft
            Out put:
            data_batch, symbol_batch, bit_batch
        '''
        data_batch = torch.zeros(batch,self.Nch,self.Nfft) + (0j)
        symbol_batch = torch.zeros(batch,self.Nch,self.Nsymb) + (0j)
        bit_batch = torch.zeros(batch,self.Nch,self.Nbit)
        for i in range(batch):
            data_batch[i], symbol_batch[i], bit_batch[i] = self.wdm_signal_sample()
        if torch.cuda.is_available():
            data_batch = data_batch.cuda()
            symbol_batch = symbol_batch.cuda()
            bit_batch = bit_batch.cuda()
        return data_batch, symbol_batch, bit_batch