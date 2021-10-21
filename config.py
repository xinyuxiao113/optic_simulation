import torch

# General Constant
CLIGHT = 299792458

# signal parameter
modulation_formate = '4QAM'          # '4QAM' or 'OOK'
Nbit = 256                           # number of bits
Nt = 8                               # number of samples every symbol
Nch = 3                              # number of channels
if modulation_formate == '4QAM':
    Nsymb = round(Nbit/2)            # number of symbols
elif modulation_formate == 'OOK':
    Nsymb = Nbit
Nfft = Nsymb * Nt


# Fiber parameters
power = torch.tensor([50,50,50])    # power [mW]   train power: 50
span = 1                            # span number 
symbol_rate = 10
noise_level = 0.002                 # noise level


# meta net parameter
meta_width = 60
meta_depth = 2

