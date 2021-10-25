import torch

# General Constant
CLIGHT = 299792458

# Tx parameter
duty  = 1              # duty cycle   
roll = 0.2             # pulse roll-off 
channel_space = 0.4    # channel spacing [nm]
lam = 1550             # cenrtal wavelength [nm]          


# Signal parameter
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
symbol_rate = 56                    # [G Hz]    
noise_level = 0.002                 # noise level
Aeff = 80                           # effective area [um^2] (1 um = 1e-6 m)
slope = 0                           # slope [ps/nm^2/km] 
dphimax = 3E-3                      # maximum nonlinear phase rotation per step
dzmax   = 2E4                       # maximum SSFM step 


# meta net parameter
meta_width = 60
meta_depth = 2

