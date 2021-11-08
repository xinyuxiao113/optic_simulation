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
k = int((Nch - 1)/2)                 # number of central channel
if modulation_formate == '4QAM':
    Nsymb = round(Nbit/2)            # number of symbols
elif modulation_formate == 'OOK':
    Nsymb = Nbit
Nfft = Nsymb * Nt

lam_set = torch.tensor([]) 
for i in range(-k,k+1):
    lam_set = torch.cat([lam_set,torch.tensor([lam + channel_space * i])])


# Fiber parameters
EDFA = False
power_diverge = False
fiber_length = 1e5                  # fiber length [m]
span = 1                            # span number 
alphaB = 0.2                        # attenuation [dB/km]
n2 = 2.7e-20                        # nonlinear index [m^2/W]
disp = 17                           # dispersion [ps/nm/km] 
dz = 1e2                            # SSFM step size
power = torch.ones(3)*1.0           # power [mW]   train power: 50
symbol_rate = 10                    # [G Hz]    
noise_level = 2e-3                  # noise level
Aeff = 80                           # effective area [um^2] (1 um = 1e-6 m)
slope = 0                           # slope [ps/nm^2/km] 
dphimax = 3E-3                      # maximum nonlinear phase rotation per step
dzmax   = 2E4                       # maximum SSFM step 
generate_noise = 'n'              # noise type: 'n' or 'n*u' or False, default False
gerbio = alphaB*fiber_length/1e3    # EDFA 放大参数

# Trainning parameter
power_range = [50,50]
meta_width = 80
meta_depth = 3
Epochs = 600
batch = 64
lr = 0.001

# Testing parameters
test_num = 100

