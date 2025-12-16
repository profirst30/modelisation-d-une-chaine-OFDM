import numpy as np
import matplotlib.pyplot as plt
from commonFunction import *
from commpy.modulation import QAMModem
from scipy.signal import convolve

# Simulation parameters
nFFTSize = 64                               # FFT size
M_qam = 16                                  # M-QAM modulation order
nbBit_qam = int(np.log2(M_qam))             # number of bits per QAM symbol
nSymbol_OFDM = 2**12                        # number of OFDM symbols
nBit = nFFTSize * nbBit_qam * nSymbol_OFDM  # number of bits transmitted
L = nFFTSize // 4                           # Number of samples in the guard interval

## ---------------------------
## The transmitter ---------------------------##
## Data generation and qam mapping
tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize) 
    
print("(1) tMsgBin Size",np.shape(tMsgBin))
print("(2) tX Size",np.shape(tX))
print("(3) tXmat Size",np.shape(tXmat))

# Add the Guard Intervall and ifff 
tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
print("(4) txMat Size",np.shape(txMat))
print("(5) tx Size",np.shape(tx))


## Upsample and ppply the root raised-cosine filter 
rolloff = 0.3
samples_per_symbol = 8
tSignal, rrcosFilter = rrcos(tx, rolloff, samples_per_symbol)
print("(6) tSignal Size",np.shape(tSignal))

## Compute PAPR 

# Calculate linear PAPR and PAPR in dB
# PAPR = max(|x[n]|^2) / E[|x[n]|^2]
power_signal = np.abs(tSignal)**2
papr_linear = np.max(power_signal) / np.mean(power_signal)
papr_db = 10 * np.log10(papr_linear)

print(f"(7) PAPR (linéaire) = {papr_linear:.3f}")
print(f"(8) PAPR (dB)       = {papr_db:.3f} dB")


## Plot the spectrum of tSignal with fs = 1 
fs = 1
plotSpectrum(tSignal, fs)


