import numpy as np
import matplotlib.pyplot as plt
from commonFunction import *

# Simulation parameters
nFFTSize = 64                               # FFT size
M_qam = 16                                  # M-QAM modulation order
nbBit_qam = int(np.log2(M_qam))             # number of bits per QAM symbol
nSymbol_OFDM = 2**12                        # number of OFDM symbols
nBit = nFFTSize * nbBit_qam * nSymbol_OFDM  # number of bits transmitted
L = nFFTSize // 4                           # Number of samples in the guard interval

print("="*60)
print("SIMULATION OFDM - CHAÎNE COMPLÈTE")
print("="*60)
print(f"FFT Size: {nFFTSize}")
print(f"Modulation: {M_qam}-QAM ({nbBit_qam} bits/symbole)")
print(f"Nombre de symboles OFDM: {nSymbol_OFDM}")
print(f"Longueur CP: {L}")
print(f"Nombre total de bits: {nBit}")
print("="*60)

## ---------------------------
## ÉMETTEUR ---------------------------##
print("\n📡 ÉMETTEUR")
print("-"*60)

## (1) Data generation and QAM mapping
tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize) 
print(f"✓ Génération de {len(tMsgBin)} bits aléatoires")
print(f"✓ Modulation QAM: {len(tX)} symboles")
print(f"✓ Matrice tXmat: {tXmat.shape}")

## (2) Add the Guard Interval and IFFT 
tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
print(f"✓ IFFT + ajout CP: {tx.shape}")

## (3) Upsample and apply the root raised-cosine filter 
rolloff = 0.3
samples_per_symbol = 8
tSignal, rrcosFilter = rrcos(tx, rolloff, samples_per_symbol)
print(f"✓ Sur-échantillonnage (×{samples_per_symbol}): {tSignal.shape}")

## (4) Compute PAPR 
power_signal = np.abs(tSignal)**2
peak_power = np.max(power_signal)
average_power = np.mean(power_signal)
papr_linear = peak_power / average_power
papr_db = 10 * np.log10(papr_linear)
print(f"✓ PAPR: {papr_db:.2f} dB (linéaire: {papr_linear:.2f})")

## ---------------------------
## CANAL AWGN ---------------------------##
print("\n📻 CANAL")
print("-"*60)
snr_db = 20  # SNR en dB
rSignal = chan_awgn(tSignal, snr_db)
print(f"✓ Ajout bruit AWGN (SNR = {snr_db} dB)")

## ---------------------------
## RÉCEPTEUR ---------------------------##
print("\n📥 RÉCEPTEUR")
print("-"*60)

## (5) Matched filter (Root Raised Cosine)
# Filtrage adapté
rSignal_filtered = np.convolve(rSignal, rrcosFilter, mode='same')
print(f"✓ Filtrage adapté RRC")

## (6) Downsampling
# Sous-échantillonnage
rx = rSignal_filtered[::samples_per_symbol]
rx = rx[:len(tx)]  # Ajuster la longueur
print(f"✓ Sous-échantillonnage (÷{samples_per_symbol}): {rx.shape}")

## (7) Remove CP and FFT
rxMat, rXmat = removeIGandFFT(rx, nSymbol_OFDM, nFFTSize, L)
print(f"✓ Suppression CP + FFT: {rXmat.shape}")

## (8) QAM Demapping
rX, rMsgBin = demapping2bit(rXmat, M_qam)
print(f"✓ Démodulation QAM: {len(rMsgBin)} bits")

## ---------------------------
## RÉSULTATS ---------------------------##
print("\n📊 RÉSULTATS")
print("="*60)

# Calcul du BER (Bit Error Rate)
bit_errors = np.sum(tMsgBin != rMsgBin)
ber = bit_errors / nBit
print(f"Nombre d'erreurs binaires: {bit_errors}")
print(f"BER (Bit Error Rate): {ber:.2e} ({ber*100:.4f}%)")

# Calcul de l'EVM
evm = calculateEvm(tX, rX)
print(f"EVM (Error Vector Magnitude): {evm:.2f}%")

# SNR estimé
if ber > 0:
    snr_est = 10 * np.log10(1 / (2 * ber * nbBit_qam))
    print(f"SNR estimé (théorique): {snr_est:.2f} dB")

print("="*60)
print("✅ Simulation terminée avec succès!")
print("="*60)
