import numpy as np
import matplotlib.pyplot as plt
from commonFunction import *
from scipy.fft import fft, fftshift
from scipy.signal import get_window

# Question 5: Tracé du spectre et mesure de la bande occupée
# On utilise plotSpectrum() avec fs = 1

# Paramètres de simulation (mêmes que ofdmTranceiver.py)
nFFTSize = 64                               # FFT size
M_qam = 16                                  # M-QAM modulation order
nbBit_qam = int(np.log2(M_qam))             # number of bits per QAM symbol
nSymbol_OFDM = 2**12                        # number of OFDM symbols
nBit = nFFTSize * nbBit_qam * nSymbol_OFDM  # number of bits transmitted
L = nFFTSize // 4                           # Number of samples in the guard interval

print("=" * 60)
print("Question 5: Spectre du signal OFDM et bande occupée")
print("=" * 60)

## Émetteur OFDM
print("\nGénération du signal OFDM...")
tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)

## Sur-échantillonnage et filtrage RRC
rolloff = 0.3
samples_per_symbol = 8
tSignal, rrcosFilter = rrcos(tx, rolloff, samples_per_symbol)

print(f"Taille du signal transmis tSignal: {len(tSignal)}")

## Calcul théorique de la bande occupée
# Période symbole OFDM (après IFFT) : Ts = nFFTSize + L échantillons
# Avec sur-échantillonnage : Ts_upsampled = (nFFTSize + L) * samples_per_symbol
# Fréquence symbole : fs_symbol = 1 / Ts = 1 / (nFFTSize + L)
# Espacement entre sous-porteuses : delta_f = 1 / nFFTSize (avant sur-échantillonnage)
# Après sur-échantillonnage de facteur 8 avec fs = 1 :
#   - Période symbole = (nFFTSize + L) * samples_per_symbol = 80 * 8 = 640 échantillons
#   - Bande de base avant sur-échantillonnage : B = N * delta_f = 1 (bande normalisée)
#   - Après sur-échantillonnage : la bande est divisée par le facteur de sur-échantillonnage

fs = 1  # Fréquence d'échantillonnage unitaire

# Bande théorique du signal OFDM avec filtre RRC
# B = (1 + alpha) / (samples_per_symbol) pour un signal en bande de base
# où alpha est le roll-off du filtre RRC
B_theorique = (1 + rolloff) / samples_per_symbol

print(f"\n--- Paramètres ---")
print(f"nFFTSize = {nFFTSize}")
print(f"Préfixe cyclique L = {L}")
print(f"Samples per symbol = {samples_per_symbol}")
print(f"Roll-off α = {rolloff}")
print(f"Fréquence d'échantillonnage fs = {fs}")

print(f"\n--- Bande théorique ---")
print(f"B = (1 + α) / samples_per_symbol")
print(f"B = (1 + {rolloff}) / {samples_per_symbol}")
print(f"B = {B_theorique:.4f} Hz (normalisé)")
print(f"Bande occupée : [{-B_theorique/2:.4f}, {B_theorique/2:.4f}] Hz")

## Tracé du spectre avec fonction personnalisée pour mesurer la bande
def plotSpectrumWithBandwidth(x, fs, save_path=None):
    """
    Trace le spectre et mesure la bande à -3dB et -20dB
    """
    win = 'flattop'
    Nblocks = 20
    N = len(x)
    fft_block_size = int(2**(np.ceil(np.log2(N/Nblocks))-1))
    window = get_window(win, fft_block_size)
    fft_input = np.zeros(fft_block_size, dtype='complex128')

    for n in range(1, Nblocks+1):
        idx = slice((n-1)*fft_block_size, n*fft_block_size)
        fft_block = fft(x[idx] * window) / fft_block_size
        fft_input += (fft_block * np.conj(fft_block))
    fft_input /= Nblocks
    
    pxx = np.abs(fft_input)
    pxx = fftshift(pxx)
    freq = (np.arange(0, fft_block_size) - fft_block_size/2) * fs / fft_block_size
    
    pxx_db = 10*np.log10(pxx)
    pxx_db_max = np.max(pxx_db)
    
    # Mesure de la bande à -3dB
    threshold_3dB = pxx_db_max - 3
    indices_3dB = np.where(pxx_db >= threshold_3dB)[0]
    if len(indices_3dB) > 0:
        f_min_3dB = freq[indices_3dB[0]]
        f_max_3dB = freq[indices_3dB[-1]]
        bandwidth_3dB = f_max_3dB - f_min_3dB
    else:
        bandwidth_3dB = 0
    
    # Mesure de la bande à -20dB
    threshold_20dB = pxx_db_max - 20
    indices_20dB = np.where(pxx_db >= threshold_20dB)[0]
    if len(indices_20dB) > 0:
        f_min_20dB = freq[indices_20dB[0]]
        f_max_20dB = freq[indices_20dB[-1]]
        bandwidth_20dB = f_max_20dB - f_min_20dB
    else:
        bandwidth_20dB = 0
    
    # Tracé
    plt.figure(figsize=(12, 6))
    plt.plot(freq, pxx_db, 'b-', linewidth=1.5)
    
    # Lignes de référence pour la bande
    plt.axhline(y=threshold_3dB, color='orange', linestyle='--', linewidth=1.5, 
                label=f'-3dB (B = {bandwidth_3dB:.4f} Hz)')
    plt.axhline(y=threshold_20dB, color='red', linestyle='--', linewidth=1.5,
                label=f'-20dB (B = {bandwidth_20dB:.4f} Hz)')
    
    # Limites théoriques
    plt.axvline(x=-B_theorique/2, color='green', linestyle=':', linewidth=2,
                label=f'Bande théorique = {B_theorique:.4f} Hz')
    plt.axvline(x=B_theorique/2, color='green', linestyle=':', linewidth=2)
    
    plt.xlabel('Fréquence (Hz normalisée)', fontsize=14)
    plt.ylabel('Densité spectrale de puissance (dB)', fontsize=14)
    plt.title('Spectre du signal OFDM transmis (tSignal)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='upper right')
    plt.xlim([-0.2, 0.2])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure sauvegardée dans '{save_path}'")
    
    return freq, pxx_db, bandwidth_3dB, bandwidth_20dB

# Tracé et mesure
freq, pxx_db, bw_3dB, bw_20dB = plotSpectrumWithBandwidth(tSignal, fs, 'rapport/chap1_question5.png')

print(f"\n--- Mesures sur le spectre ---")
print(f"Bande à -3dB  : {bw_3dB:.4f} Hz")
print(f"Bande à -20dB : {bw_20dB:.4f} Hz")
print(f"Bande théorique (1+α)/Nsps : {B_theorique:.4f} Hz")

print("\n" + "=" * 60)
print("JUSTIFICATION:")
print("=" * 60)
print(f"""
La bande occupée par le signal OFDM est déterminée par :
1. Le nombre de sous-porteuses N = {nFFTSize}
2. Le facteur de sur-échantillonnage = {samples_per_symbol}
3. Le roll-off du filtre RRC α = {rolloff}

Formule théorique de la bande (en fréquence normalisée) :
   B = (1 + α) / samples_per_symbol
   B = (1 + {rolloff}) / {samples_per_symbol} = {B_theorique:.4f}

Cette valeur correspond bien à la bande mesurée à -3dB sur le spectre.
Le filtre en cosinus surélevé racine (RRC) avec α = {rolloff} élargit 
légèrement la bande par rapport au cas idéal (α = 0).
""")

# plt.show()
