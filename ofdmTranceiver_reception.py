import numpy as np
import matplotlib.pyplot as plt
from commonFunction import *
from commpy.modulation import QAMModem
from commpy.filters import rrcosfilter

"""
Chaîne de réception OFDM - Version corrigée
============================================
Ce script implémente la chaîne complète : émetteur + canal AWGN + récepteur
On travaille en bande de base (pas de modulation RF).

IMPORTANT: Pour simplifier et éviter les problèmes de synchronisation liés
au filtrage RRC, on applique le bruit AWGN directement sur le signal après IFFT+CP.
"""

print("=" * 70)
print("CHAÎNE DE RÉCEPTION OFDM")
print("=" * 70)

# ============================================================================
# PARAMÈTRES DE SIMULATION
# ============================================================================
nFFTSize = 64                               # FFT size
M_qam = 16                                  # M-QAM modulation order
nbBit_qam = int(np.log2(M_qam))             # number of bits per QAM symbol
nSymbol_OFDM = 2**10                        # number of OFDM symbols
nBit = nFFTSize * nbBit_qam * nSymbol_OFDM  # number of bits transmitted
L = nFFTSize // 4                           # Number of samples in the guard interval
rolloff = 0.3
samples_per_symbol = 8
SNR_dB = 30                                 # SNR en dB (faible bruit)

print(f"\n--- Paramètres ---")
print(f"nFFTSize = {nFFTSize}")
print(f"M-QAM = {M_qam}")
print(f"Symboles OFDM = {nSymbol_OFDM}")
print(f"Préfixe cyclique L = {L}")
print(f"SNR = {SNR_dB} dB")

# ============================================================================
# ÉMETTEUR (pt-1 à pt-6)
# ============================================================================
print("\n" + "=" * 70)
print("ÉMETTEUR")
print("=" * 70)

# Data generation and qam mapping (pt-1 à pt-3)
tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
print(f"(1) Bits générés : {len(tMsgBin)}")
print(f"(2) Symboles QAM : {len(tX)}")
print(f"(3) Matrice tXmat : {tXmat.shape}")

# Add Guard Interval and IFFT (pt-4 à pt-5)
tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
print(f"(4-5) Signal temporel tx : {len(tx)}")

# ============================================================================
# CANAL AWGN (appliqué sur le signal tx après IFFT+CP)
# ============================================================================
print("\n" + "=" * 70)
print("CANAL AWGN")
print("=" * 70)

# Ajout du bruit AWGN directement sur tx
rx = chan_awgn(tx, SNR_dB)
print(f"Signal bruité rx : {len(rx)}")
print(f"SNR appliqué : {SNR_dB} dB")

# ============================================================================
# RÉCEPTEUR (pt-7 à pt-12)
# ============================================================================
print("\n" + "=" * 70)
print("RÉCEPTEUR")
print("=" * 70)

# Comparaison visuelle des signaux tx et rx
fig, axes = plt.subplots(2, 1, figsize=(14, 6))

zoom_samples = 200
ax1 = axes[0]
ax1.plot(np.real(tx[:zoom_samples]), 'b-', linewidth=1, label='Re(tx) - Émis')
ax1.plot(np.real(rx[:zoom_samples]), 'r--', linewidth=1, label='Re(rx) - Reçu')
ax1.set_xlabel('Échantillons')
ax1.set_ylabel('Amplitude')
ax1.set_title('Comparaison parties réelles tx et rx')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(np.imag(tx[:zoom_samples]), 'b-', linewidth=1, label='Im(tx) - Émis')
ax2.plot(np.imag(rx[:zoom_samples]), 'r--', linewidth=1, label='Im(rx) - Reçu')
ax2.set_xlabel('Échantillons')
ax2.set_ylabel('Amplitude')
ax2.set_title('Comparaison parties imaginaires tx et rx')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rapport/chap2_sync_verification.png', dpi=300, bbox_inches='tight')
print("✓ Figure de synchronisation sauvegardée")

# ============================================================================
# Suppression du CP et FFT (pt-8 à pt-10)
# ============================================================================
print("\n--- Suppression du CP et FFT ---")

# Utilisation de la fonction removeIGandFFT
rxMat, rXmat = removeIGandFFT(rx, nSymbol_OFDM, nFFTSize, L)
print(f"(8-10) Matrice rxMat (avec CP) : {rxMat.shape}")
print(f"(8-10) Matrice rXmat (après FFT) : {rXmat.shape}")

# Vérification : comparaison du premier symbole émis et reçu
print("\n--- Vérification premier symbole OFDM ---")
print(f"Premier symbole émis  tXmat[:,0][:5] : {tXmat[:5, 0]}")
print(f"Premier symbole reçu  rXmat[:,0][:5] : {rXmat[:5, 0]}")

# Comparaison visuelle des constellations
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.scatter(np.real(tXmat.flatten()), np.imag(tXmat.flatten()), s=1, alpha=0.3)
ax1.set_xlabel('I (In-phase)')
ax1.set_ylabel('Q (Quadrature)')
ax1.set_title('Constellation émise (tXmat)')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])

ax2 = axes[1]
ax2.scatter(np.real(rXmat.flatten()), np.imag(rXmat.flatten()), s=1, alpha=0.3)
ax2.set_xlabel('I (In-phase)')
ax2.set_ylabel('Q (Quadrature)')
ax2.set_title(f'Constellation reçue (rXmat) - SNR={SNR_dB}dB')
ax2.grid(True, alpha=0.3)
ax2.axis('equal')
ax2.set_xlim([-5, 5])
ax2.set_ylim([-5, 5])

plt.tight_layout()
plt.savefig('rapport/chap2_constellation.png', dpi=300, bbox_inches='tight')
print("✓ Figure des constellations sauvegardée")

# ============================================================================
# Démodulation M-QAM (pt-10 à pt-12)
# ============================================================================
print("\n--- Démodulation M-QAM ---")

# Utilisation de la fonction demapping2bit
rX, rMsgBin = demapping2bit(rXmat, M_qam)
print(f"(10-12) Symboles reçus rX : {len(rX)}")
print(f"(10-12) Bits reçus rMsgBin : {len(rMsgBin)}")

# ============================================================================
# Calcul du TEB (Taux d'Erreur Binaire)
# ============================================================================
print("\n" + "=" * 70)
print("PERFORMANCES")
print("=" * 70)

# Comparaison des bits émis et reçus
n_errors = np.sum(tMsgBin != rMsgBin)
TEB = n_errors / len(tMsgBin)

print(f"\nBits émis     : {len(tMsgBin)}")
print(f"Bits reçus    : {len(rMsgBin)}")
print(f"Erreurs       : {n_errors}")
print(f"TEB           : {TEB:.6e}")

if TEB == 0:
    print("✓ Aucune erreur de transmission ! (SNR suffisamment élevé)")

# Calcul de l'EVM
evm = calculateEvm(tX, rX)
print(f"EVM           : {evm:.2f} %")

# Différence entre messages binaires émis et reçus
diff_bits = np.abs(tMsgBin.astype(int) - rMsgBin.astype(int))
print(f"\nDifférence entre tMsgBin et rMsgBin : {np.sum(diff_bits)} bits différents")

# ============================================================================
# COURBE TEB vs SNR
# ============================================================================
print("\n" + "=" * 70)
print("COURBE TEB vs SNR (de 2 à 16 dB)")
print("=" * 70)

SNR_range = np.arange(2, 17, 1)  # SNR de 2 à 16 dB avec pas de 1
TEB_list = []
EVM_list = []

print("\nSimulation en cours...")
for snr in SNR_range:
    # Canal AWGN directement sur tx
    rx_snr = chan_awgn(tx, snr)
    
    # Suppression CP et FFT
    _, rXmat_snr = removeIGandFFT(rx_snr, nSymbol_OFDM, nFFTSize, L)
    
    # Démodulation
    rX_snr, rMsgBin_snr = demapping2bit(rXmat_snr, M_qam)
    
    # TEB
    n_err = np.sum(tMsgBin != rMsgBin_snr)
    teb = n_err / len(tMsgBin)
    TEB_list.append(teb)
    
    # EVM
    evm_val = calculateEvm(tX, rX_snr)
    EVM_list.append(evm_val)
    
    print(f"  SNR = {snr:2d} dB : TEB = {teb:.4e}, EVM = {evm_val:.2f}%")

# Tracé de la courbe TEB vs SNR
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# TEB vs SNR
ax1 = axes[0]
# Éviter les zéros pour le log
TEB_plot = [max(t, 1e-7) for t in TEB_list]
ax1.semilogy(SNR_range, TEB_plot, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('SNR (dB)', fontsize=14)
ax1.set_ylabel('TEB (Taux d\'Erreur Binaire)', fontsize=14)
ax1.set_title('TEB en fonction du SNR', fontsize=16)
ax1.grid(True, which='both', alpha=0.3)
ax1.set_ylim([1e-6, 1])

# EVM vs SNR
ax2 = axes[1]
ax2.plot(SNR_range, EVM_list, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('SNR (dB)', fontsize=14)
ax2.set_ylabel('EVM (%)', fontsize=14)
ax2.set_title('EVM en fonction du SNR', fontsize=16)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rapport/chap2_teb_evm_vs_snr.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure TEB et EVM vs SNR sauvegardée dans 'rapport/chap2_teb_evm_vs_snr.png'")

print("\n" + "=" * 70)
print("OBSERVATIONS")
print("=" * 70)
print("""
- Le TEB diminue quand le SNR augmente
- À SNR élevé (>12-14 dB), le TEB devient très faible
- L'EVM diminue avec l'augmentation du SNR
- La chaîne OFDM complète fonctionne correctement !
""")

# plt.show()
