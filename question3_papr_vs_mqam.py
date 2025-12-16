import numpy as np
import matplotlib.pyplot as plt
from commonFunction import *

# Question 3: Variation du PAPR en fonction de l'ordre de modulation M-QAM
# N = 64 fixe, M varie dans {4, 16, 64, 256, 1024} (QAM carrées uniquement)
# Note: commpy.QAMModem ne supporte que les QAM carrées (M = 4, 16, 64, 256, 1024...)

# Paramètres de simulation fixes
nFFTSize = 64                       # FFT size fixe
nSymbol_OFDM = 2**12                # number of OFDM symbols
L = nFFTSize // 4                   # Guard interval
rolloff = 0.3
samples_per_symbol = 8

# Liste des ordres de modulation M-QAM à tester (QAM carrées uniquement)
# M doit être une puissance paire de 2 : 4, 16, 64, 256, 1024
M_qam_list = [4, 16, 64, 256, 1024]

# Listes pour stocker les résultats
papr_linear_list = []

print("=" * 60)
print("Question 3: PAPR en fonction de l'ordre de modulation M-QAM")
print(f"(N = {nFFTSize} fixe)")
print("=" * 60)

for M_qam in M_qam_list:
    print(f"\n--- M_qam = {M_qam} ---")
    
    # Nombre de bits par symbole QAM
    nbBit_qam = int(np.log2(M_qam))
    
    # Calcul du nombre de bits
    nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
    
    # Data generation and qam mapping
    tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
    
    # Add the Guard Interval and IFFT
    tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
    
    # Upsample and apply the root raised-cosine filter
    tSignal, rrcosFilter = rrcos(tx, rolloff, samples_per_symbol)
    
    # Calculate linear PAPR
    power_signal = np.abs(tSignal)**2
    papr_linear = np.max(power_signal) / np.mean(power_signal)
    papr_db = 10 * np.log10(papr_linear)
    
    # Stocker les résultats
    papr_linear_list.append(papr_linear)
    
    print(f"  Bits/symbole = {nbBit_qam}")
    print(f"  PAPR (linéaire) = {papr_linear:.3f}")
    print(f"  PAPR (dB) = {papr_db:.3f} dB")

print("\n" + "=" * 60)
print("Résumé des résultats:")
print("=" * 60)
print(f"{'M':<8} {'Bits/symb':<12} {'PAPR lin':<12} {'PAPR (dB)':<12}")
print("-" * 60)
for i, M in enumerate(M_qam_list):
    nbBit = int(np.log2(M))
    papr_db = 10 * np.log10(papr_linear_list[i])
    print(f"{M:<8} {nbBit:<12} {papr_linear_list[i]:<12.3f} {papr_db:<12.3f}")

# Tracer le graphique
plt.figure(figsize=(10, 6))
plt.plot(M_qam_list, papr_linear_list, 'o-', linewidth=2, markersize=8, color='blue', label='PAPR simulé')

plt.xlabel('Ordre de modulation M', fontsize=14)
plt.ylabel('PAPR (linéaire)', fontsize=14)
plt.title(r'PAPR linéaire en fonction de M (avec $N_{FFT} = 64$)', fontsize=16)
plt.xscale('log', base=2)  # Échelle log base 2 pour M
plt.xticks(M_qam_list, [str(m) for m in M_qam_list])
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# Sauvegarder la figure
plt.savefig('rapport/chap1_question3.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure sauvegardée dans 'rapport/chap1_question3.png'")

# plt.show()
