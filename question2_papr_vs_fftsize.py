import numpy as np
import matplotlib.pyplot as plt
from commonFunction import *

# Question 2: Variation du PAPR en fonction de la taille de la FFT
# On trace PAPR linéaire en fonction de sqrt(3*N)

# Paramètres de simulation
M_qam = 16                          # M-QAM modulation order
nbBit_qam = int(np.log2(M_qam))     # number of bits per QAM symbol
nSymbol_OFDM = 2**12                # number of OFDM symbols
rolloff = 0.3
samples_per_symbol = 8

# Liste des tailles de FFT à tester
nFFTSize_list = [16, 32, 64, 128, 256, 512]

# Listes pour stocker les résultats
papr_linear_list = []
sqrt_3N_list = []

print("=" * 60)
print("Question 2: PAPR en fonction de la taille de FFT")
print("=" * 60)

for nFFTSize in nFFTSize_list:
    print(f"\n--- nFFTSize = {nFFTSize} ---")
    
    # Calcul du nombre de bits
    nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
    L = nFFTSize // 4  # Guard interval
    
    # Data generation and qam mapping
    tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
    
    # Add the Guard Interval and IFFT
    tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
    
    # Upsample and apply the root raised-cosine filter
    tSignal, rrcosFilter = rrcos(tx, rolloff, samples_per_symbol)
    
    # Calculate linear PAPR
    power_signal = np.abs(tSignal)**2
    papr_linear = np.max(power_signal) / np.mean(power_signal)
    
    # Stocker les résultats
    papr_linear_list.append(papr_linear)
    sqrt_3N = np.sqrt(3 * nFFTSize)
    sqrt_3N_list.append(sqrt_3N)
    
    print(f"  sqrt(3*N) = {sqrt_3N:.3f}")
    print(f"  PAPR (linéaire) = {papr_linear:.3f}")
    print(f"  Ratio PAPR/sqrt(3*N) = {papr_linear/sqrt_3N:.3f}")

print("\n" + "=" * 60)
print("Résumé des résultats:")
print("=" * 60)
print(f"{'N':<8} {'sqrt(3*N)':<12} {'PAPR lin':<12} {'Ratio':<12}")
print("-" * 60)
for i, N in enumerate(nFFTSize_list):
    print(f"{N:<8} {sqrt_3N_list[i]:<12.3f} {papr_linear_list[i]:<12.3f} {papr_linear_list[i]/sqrt_3N_list[i]:<12.3f}")

# Tracer le graphique
plt.figure(figsize=(10, 6))
plt.plot(sqrt_3N_list, papr_linear_list, 'o-', linewidth=2, markersize=8, label='PAPR simulé')

# Tracer la droite y = x pour comparaison
x_ref = np.array(sqrt_3N_list)
plt.plot(x_ref, x_ref, '--', linewidth=2, color='red', label=r'$y = \sqrt{3N}$ (référence)')

plt.xlabel(r'$\sqrt{3N}$', fontsize=14)
plt.ylabel('PAPR (linéaire)', fontsize=14)
plt.title(r'PAPR linéaire en fonction de $\sqrt{3N}$', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# Sauvegarder la figure
plt.savefig('rapport/chap1_question2.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure sauvegardée dans 'rapport/chap1_question2.png'")

# plt.show()  # Commenté pour éviter le blocage
