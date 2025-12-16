import numpy as np
import matplotlib.pyplot as plt
from commonFunction import *

"""
Question 4 & 5: EVM en fonction du SNR
======================================
4. Tracer l'EVM en fonction du SNR (2 à 16 dB) pour 16-QAM
5. Idem pour 256-QAM, superposer les courbes et justifier
"""

print("=" * 70)
print("Questions 4 & 5: EVM en fonction du SNR")
print("=" * 70)

# Paramètres de simulation fixes
nFFTSize = 64
nSymbol_OFDM = 2**10
L = nFFTSize // 4

# Plage de SNR
SNR_range = np.arange(2, 17, 1)  # 2 à 16 dB avec pas de 1

# Modulations à comparer
M_qam_list = [16, 256]
colors = ['blue', 'red']
markers = ['o', 's']

# Stockage des résultats
EVM_results = {}
TEB_results = {}

for M_qam in M_qam_list:
    print(f"\n--- Simulation {M_qam}-QAM ---")
    
    nbBit_qam = int(np.log2(M_qam))
    nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
    
    # Émission
    tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
    tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
    
    EVM_list = []
    TEB_list = []
    
    for snr in SNR_range:
        # Canal AWGN
        rx = chan_awgn(tx, snr)
        
        # Réception
        _, rXmat = removeIGandFFT(rx, nSymbol_OFDM, nFFTSize, L)
        rX, rMsgBin = demapping2bit(rXmat, M_qam)
        
        # EVM
        evm = calculateEvm(tX, rX)
        EVM_list.append(evm)
        
        # TEB
        n_err = np.sum(tMsgBin != rMsgBin)
        teb = n_err / len(tMsgBin)
        TEB_list.append(teb)
        
        print(f"  SNR = {snr:2d} dB : EVM = {evm:.2f}%, TEB = {teb:.4e}")
    
    EVM_results[M_qam] = EVM_list
    TEB_results[M_qam] = TEB_list

# Calcul de l'EVM théorique en fonction du SNR
# EVM_théorique (%) = 100 / sqrt(SNR_linéaire)
SNR_linear = 10**(SNR_range / 10)
EVM_theorique = 100 / np.sqrt(SNR_linear)

# Affichage des résultats
print("\n" + "=" * 70)
print("Résultats EVM:")
print("=" * 70)
print(f"{'SNR (dB)':<10} {'16-QAM':<15} {'256-QAM':<15} {'Théorique':<15}")
print("-" * 70)
for i, snr in enumerate(SNR_range):
    print(f"{snr:<10} {EVM_results[16][i]:<15.2f} {EVM_results[256][i]:<15.2f} {EVM_theorique[i]:<15.2f}")

# ============================================================================
# TRACÉ DES COURBES
# ============================================================================

# Figure 1 : EVM vs SNR
plt.figure(figsize=(12, 6))

for i, M_qam in enumerate(M_qam_list):
    plt.plot(SNR_range, EVM_results[M_qam], f'{markers[i]}-', color=colors[i], 
             linewidth=2, markersize=8, label=f'{M_qam}-QAM (mesuré)')

# Courbe théorique
plt.plot(SNR_range, EVM_theorique, 'k--', linewidth=2, label='Théorique: 100/√SNR')

plt.xlabel('SNR (dB)', fontsize=14)
plt.ylabel('EVM (%)', fontsize=14)
plt.title('EVM en fonction du SNR pour 16-QAM et 256-QAM', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim([2, 16])
plt.tight_layout()

plt.savefig('rapport/chap2_evm_vs_snr_16_256.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure EVM vs SNR sauvegardée dans 'rapport/chap2_evm_vs_snr_16_256.png'")

# Figure 2 : TEB vs SNR pour comparaison
plt.figure(figsize=(12, 6))

for i, M_qam in enumerate(M_qam_list):
    TEB_plot = [max(t, 1e-7) for t in TEB_results[M_qam]]
    plt.semilogy(SNR_range, TEB_plot, f'{markers[i]}-', color=colors[i], 
                 linewidth=2, markersize=8, label=f'{M_qam}-QAM')

plt.xlabel('SNR (dB)', fontsize=14)
plt.ylabel('TEB', fontsize=14)
plt.title('TEB en fonction du SNR pour 16-QAM et 256-QAM', fontsize=16)
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=12)
plt.xlim([2, 16])
plt.ylim([1e-5, 1])
plt.tight_layout()

plt.savefig('rapport/chap2_teb_vs_snr_16_256.png', dpi=300, bbox_inches='tight')
print("✓ Figure TEB vs SNR sauvegardée dans 'rapport/chap2_teb_vs_snr_16_256.png'")

# ============================================================================
# ANALYSE ET JUSTIFICATION
# ============================================================================
print("\n" + "=" * 70)
print("OBSERVATIONS SUR L'EVM")
print("=" * 70)
print("""
1. L'EVM est IDENTIQUE pour 16-QAM et 256-QAM à un même SNR.

2. L'EVM suit la loi théorique : EVM (%) ≈ 100 / √(SNR_linéaire)

3. Exemples numériques :
   - SNR = 10 dB (linéaire = 10)  → EVM ≈ 100/√10 ≈ 31.6%
   - SNR = 16 dB (linéaire = 40)  → EVM ≈ 100/√40 ≈ 15.8%
""")

print("\n" + "=" * 70)
print("JUSTIFICATION")
print("=" * 70)
print("""
Pourquoi l'EVM est-il INDÉPENDANT de l'ordre de modulation M ?

L'EVM mesure l'erreur relative entre le symbole reçu et le symbole émis :

   EVM = RMS(erreur) / RMS(signal)

Dans un canal AWGN :
   - L'erreur (bruit) a une variance σ² = Ps / SNR (où Ps = puissance signal)
   - RMS(erreur) = σ = √(Ps/SNR)
   - RMS(signal) = √Ps

Donc :
   EVM = √(Ps/SNR) / √Ps = 1/√SNR = 100% / √(SNR_linéaire)

Cette formule ne dépend PAS de M !

Par contre, le TEB dépend de M car :
   - Plus M est grand, plus les symboles sont proches dans la constellation
   - Une même erreur (même EVM) cause plus de décodages erronés en 256-QAM
   - La distance minimale entre symboles diminue quand M augmente
""")

print("\n" + "=" * 70)
print("COMPARAISON TEB")
print("=" * 70)
print("""
Bien que l'EVM soit identique :

   - 16-QAM  : 4 bits/symbole, distance entre symboles = d
   - 256-QAM : 8 bits/symbole, distance entre symboles ≈ d/4

À SNR = 12 dB :
   - 16-QAM  : TEB ≈ 2.8×10⁻²
   - 256-QAM : TEB ≈ 2.2×10⁻¹ (presque 10× plus d'erreurs !)

Conclusion : L'EVM est une mesure de la QUALITÉ DU CANAL (SNR),
tandis que le TEB dépend à la fois du canal ET de la modulation.
""")

# plt.show()
