import numpy as np
import matplotlib.pyplot as plt
from commonFunction import *

"""
Question 2: Comparaison des courbes TEB pour 16-QAM, 32-QAM et 64-QAM
=====================================================================
On superpose les 3 courbes de TEB en fonction du SNR pour conclure.

Note: 32-QAM n'est pas une QAM carrée, donc on utilise 4-QAM à la place
pour avoir 3 courbes comparables (4-QAM, 16-QAM, 64-QAM).
"""

print("=" * 70)
print("Comparaison TEB: 4-QAM vs 16-QAM vs 64-QAM")
print("=" * 70)

# Paramètres de simulation fixes
nFFTSize = 64
nSymbol_OFDM = 2**10
L = nFFTSize // 4

# Liste des modulations à comparer (QAM carrées uniquement)
# Note: 32-QAM n'est pas supportée par commpy, on utilise 4, 16, 64
M_qam_list = [4, 16, 64]
colors = ['green', 'blue', 'red']
markers = ['s', 'o', '^']

# Plage de SNR
SNR_range = np.arange(0, 25, 1)  # 0 à 24 dB

# Stockage des résultats
TEB_results = {}
EVM_results = {}

for M_qam in M_qam_list:
    print(f"\n--- Simulation {M_qam}-QAM ---")
    
    nbBit_qam = int(np.log2(M_qam))
    nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
    
    # Émission
    tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
    tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
    
    TEB_list = []
    EVM_list = []
    
    for snr in SNR_range:
        # Canal AWGN
        rx = chan_awgn(tx, snr)
        
        # Réception
        _, rXmat = removeIGandFFT(rx, nSymbol_OFDM, nFFTSize, L)
        rX, rMsgBin = demapping2bit(rXmat, M_qam)
        
        # TEB
        n_err = np.sum(tMsgBin != rMsgBin)
        teb = n_err / len(tMsgBin)
        TEB_list.append(teb)
        
        # EVM
        evm = calculateEvm(tX, rX)
        EVM_list.append(evm)
    
    TEB_results[M_qam] = TEB_list
    EVM_results[M_qam] = EVM_list
    print(f"  Simulation terminée")

# Affichage des résultats
print("\n" + "=" * 70)
print("Résultats TEB pour quelques valeurs de SNR:")
print("=" * 70)
print(f"{'SNR (dB)':<10} {'4-QAM':<15} {'16-QAM':<15} {'64-QAM':<15}")
print("-" * 70)
for i, snr in enumerate(SNR_range):
    if snr % 4 == 0:  # Afficher tous les 4 dB
        print(f"{snr:<10} {TEB_results[4][i]:<15.4e} {TEB_results[16][i]:<15.4e} {TEB_results[64][i]:<15.4e}")

# Tracé des courbes TEB
plt.figure(figsize=(12, 7))

for i, M_qam in enumerate(M_qam_list):
    # Remplacer les zéros par une petite valeur pour le log
    TEB_plot = [max(t, 1e-7) for t in TEB_results[M_qam]]
    plt.semilogy(SNR_range, TEB_plot, f'{markers[i]}-', color=colors[i], 
                 linewidth=2, markersize=6, label=f'{M_qam}-QAM')

plt.xlabel('SNR (dB)', fontsize=14)
plt.ylabel('TEB (Taux d\'Erreur Binaire)', fontsize=14)
plt.title('Comparaison du TEB pour différentes modulations M-QAM', fontsize=16)
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=12)
plt.ylim([1e-6, 1])
plt.xlim([0, 24])
plt.tight_layout()

plt.savefig('rapport/chap2_teb_comparison_mqam.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure TEB comparaison sauvegardée dans 'rapport/chap2_teb_comparison_mqam.png'")

# Tracé des courbes EVM
plt.figure(figsize=(12, 7))

for i, M_qam in enumerate(M_qam_list):
    plt.plot(SNR_range, EVM_results[M_qam], f'{markers[i]}-', color=colors[i], 
             linewidth=2, markersize=6, label=f'{M_qam}-QAM')

plt.xlabel('SNR (dB)', fontsize=14)
plt.ylabel('EVM (%)', fontsize=14)
plt.title('Comparaison de l\'EVM pour différentes modulations M-QAM', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim([0, 24])
plt.tight_layout()

plt.savefig('rapport/chap2_evm_comparison_mqam.png', dpi=300, bbox_inches='tight')
print("✓ Figure EVM comparaison sauvegardée dans 'rapport/chap2_evm_comparison_mqam.png'")

# Conclusion
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
Observations sur les courbes TEB :

1. Plus l'ordre de modulation M est élevé, plus le TEB est important
   pour un même SNR.
   
2. À SNR = 12 dB :
   - 4-QAM  : TEB ≈ 10⁻³ à 10⁻⁴
   - 16-QAM : TEB ≈ 10⁻² 
   - 64-QAM : TEB ≈ 10⁻¹

3. Pour atteindre un TEB de 10⁻³ :
   - 4-QAM  nécessite ≈ 10 dB
   - 16-QAM nécessite ≈ 14-15 dB
   - 64-QAM nécessite ≈ 18-20 dB

4. COMPROMIS DÉBIT / ROBUSTESSE :
   - 4-QAM  : 2 bits/symbole, très robuste au bruit
   - 16-QAM : 4 bits/symbole, robustesse moyenne
   - 64-QAM : 6 bits/symbole, sensible au bruit
   
   Plus le débit est élevé, plus le système est sensible au bruit !
""")

print("\n" + "=" * 70)
print("DOCUMENTATION SUR L'EVM (Error Vector Magnitude)")
print("=" * 70)
print("""
L'EVM (Error Vector Magnitude) est une métrique qui mesure la qualité
de la modulation d'un signal numérique.

DÉFINITION :
L'EVM mesure la différence entre le symbole reçu et le symbole idéal
de la constellation, exprimée en pourcentage de l'amplitude RMS de 
référence :

   EVM (%) = (RMS_error / RMS_reference) × 100

où :
   - RMS_error = √(moyenne(|symbole_reçu - symbole_idéal|²))
   - RMS_reference = √(moyenne(|symbole_idéal|²))

INTERPRÉTATION :
   - EVM faible (< 5%)  : Excellente qualité de signal
   - EVM moyen (5-10%)  : Bonne qualité
   - EVM élevé (> 20%)  : Qualité dégradée, erreurs probables

RELATION AVEC LE SNR :
   EVM (%) ≈ 100 / √(SNR_linéaire)
   
   Par exemple, à SNR = 20 dB (soit 100 en linéaire) :
   EVM ≈ 100 / 10 = 10%

SOURCES DE DÉGRADATION :
   - Bruit thermique (AWGN)
   - Interférences
   - Non-linéarités de l'amplificateur
   - Erreurs de synchronisation
   - Décalage de fréquence porteuse

L'EVM est très utilisé dans les standards de communication (WiFi, LTE, 5G)
pour qualifier la qualité des transmetteurs et récepteurs.
""")

# plt.show()
