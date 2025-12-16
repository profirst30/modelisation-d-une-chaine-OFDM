import numpy as np
import matplotlib.pyplot as plt
from commonFunction import *

# Question 6: Passage en RF (Radio Fréquence)
# 1. Expression du signal RF en fonction de l'enveloppe complexe
# 2. Génération du signal RF avec fc = 0.1 (normalisé)
# 3. Comparaison du PAPR bande de base vs RF

print("=" * 70)
print("Question 6: Passage du signal en bande de base vers RF")
print("=" * 70)

# Paramètres de simulation
nFFTSize = 64
M_qam = 16
nbBit_qam = int(np.log2(M_qam))
nSymbol_OFDM = 2**12
nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
L = nFFTSize // 4
rolloff = 0.3
samples_per_symbol = 8

# Fréquence porteuse normalisée
fc = 0.1  # fc = 0.1 Hz avec fs = 1

print("\n--- Paramètres ---")
print(f"nFFTSize = {nFFTSize}")
print(f"M-QAM = {M_qam}")
print(f"Fréquence porteuse fc = {fc} (normalisée)")

## =================================================================
## 1. Expression théorique du signal RF
## =================================================================
print("\n" + "=" * 70)
print("1. EXPRESSION DU SIGNAL RF")
print("=" * 70)
print("""
Le signal en bande de base est l'enveloppe complexe :
   ṽ(t) = I(t) + j·Q(t)

où I(t) est la composante en phase et Q(t) la composante en quadrature.

Le signal RF réel transmis s'obtient par modulation sur la porteuse fc :
   
   s_RF(t) = Re{ ṽ(t) · e^(j·2π·fc·t) }
   
   s_RF(t) = I(t)·cos(2π·fc·t) - Q(t)·sin(2π·fc·t)

C'est la formule classique de la modulation IQ.
""")

## =================================================================
## 2. Génération du signal OFDM en bande de base
## =================================================================
print("Génération du signal OFDM en bande de base...")
tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
tSignal, rrcosFilter = rrcos(tx, rolloff, samples_per_symbol)

print(f"Taille de tSignal : {len(tSignal)}")

## =================================================================
## 3. Génération du signal RF
## =================================================================
print("\n" + "=" * 70)
print("2. GÉNÉRATION DU SIGNAL RF")
print("=" * 70)

# Vecteur temps (échantillons)
N = len(tSignal)
n = np.arange(N)
fs = 1  # Fréquence d'échantillonnage normalisée

# Composantes I et Q du signal en bande de base
I_t = np.real(tSignal)  # Composante en phase
Q_t = np.imag(tSignal)  # Composante en quadrature

# Signal RF : s_RF(t) = I(t)·cos(2π·fc·t) - Q(t)·sin(2π·fc·t)
# En discret : s_RF[n] = I[n]·cos(2π·fc·n/fs) - Q[n]·sin(2π·fc·n/fs)
cos_carrier = np.cos(2 * np.pi * fc * n / fs)
sin_carrier = np.sin(2 * np.pi * fc * n / fs)

tSignal_RF = I_t * cos_carrier - Q_t * sin_carrier

print(f"Signal RF généré avec fc = {fc}")
print(f"Nombre de périodes de la porteuse : {fc * N / fs:.0f}")

## =================================================================
## 4. Visualisation et comparaison
## =================================================================
print("\n" + "=" * 70)
print("3. VISUALISATION")
print("=" * 70)

# Zoom sur une portion du signal pour voir la porteuse
zoom_start = 10000
zoom_length = 500  # Nombre d'échantillons à afficher
zoom_end = zoom_start + zoom_length

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Signal en bande de base (enveloppe = |tSignal|)
ax1 = axes[0]
ax1.plot(n[zoom_start:zoom_end], np.abs(tSignal[zoom_start:zoom_end]), 'b-', linewidth=1.5, label='|tSignal| (enveloppe)')
ax1.set_xlabel('Échantillons', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_title('Signal en bande de base : enveloppe |tSignal|', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Signal RF
ax2 = axes[1]
ax2.plot(n[zoom_start:zoom_end], tSignal_RF[zoom_start:zoom_end], 'r-', linewidth=0.8, label='Signal RF')
ax2.plot(n[zoom_start:zoom_end], np.abs(tSignal[zoom_start:zoom_end]), 'b--', linewidth=1.5, alpha=0.7, label='Enveloppe |tSignal|')
ax2.plot(n[zoom_start:zoom_end], -np.abs(tSignal[zoom_start:zoom_end]), 'b--', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Échantillons', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_title(f'Signal RF modulé sur porteuse fc = {fc}', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('rapport/chap1_question6_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Figure de comparaison sauvegardée dans 'rapport/chap1_question6_comparison.png'")

## =================================================================
## 5. Calcul et comparaison des PAPR
## =================================================================
print("\n" + "=" * 70)
print("4. COMPARAISON DES PAPR")
print("=" * 70)

# PAPR du signal en bande de base
power_baseband = np.abs(tSignal)**2
papr_baseband_linear = np.max(power_baseband) / np.mean(power_baseband)
papr_baseband_dB = 10 * np.log10(papr_baseband_linear)

# PAPR du signal RF (signal réel)
power_RF = tSignal_RF**2  # Signal réel, pas besoin de abs()
papr_RF_linear = np.max(power_RF) / np.mean(power_RF)
papr_RF_dB = 10 * np.log10(papr_RF_linear)

# Différence
delta_papr_dB = papr_RF_dB - papr_baseband_dB

print(f"\nSignal en bande de base (enveloppe complexe) :")
print(f"  PAPR linéaire = {papr_baseband_linear:.3f}")
print(f"  PAPR (dB)     = {papr_baseband_dB:.3f} dB")

print(f"\nSignal RF (réel) :")
print(f"  PAPR linéaire = {papr_RF_linear:.3f}")
print(f"  PAPR (dB)     = {papr_RF_dB:.3f} dB")

print(f"\nDifférence :")
print(f"  ΔPAPR = {delta_papr_dB:.3f} dB")

## =================================================================
## 6. Justification
## =================================================================
print("\n" + "=" * 70)
print("5. JUSTIFICATION")
print("=" * 70)
print(f"""
Pourquoi le PAPR RF est environ 3 dB plus grand ?

1) Signal en bande de base (complexe) :
   - Puissance instantanée : P(t) = |ṽ(t)|² = I(t)² + Q(t)²
   - Puissance moyenne : E[|ṽ(t)|²]

2) Signal RF (réel) :
   - s_RF(t) = I(t)·cos(ωt) - Q(t)·sin(ωt)
   - Puissance instantanée : s_RF(t)²
   
   En moyennant sur une période de la porteuse :
   E[s_RF²] = (1/2)·E[I² + Q²] = (1/2)·E[|ṽ|²]
   
   La puissance moyenne du signal RF est donc 2× plus petite (−3 dB)
   que celle du signal en bande de base.

3) Pic de puissance :
   - Le pic du signal RF peut atteindre max|ṽ(t)| (quand la porteuse 
     est en phase avec l'enveloppe)
   - Donc max(s_RF²) ≈ max(|ṽ|²)

4) Conclusion :
   PAPR_RF = max(s_RF²) / mean(s_RF²)
           ≈ max(|ṽ|²) / (mean(|ṽ|²)/2)
           = 2 × PAPR_baseband
   
   En dB : PAPR_RF(dB) ≈ PAPR_baseband(dB) + 3 dB

Résultat obtenu : ΔPAPR = {delta_papr_dB:.2f} dB ≈ 3 dB ✓
""")

# plt.show()
