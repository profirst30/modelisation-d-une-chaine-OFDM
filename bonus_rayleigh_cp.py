import numpy as np
import matplotlib.pyplot as plt
from commonFunction import *

"""
PARTIE BONUS : Canal sélectif de type Rayleigh
===============================================
1. Comparer EVM et TEB avec canal AWGN
2. Faire varier la longueur du préfixe cyclique (IG) : 0, 2, 4, 16
"""

print("=" * 70)
print("PARTIE BONUS : Canal Rayleigh")
print("=" * 70)

# ============================================================================
# FONCTIONS CANAL RAYLEIGH AVEC ÉGALISATION
# ============================================================================

def generate_rayleigh_channel(L_channel, nFFTSize):
    """
    Génère un canal de Rayleigh et sa réponse fréquentielle.
    
    Args:
        L_channel: Nombre de trajets du canal
        nFFTSize: Taille FFT
    
    Returns:
        h: Réponse impulsionnelle du canal
        H: Réponse fréquentielle du canal (taille nFFTSize)
    """
    # Génération de la réponse impulsionnelle du canal Rayleigh
    h_real = np.random.randn(L_channel)
    h_imag = np.random.randn(L_channel)
    h = (h_real + 1j * h_imag) / np.sqrt(2)
    
    # Normalisation de l'énergie du canal
    h = h / np.sqrt(np.sum(np.abs(h)**2))
    
    # Réponse fréquentielle du canal (FFT zéro-paddée)
    H = np.fft.fft(h, nFFTSize)
    
    return h, H


def apply_rayleigh_channel(tx, h, snr_db):
    """
    Applique le canal Rayleigh au signal.
    
    Args:
        tx: Signal émis
        h: Réponse impulsionnelle du canal
        snr_db: SNR en dB
    
    Returns:
        rx: Signal reçu bruité
    """
    # Convolution du signal avec le canal
    signal_conv = np.convolve(tx, h, mode='full')
    # Tronquer pour garder la même taille
    signal_conv = signal_conv[:len(tx)]
    
    # Ajout du bruit AWGN
    rx = chan_awgn(signal_conv, snr_db)
    
    return rx


def equalize_zf(rXmat, H):
    """
    Égalisation Zero-Forcing dans le domaine fréquentiel.
    
    Args:
        rXmat: Symboles reçus en fréquence (nFFTSize x nSymbol)
        H: Réponse fréquentielle du canal
    
    Returns:
        rXmat_eq: Symboles égalisés
    """
    # Égalisation ZF : diviser par la réponse du canal
    H_matrix = H[:, np.newaxis]  # Broadcast sur les symboles
    rXmat_eq = rXmat / H_matrix
    
    return rXmat_eq


# ============================================================================
# PARAMÈTRES DE SIMULATION
# ============================================================================

nFFTSize = 64
M_qam = 16
nbBit_qam = int(np.log2(M_qam))
nSymbol_OFDM = 2**10
L = nFFTSize // 4  # Préfixe cyclique = 16
L_channel = 8      # Longueur du canal Rayleigh (nombre de trajets)

nBit = nFFTSize * nbBit_qam * nSymbol_OFDM

print(f"\nParamètres:")
print(f"  nFFTSize = {nFFTSize}")
print(f"  M-QAM = {M_qam}")
print(f"  Préfixe cyclique L = {L}")
print(f"  Longueur canal Rayleigh = {L_channel}")

# ============================================================================
# QUESTION 1: COMPARAISON AWGN vs RAYLEIGH (avec égalisation)
# ============================================================================

print("\n" + "=" * 70)
print("QUESTION 1: Comparaison canal AWGN vs Rayleigh")
print("=" * 70)

SNR_range = np.arange(2, 31, 2)  # 2 à 30 dB

# Résultats pour les deux canaux
results = {
    'AWGN': {'EVM': [], 'TEB': []},
    'Rayleigh': {'EVM': [], 'TEB': []},
    'Rayleigh_eq': {'EVM': [], 'TEB': []}  # Avec égalisation
}

print("\nSimulation en cours...")
print("(Rayleigh_EQ = avec égalisation Zero-Forcing)")

for snr in SNR_range:
    # Nouvelle réalisation du canal et des données pour chaque SNR
    tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
    tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
    
    # Générer un nouveau canal Rayleigh
    h, H = generate_rayleigh_channel(L_channel, nFFTSize)
    
    # --- Canal AWGN ---
    rx_awgn = chan_awgn(tx, snr)
    _, rXmat_awgn = removeIGandFFT(rx_awgn, nSymbol_OFDM, nFFTSize, L)
    rX_awgn, rMsgBin_awgn = demapping2bit(rXmat_awgn, M_qam)
    
    evm_awgn = calculateEvm(tX, rX_awgn)
    teb_awgn = np.sum(tMsgBin != rMsgBin_awgn) / len(tMsgBin)
    
    results['AWGN']['EVM'].append(evm_awgn)
    results['AWGN']['TEB'].append(teb_awgn)
    
    # --- Canal Rayleigh SANS égalisation ---
    rx_rayleigh = apply_rayleigh_channel(tx, h, snr)
    _, rXmat_rayleigh = removeIGandFFT(rx_rayleigh, nSymbol_OFDM, nFFTSize, L)
    rX_rayleigh, rMsgBin_rayleigh = demapping2bit(rXmat_rayleigh, M_qam)
    
    evm_rayleigh = calculateEvm(tX, rX_rayleigh)
    teb_rayleigh = np.sum(tMsgBin != rMsgBin_rayleigh) / len(tMsgBin)
    
    results['Rayleigh']['EVM'].append(evm_rayleigh)
    results['Rayleigh']['TEB'].append(teb_rayleigh)
    
    # --- Canal Rayleigh AVEC égalisation ZF ---
    rXmat_eq = equalize_zf(rXmat_rayleigh, H)
    rX_rayleigh_eq, rMsgBin_rayleigh_eq = demapping2bit(rXmat_eq, M_qam)
    
    evm_rayleigh_eq = calculateEvm(tX, rX_rayleigh_eq)
    teb_rayleigh_eq = np.sum(tMsgBin != rMsgBin_rayleigh_eq) / len(tMsgBin)
    
    results['Rayleigh_eq']['EVM'].append(evm_rayleigh_eq)
    results['Rayleigh_eq']['TEB'].append(teb_rayleigh_eq)
    
    print(f"  SNR={snr:2d}dB: AWGN(TEB={teb_awgn:.2e}) | Rayleigh(TEB={teb_rayleigh:.2e}) | Rayleigh_EQ(TEB={teb_rayleigh_eq:.2e})")

# Tracé des courbes de comparaison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# EVM
ax1 = axes[0]
ax1.plot(SNR_range, results['AWGN']['EVM'], 'bo-', linewidth=2, markersize=6, label='AWGN')
ax1.plot(SNR_range, results['Rayleigh']['EVM'], 'rs-', linewidth=2, markersize=6, label='Rayleigh (sans égal.)')
ax1.plot(SNR_range, results['Rayleigh_eq']['EVM'], 'g^-', linewidth=2, markersize=6, label='Rayleigh (avec égal. ZF)')
ax1.set_xlabel('SNR (dB)', fontsize=14)
ax1.set_ylabel('EVM (%)', fontsize=14)
ax1.set_title('EVM: Canal AWGN vs Rayleigh', fontsize=16)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12)

# TEB
ax2 = axes[1]
TEB_awgn_plot = [max(t, 1e-7) for t in results['AWGN']['TEB']]
TEB_rayleigh_plot = [max(t, 1e-7) for t in results['Rayleigh']['TEB']]
TEB_rayleigh_eq_plot = [max(t, 1e-7) for t in results['Rayleigh_eq']['TEB']]
ax2.semilogy(SNR_range, TEB_awgn_plot, 'bo-', linewidth=2, markersize=6, label='AWGN')
ax2.semilogy(SNR_range, TEB_rayleigh_plot, 'rs-', linewidth=2, markersize=6, label='Rayleigh (sans égal.)')
ax2.semilogy(SNR_range, TEB_rayleigh_eq_plot, 'g^-', linewidth=2, markersize=6, label='Rayleigh (avec égal. ZF)')
ax2.set_xlabel('SNR (dB)', fontsize=14)
ax2.set_ylabel('TEB', fontsize=14)
ax2.set_title('TEB: Canal AWGN vs Rayleigh', fontsize=16)
ax2.grid(True, which='both', alpha=0.3)
ax2.legend(fontsize=12)
ax2.set_ylim([1e-6, 1])

plt.tight_layout()
plt.savefig('rapport/bonus_awgn_vs_rayleigh.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure AWGN vs Rayleigh sauvegardée")

# ============================================================================
# QUESTION 2: VARIATION DE LA LONGUEUR DU PRÉFIXE CYCLIQUE
# ============================================================================

print("\n" + "=" * 70)
print("QUESTION 2: Variation de la longueur du préfixe cyclique (IG)")
print("=" * 70)

SNR_test = 30  # SNR élevé (faible bruit)
L_values = [0, 2, 4, 8, 16]  # Longueurs de préfixe cyclique à tester

print(f"\nSNR fixé à {SNR_test} dB (faible bruit)")
print(f"Longueurs de CP testées: {L_values}")
print(f"Longueur du canal Rayleigh: {L_channel} échantillons")

results_L = {'EVM': [], 'TEB': [], 'EVM_eq': [], 'TEB_eq': []}

# Générer un canal fixe pour cette étude
h_fixed, H_fixed = generate_rayleigh_channel(L_channel, nFFTSize)

for L_test in L_values:
    print(f"\n--- L = {L_test} (préfixe cyclique) ---")
    
    # Recalculer le nombre de bits pour cette valeur de L
    nBit_test = nFFTSize * nbBit_qam * nSymbol_OFDM
    
    # Émission avec ce préfixe cyclique
    tMsgBin_test, tX_test, tXmat_test = bitMapping(nBit_test, M_qam, nFFTSize)
    tx_test, txMat_test = ifftAddIg(tXmat_test, nSymbol_OFDM, nFFTSize, L_test)
    
    # Canal Rayleigh (utiliser le canal fixe)
    rx_test = apply_rayleigh_channel(tx_test, h_fixed, SNR_test)
    
    # Réception sans égalisation
    _, rXmat_test = removeIGandFFT(rx_test, nSymbol_OFDM, nFFTSize, L_test)
    rX_test, rMsgBin_test = demapping2bit(rXmat_test, M_qam)
    
    # Réception avec égalisation ZF
    rXmat_eq = equalize_zf(rXmat_test, H_fixed)
    rX_eq, rMsgBin_eq = demapping2bit(rXmat_eq, M_qam)
    
    # Métriques sans égalisation
    evm_test = calculateEvm(tX_test, rX_test)
    teb_test = np.sum(tMsgBin_test != rMsgBin_test) / len(tMsgBin_test)
    
    # Métriques avec égalisation
    evm_eq = calculateEvm(tX_test, rX_eq)
    teb_eq = np.sum(tMsgBin_test != rMsgBin_eq) / len(tMsgBin_test)
    
    results_L['EVM'].append(evm_test)
    results_L['TEB'].append(teb_test)
    results_L['EVM_eq'].append(evm_eq)
    results_L['TEB_eq'].append(teb_eq)
    
    print(f"  Sans égalisation: EVM = {evm_test:.2f}%, TEB = {teb_test:.4e}")
    print(f"  Avec égalisation:  EVM = {evm_eq:.2f}%, TEB = {teb_eq:.4e}")
    
    if L_test < L_channel:
        print(f"  ⚠ ATTENTION: L={L_test} < L_canal={L_channel} → Interférence inter-symboles (ISI) !")

# Tableau récapitulatif
print("\n" + "=" * 70)
print("RÉSUMÉ: Effet de la longueur du préfixe cyclique (avec égalisation ZF)")
print("=" * 70)
print(f"{'L (CP)':<10} {'EVM (%)':<12} {'TEB':<15} {'EVM_eq (%)':<12} {'TEB_eq':<15} {'Commentaire'}")
print("-" * 80)
for i, L_test in enumerate(L_values):
    comment = "ISI !" if L_test < L_channel else "OK"
    print(f"{L_test:<10} {results_L['EVM'][i]:<12.2f} {results_L['TEB'][i]:<15.4e} {results_L['EVM_eq'][i]:<12.2f} {results_L['TEB_eq'][i]:<15.4e} {comment}")

# Tracé
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# EVM vs L
ax1 = axes[0]
x = np.arange(len(L_values))
width = 0.35
bars1 = ax1.bar(x - width/2, results_L['EVM_eq'], width, label='Avec égalisation', 
                color=['red' if L < L_channel else 'green' for L in L_values])
ax1.set_xticks(x)
ax1.set_xticklabels([f'L={L}' for L in L_values])
ax1.set_xlabel('Longueur du préfixe cyclique', fontsize=14)
ax1.set_ylabel('EVM (%)', fontsize=14)
ax1.set_title(f'EVM en fonction de L (SNR={SNR_test}dB, L_canal={L_channel})', fontsize=14)
ax1.axvline(x=L_values.index(L_channel)-0.5, color='orange', linestyle='--', linewidth=2, label=f'L_canal={L_channel}')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# TEB vs L
ax2 = axes[1]
TEB_plot = [max(t, 1e-7) for t in results_L['TEB_eq']]
bars = ax2.bar(x, TEB_plot, color=['red' if L < L_channel else 'green' for L in L_values])
ax2.set_xticks(x)
ax2.set_xticklabels([f'L={L}' for L in L_values])
ax2.set_xlabel('Longueur du préfixe cyclique', fontsize=14)
ax2.set_ylabel('TEB', fontsize=14)
ax2.set_title(f'TEB en fonction de L (SNR={SNR_test}dB, L_canal={L_channel})', fontsize=14)
ax2.axvline(x=L_values.index(L_channel)-0.5, color='orange', linestyle='--', linewidth=2, label=f'L_canal={L_channel}')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rapport/bonus_variation_cp.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure variation CP sauvegardée")

# ============================================================================
# CONCLUSIONS
# ============================================================================

print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║ 1. CANAL RAYLEIGH vs AWGN                                             ║
╠═══════════════════════════════════════════════════════════════════════╣
║ • Le canal Rayleigh (multi-trajets) dégrade fortement les performances║
║ • À même SNR, l'EVM et le TEB sont BEAUCOUP plus élevés               ║
║ • Le canal Rayleigh est sélectif en fréquence : certaines             ║
║   sous-porteuses subissent des évanouissements profonds (fading)      ║
║ • Solution : utiliser un égaliseur pour compenser le canal            ║
╚═══════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════╗
║ 2. RÔLE DU PRÉFIXE CYCLIQUE (CP)                                      ║
╠═══════════════════════════════════════════════════════════════════════╣
║ • Le CP doit être ≥ à la longueur du canal (réponse impulsionnelle)   ║
║ • Si L_CP < L_canal : Interférence Inter-Symboles (ISI) !             ║
║   → Dégradation importante de l'EVM et du TEB                         ║
║ • Si L_CP ≥ L_canal : Le CP absorbe l'ISI, performances optimales     ║
║                                                                        ║
║ Compromis :                                                            ║
║   • CP trop court → ISI                                                ║
║   • CP trop long  → Perte d'efficacité spectrale (overhead)           ║
║                                                                        ║
║ En pratique : L_CP ≈ 1/4 de la durée symbole OFDM (standard WiFi/LTE) ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

# plt.show()
