"""
Created on Thu Oct  2 22:24:37 2025
@author: kdjessou

    File    : main_tp.py
    Date    : 08/11/2025
    Status  : OK 
    
    TP OFDM - Script Principal avec Menu Interactif
    ================================================
    Ce script regroupe toutes les parties du TP sur la modélisation
    d'une chaîne OFDM (émission, réception, canal).
"""

##############################################################################
###             Zone 1 : Import Libraries
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from commonFunction import *

##############################################################################
###             Zone 2 : Constantes par défaut
##############################################################################

# Paramètres OFDM par défaut
DEFAULT_PARAMS = {
    'nFFTSize': 64,
    'M_qam': 16,
    'nSymbol_OFDM': 2**12,
    'rolloff': 0.3,
    'samples_per_symbol': 8,
    'fc': 2.4e9  # Fréquence porteuse pour RF
}

##############################################################################
###             Zone 3 : Functions
##############################################################################

def print_header():
    """Affiche l'en-tête du programme."""
    print("\n" + "=" * 70)
    print("         TP OFDM - Modélisation d'une chaîne de transmission")
    print("=" * 70)
    print("  Auteur : K. Djessou")
    print("  M2 Objets Connectés - Systèmes Radio et Intégration")
    print("=" * 70)


def print_menu():
    """Affiche le menu principal."""
    print("\n" + "-" * 50)
    print("                    MENU PRINCIPAL")
    print("-" * 50)
    print("  [1] Question 1 : Calcul du PAPR")
    print("  [2] Question 2 : PAPR vs Taille FFT")
    print("  [3] Question 3 : PAPR vs Ordre M-QAM (N=64)")
    print("  [4] Question 4 : Vérification PAPR (N=128)")
    print("  [5] Question 5 : Spectre et bande passante")
    print("  [6] Question 6 : Signal RF et PAPR")
    print("-" * 50)
    print("  [7] Réception : Chaîne complète avec TEB/EVM")
    print("  [8] Comparaison TEB : 4-QAM, 16-QAM, 64-QAM")
    print("  [9] EVM vs SNR (indépendance de M)")
    print("-" * 50)
    print("  [10] BONUS : Canal Rayleigh vs AWGN")
    print("  [11] BONUS : Variation du préfixe cyclique")
    print("-" * 50)
    print("  [0] Quitter")
    print("-" * 50)


# ============================================================================
# QUESTION 1 : Calcul du PAPR
# ============================================================================

def question1_papr():
    """Calcule et affiche le PAPR du signal OFDM."""
    print("\n" + "=" * 60)
    print("QUESTION 1 : Calcul du PAPR du signal OFDM")
    print("=" * 60)
    
    # Paramètres
    nFFTSize = DEFAULT_PARAMS['nFFTSize']
    M_qam = DEFAULT_PARAMS['M_qam']
    nSymbol_OFDM = DEFAULT_PARAMS['nSymbol_OFDM']
    L = nFFTSize // 4
    nbBit_qam = int(np.log2(M_qam))
    nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
    
    print(f"\nParamètres:")
    print(f"  N_FFT = {nFFTSize}")
    print(f"  M-QAM = {M_qam}")
    print(f"  Symboles OFDM = {nSymbol_OFDM}")
    print(f"  Préfixe cyclique L = {L}")
    
    # Émission
    tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
    tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
    tSignal, rrcos_filter = rrcos(tx, DEFAULT_PARAMS['rolloff'], 
                                   DEFAULT_PARAMS['samples_per_symbol'])
    
    # Calcul du PAPR
    power_signal = np.abs(tSignal)**2
    papr_linear = np.max(power_signal) / np.mean(power_signal)
    papr_db = 10 * np.log10(papr_linear)
    
    print(f"\n--- Résultats ---")
    print(f"  PAPR linéaire = {papr_linear:.2f}")
    print(f"  PAPR (dB)     = {papr_db:.2f} dB")
    print(f"  √(3N)         = {np.sqrt(3 * nFFTSize):.2f}")
    
    # Tracé du spectre
    plt.figure(figsize=(10, 5))
    freq = np.fft.fftfreq(len(tSignal), d=1)
    spectrum = np.abs(np.fft.fft(tSignal))**2
    spectrum_db = 10 * np.log10(spectrum / np.max(spectrum) + 1e-12)
    plt.plot(np.fft.fftshift(freq), np.fft.fftshift(spectrum_db))
    plt.xlabel('Fréquence normalisée')
    plt.ylabel('DSP (dB)')
    plt.title(f'Spectre du signal OFDM (PAPR = {papr_db:.2f} dB)')
    plt.grid(True, alpha=0.3)
    plt.ylim([-80, 5])
    plt.tight_layout()
    plt.savefig('rapport/q1_papr_spectrum.png', dpi=150)
    plt.show()
    
    return papr_db


# ============================================================================
# QUESTION 2 : PAPR vs Taille FFT
# ============================================================================

def question2_papr_vs_fft():
    """Étudie l'évolution du PAPR en fonction de la taille FFT."""
    print("\n" + "=" * 60)
    print("QUESTION 2 : PAPR vs Taille de FFT")
    print("=" * 60)
    
    nFFT_values = [16, 32, 64, 128, 256, 512]
    M_qam = DEFAULT_PARAMS['M_qam']
    nSymbol_OFDM = 2**10
    
    results = {'N': [], 'sqrt3N': [], 'PAPR_lin': [], 'PAPR_dB': []}
    
    print(f"\n{'N_FFT':<10} {'√(3N)':<10} {'PAPR_lin':<12} {'PAPR_dB':<10}")
    print("-" * 45)
    
    for nFFTSize in nFFT_values:
        L = nFFTSize // 4
        nbBit_qam = int(np.log2(M_qam))
        nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
        
        tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
        tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
        tSignal, _ = rrcos(tx, DEFAULT_PARAMS['rolloff'], 
                           DEFAULT_PARAMS['samples_per_symbol'])
        
        power = np.abs(tSignal)**2
        papr_lin = np.max(power) / np.mean(power)
        papr_db = 10 * np.log10(papr_lin)
        sqrt3N = np.sqrt(3 * nFFTSize)
        
        results['N'].append(nFFTSize)
        results['sqrt3N'].append(sqrt3N)
        results['PAPR_lin'].append(papr_lin)
        results['PAPR_dB'].append(papr_db)
        
        print(f"{nFFTSize:<10} {sqrt3N:<10.2f} {papr_lin:<12.2f} {papr_db:<10.2f}")
    
    # Tracé
    plt.figure(figsize=(10, 5))
    plt.plot(results['sqrt3N'], results['PAPR_lin'], 'bo-', 
             markersize=10, linewidth=2, label='PAPR mesuré')
    plt.plot(results['sqrt3N'], results['sqrt3N'], 'r--', 
             linewidth=2, label='y = √(3N)')
    plt.xlabel('√(3N)', fontsize=14)
    plt.ylabel('PAPR linéaire', fontsize=14)
    plt.title('PAPR en fonction de √(3N)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rapport/q2_papr_vs_fft.png', dpi=150)
    plt.show()
    
    print("\n✓ Conclusion: Le PAPR sature pour les grandes valeurs de N")


# ============================================================================
# QUESTION 3 : PAPR vs Ordre M-QAM
# ============================================================================

def question3_papr_vs_mqam():
    """Étudie l'influence de l'ordre de modulation sur le PAPR."""
    print("\n" + "=" * 60)
    print("QUESTION 3 : PAPR vs Ordre de modulation M-QAM (N=64)")
    print("=" * 60)
    
    nFFTSize = 64
    M_values = [4, 16, 64, 256, 1024]
    nSymbol_OFDM = 2**10
    L = nFFTSize // 4
    
    results = {'M': [], 'PAPR_dB': []}
    
    print(f"\n{'M-QAM':<10} {'PAPR (dB)':<12}")
    print("-" * 25)
    
    for M_qam in M_values:
        nbBit_qam = int(np.log2(M_qam))
        nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
        
        tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
        tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
        tSignal, _ = rrcos(tx, DEFAULT_PARAMS['rolloff'], 
                           DEFAULT_PARAMS['samples_per_symbol'])
        
        power = np.abs(tSignal)**2
        papr_lin = np.max(power) / np.mean(power)
        papr_db = 10 * np.log10(papr_lin)
        
        results['M'].append(M_qam)
        results['PAPR_dB'].append(papr_db)
        
        print(f"{M_qam:<10} {papr_db:<12.2f}")
    
    # Tracé
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(M_values)), results['PAPR_dB'], color='steelblue')
    plt.xticks(range(len(M_values)), [f'{M}-QAM' for M in M_values])
    plt.xlabel('Ordre de modulation', fontsize=14)
    plt.ylabel('PAPR (dB)', fontsize=14)
    plt.title('PAPR en fonction de M-QAM (N=64)', fontsize=16)
    plt.ylim([0, 15])
    plt.axhline(y=np.mean(results['PAPR_dB']), color='r', 
                linestyle='--', label=f'Moyenne = {np.mean(results["PAPR_dB"]):.1f} dB')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('rapport/q3_papr_vs_mqam.png', dpi=150)
    plt.show()
    
    print(f"\n✓ Conclusion: Le PAPR est quasi-constant (~{np.mean(results['PAPR_dB']):.1f} dB)")


# ============================================================================
# QUESTION 4 : Vérification avec N=128
# ============================================================================

def question4_verification():
    """Vérifie l'indépendance du PAPR par rapport à M avec N=128."""
    print("\n" + "=" * 60)
    print("QUESTION 4 : Vérification PAPR vs M-QAM (N=128)")
    print("=" * 60)
    
    nFFTSize = 128
    M_values = [4, 16, 64, 256]
    nSymbol_OFDM = 2**10
    L = nFFTSize // 4
    
    results = []
    
    print(f"\n{'M-QAM':<10} {'PAPR (dB)':<12}")
    print("-" * 25)
    
    for M_qam in M_values:
        nbBit_qam = int(np.log2(M_qam))
        nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
        
        tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
        tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
        tSignal, _ = rrcos(tx, DEFAULT_PARAMS['rolloff'], 
                           DEFAULT_PARAMS['samples_per_symbol'])
        
        power = np.abs(tSignal)**2
        papr_db = 10 * np.log10(np.max(power) / np.mean(power))
        results.append(papr_db)
        
        print(f"{M_qam:<10} {papr_db:<12.2f}")
    
    print(f"\n✓ Confirmation: PAPR indépendant de M (écart-type = {np.std(results):.2f} dB)")


# ============================================================================
# QUESTION 5 : Spectre et bande passante
# ============================================================================

def question5_spectrum():
    """Analyse le spectre et calcule la bande passante."""
    print("\n" + "=" * 60)
    print("QUESTION 5 : Spectre et bande passante")
    print("=" * 60)
    
    nFFTSize = DEFAULT_PARAMS['nFFTSize']
    M_qam = DEFAULT_PARAMS['M_qam']
    nSymbol_OFDM = 2**12
    L = nFFTSize // 4
    rolloff = DEFAULT_PARAMS['rolloff']
    sps = DEFAULT_PARAMS['samples_per_symbol']
    
    nbBit_qam = int(np.log2(M_qam))
    nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
    
    tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
    tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
    tSignal, rrcos_filter = rrcos(tx, rolloff, sps)
    
    # Calcul du spectre
    nfft = 2**14
    freq = np.fft.fftfreq(nfft, d=1)
    spectrum = np.abs(np.fft.fft(tSignal, nfft))**2
    spectrum_db = 10 * np.log10(spectrum / np.max(spectrum) + 1e-12)
    
    # Bande passante théorique
    B_theo = (1 + rolloff) / sps
    
    print(f"\nParamètres du filtre RRC:")
    print(f"  Roll-off α = {rolloff}")
    print(f"  Samples per symbol = {sps}")
    print(f"\nBande passante théorique:")
    print(f"  B = (1+α)/T_s = {B_theo:.4f} (normalisée)")
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Spectre complet
    ax1 = axes[0]
    ax1.plot(np.fft.fftshift(freq), np.fft.fftshift(spectrum_db))
    ax1.axvline(x=B_theo/2, color='r', linestyle='--', label=f'B/2 = {B_theo/2:.3f}')
    ax1.axvline(x=-B_theo/2, color='r', linestyle='--')
    ax1.set_xlabel('Fréquence normalisée', fontsize=12)
    ax1.set_ylabel('DSP (dB)', fontsize=12)
    ax1.set_title('Spectre du signal OFDM', fontsize=14)
    ax1.set_xlim([-0.2, 0.2])
    ax1.set_ylim([-80, 5])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Réponse du filtre RRC
    ax2 = axes[1]
    freq_filter = np.fft.fftfreq(len(rrcos_filter), d=1)
    H = np.abs(np.fft.fft(rrcos_filter))
    ax2.plot(np.fft.fftshift(freq_filter), np.fft.fftshift(H/np.max(H)))
    ax2.set_xlabel('Fréquence normalisée', fontsize=12)
    ax2.set_ylabel('|H(f)|', fontsize=12)
    ax2.set_title('Réponse fréquentielle du filtre RRC', fontsize=14)
    ax2.set_xlim([-0.3, 0.3])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rapport/q5_spectrum.png', dpi=150)
    plt.show()


# ============================================================================
# QUESTION 6 : Signal RF
# ============================================================================

def question6_rf_signal():
    """Génère le signal RF et compare le PAPR."""
    print("\n" + "=" * 60)
    print("QUESTION 6 : Signal RF et PAPR")
    print("=" * 60)
    
    nFFTSize = DEFAULT_PARAMS['nFFTSize']
    M_qam = DEFAULT_PARAMS['M_qam']
    nSymbol_OFDM = 2**10
    L = nFFTSize // 4
    fc = 10  # Fréquence porteuse normalisée
    
    nbBit_qam = int(np.log2(M_qam))
    nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
    
    tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
    tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
    tSignal, _ = rrcos(tx, DEFAULT_PARAMS['rolloff'], 
                       DEFAULT_PARAMS['samples_per_symbol'])
    
    # PAPR bande de base
    power_bb = np.abs(tSignal)**2
    papr_bb = 10 * np.log10(np.max(power_bb) / np.mean(power_bb))
    
    # Signal RF
    t = np.arange(len(tSignal))
    tSignal_RF = np.real(tSignal * np.exp(1j * 2 * np.pi * fc * t / len(t) * 100))
    
    # PAPR RF
    power_rf = tSignal_RF**2
    papr_rf = 10 * np.log10(np.max(power_rf) / np.mean(power_rf))
    
    print(f"\nRésultats:")
    print(f"  PAPR bande de base = {papr_bb:.2f} dB")
    print(f"  PAPR RF            = {papr_rf:.2f} dB")
    print(f"  Différence         = {papr_rf - papr_bb:.2f} dB")
    print(f"\n✓ Théorie: PAPR_RF ≈ PAPR_BB + 3 dB")
    
    # Tracé
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Signal bande de base
    ax1 = axes[0]
    ax1.plot(t[:500], np.real(tSignal[:500]), 'b-', alpha=0.7)
    ax1.set_xlabel('Échantillons')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Signal bande de base (PAPR = {papr_bb:.1f} dB)')
    ax1.grid(True, alpha=0.3)
    
    # Signal RF
    ax2 = axes[1]
    ax2.plot(t[:500], tSignal_RF[:500], 'r-', alpha=0.7)
    ax2.set_xlabel('Échantillons')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Signal RF (PAPR = {papr_rf:.1f} dB)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rapport/q6_rf_signal.png', dpi=150)
    plt.show()


# ============================================================================
# RÉCEPTION : Chaîne complète
# ============================================================================

def reception_chain():
    """Simule la chaîne complète avec courbes TEB et EVM."""
    print("\n" + "=" * 60)
    print("RÉCEPTION : Chaîne complète avec TEB et EVM")
    print("=" * 60)
    
    nFFTSize = DEFAULT_PARAMS['nFFTSize']
    M_qam = DEFAULT_PARAMS['M_qam']
    nSymbol_OFDM = 2**10
    L = nFFTSize // 4
    
    nbBit_qam = int(np.log2(M_qam))
    nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
    
    SNR_range = np.arange(0, 31, 2)
    results = {'SNR': [], 'TEB': [], 'EVM': []}
    
    # Émission (une seule fois)
    tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
    tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
    
    print("\nSimulation en cours...")
    
    for snr in SNR_range:
        # Canal AWGN
        rx = chan_awgn(tx, snr)
        
        # Réception
        _, rXmat = removeIGandFFT(rx, nSymbol_OFDM, nFFTSize, L)
        rX, rMsgBin = demapping2bit(rXmat, M_qam)
        
        # Métriques
        evm = calculateEvm(tX, rX)
        teb = np.sum(tMsgBin != rMsgBin) / len(tMsgBin)
        
        results['SNR'].append(snr)
        results['TEB'].append(teb)
        results['EVM'].append(evm)
        
        print(f"  SNR={snr:2d}dB: TEB={teb:.2e}, EVM={evm:.1f}%")
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # TEB
    ax1 = axes[0]
    teb_plot = [max(t, 1e-7) for t in results['TEB']]
    ax1.semilogy(results['SNR'], teb_plot, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('SNR (dB)', fontsize=14)
    ax1.set_ylabel('TEB', fontsize=14)
    ax1.set_title(f'TEB vs SNR ({M_qam}-QAM)', fontsize=16)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_ylim([1e-6, 1])
    
    # EVM
    ax2 = axes[1]
    ax2.plot(results['SNR'], results['EVM'], 'rs-', linewidth=2, markersize=8)
    ax2.set_xlabel('SNR (dB)', fontsize=14)
    ax2.set_ylabel('EVM (%)', fontsize=14)
    ax2.set_title(f'EVM vs SNR ({M_qam}-QAM)', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rapport/reception_teb_evm.png', dpi=150)
    plt.show()


# ============================================================================
# COMPARAISON TEB pour différentes modulations
# ============================================================================

def teb_comparison():
    """Compare le TEB pour différentes modulations M-QAM."""
    print("\n" + "=" * 60)
    print("COMPARAISON TEB : 4-QAM, 16-QAM, 64-QAM")
    print("=" * 60)
    
    nFFTSize = DEFAULT_PARAMS['nFFTSize']
    nSymbol_OFDM = 2**10
    L = nFFTSize // 4
    M_values = [4, 16, 64]
    
    SNR_range = np.arange(0, 31, 2)
    results = {M: [] for M in M_values}
    
    print("\nSimulation en cours...")
    
    for M_qam in M_values:
        nbBit_qam = int(np.log2(M_qam))
        nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
        
        tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
        tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
        
        for snr in SNR_range:
            rx = chan_awgn(tx, snr)
            _, rXmat = removeIGandFFT(rx, nSymbol_OFDM, nFFTSize, L)
            _, rMsgBin = demapping2bit(rXmat, M_qam)
            teb = np.sum(tMsgBin != rMsgBin) / len(tMsgBin)
            results[M_qam].append(teb)
        
        print(f"  {M_qam}-QAM terminé")
    
    # Tracé
    plt.figure(figsize=(10, 6))
    colors = ['b', 'r', 'g']
    markers = ['o', 's', '^']
    
    for i, M in enumerate(M_values):
        teb_plot = [max(t, 1e-7) for t in results[M]]
        plt.semilogy(SNR_range, teb_plot, f'{colors[i]}{markers[i]}-', 
                     linewidth=2, markersize=8, label=f'{M}-QAM')
    
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('TEB', fontsize=14)
    plt.title('Comparaison TEB pour différentes modulations', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', alpha=0.3)
    plt.ylim([1e-6, 1])
    plt.tight_layout()
    plt.savefig('rapport/teb_comparison.png', dpi=150)
    plt.show()
    
    print("\n✓ Les modulations d'ordre élevé nécessitent plus de SNR")


# ============================================================================
# EVM vs SNR (indépendance de M)
# ============================================================================

def evm_vs_snr():
    """Montre l'indépendance de l'EVM par rapport à M."""
    print("\n" + "=" * 60)
    print("EVM vs SNR : Indépendance de l'ordre de modulation")
    print("=" * 60)
    
    nFFTSize = DEFAULT_PARAMS['nFFTSize']
    nSymbol_OFDM = 2**10
    L = nFFTSize // 4
    M_values = [4, 16, 64]
    
    SNR_range = np.arange(0, 31, 2)
    results = {M: [] for M in M_values}
    
    print("\nSimulation en cours...")
    
    for M_qam in M_values:
        nbBit_qam = int(np.log2(M_qam))
        nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
        
        tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
        tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
        
        for snr in SNR_range:
            rx = chan_awgn(tx, snr)
            _, rXmat = removeIGandFFT(rx, nSymbol_OFDM, nFFTSize, L)
            rX, _ = demapping2bit(rXmat, M_qam)
            evm = calculateEvm(tX, rX)
            results[M_qam].append(evm)
        
        print(f"  {M_qam}-QAM terminé")
    
    # Tracé
    plt.figure(figsize=(10, 6))
    colors = ['b', 'r', 'g']
    markers = ['o', 's', '^']
    
    for i, M in enumerate(M_values):
        plt.plot(SNR_range, results[M], f'{colors[i]}{markers[i]}-', 
                 linewidth=2, markersize=8, label=f'{M}-QAM')
    
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('EVM (%)', fontsize=14)
    plt.title('EVM vs SNR pour différentes modulations', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rapport/evm_vs_snr.png', dpi=150)
    plt.show()
    
    print("\n✓ L'EVM est indépendant de l'ordre de modulation")


# ============================================================================
# BONUS : Canal Rayleigh
# ============================================================================

def bonus_rayleigh():
    """Compare les performances AWGN vs Rayleigh."""
    print("\n" + "=" * 60)
    print("BONUS : Canal Rayleigh vs AWGN")
    print("=" * 60)
    
    nFFTSize = DEFAULT_PARAMS['nFFTSize']
    M_qam = DEFAULT_PARAMS['M_qam']
    nSymbol_OFDM = 2**10
    L = nFFTSize // 4
    L_channel = 8
    
    nbBit_qam = int(np.log2(M_qam))
    nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
    
    SNR_range = np.arange(2, 31, 2)
    results = {'AWGN': [], 'Rayleigh': [], 'Rayleigh_EQ': []}
    
    print(f"\nParamètres:")
    print(f"  Longueur canal Rayleigh = {L_channel}")
    print(f"  Préfixe cyclique L = {L}")
    print("\nSimulation en cours...")
    
    for snr in SNR_range:
        # Nouvelle réalisation
        tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
        tx, txMat = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L)
        
        # Canal Rayleigh
        h_real = np.random.randn(L_channel)
        h_imag = np.random.randn(L_channel)
        h = (h_real + 1j * h_imag) / np.sqrt(2)
        h = h / np.sqrt(np.sum(np.abs(h)**2))
        H = np.fft.fft(h, nFFTSize)
        
        # AWGN
        rx_awgn = chan_awgn(tx, snr)
        _, rXmat_awgn = removeIGandFFT(rx_awgn, nSymbol_OFDM, nFFTSize, L)
        _, rMsgBin_awgn = demapping2bit(rXmat_awgn, M_qam)
        teb_awgn = np.sum(tMsgBin != rMsgBin_awgn) / len(tMsgBin)
        
        # Rayleigh sans égalisation
        rx_ray = np.convolve(tx, h, 'full')[:len(tx)]
        rx_ray = chan_awgn(rx_ray, snr)
        _, rXmat_ray = removeIGandFFT(rx_ray, nSymbol_OFDM, nFFTSize, L)
        _, rMsgBin_ray = demapping2bit(rXmat_ray, M_qam)
        teb_ray = np.sum(tMsgBin != rMsgBin_ray) / len(tMsgBin)
        
        # Rayleigh avec égalisation ZF
        rXmat_eq = rXmat_ray / H[:, np.newaxis]
        _, rMsgBin_eq = demapping2bit(rXmat_eq, M_qam)
        teb_eq = np.sum(tMsgBin != rMsgBin_eq) / len(tMsgBin)
        
        results['AWGN'].append(teb_awgn)
        results['Rayleigh'].append(teb_ray)
        results['Rayleigh_EQ'].append(teb_eq)
        
        print(f"  SNR={snr:2d}dB: AWGN={teb_awgn:.2e} | Ray={teb_ray:.2e} | Ray_EQ={teb_eq:.2e}")
    
    # Tracé
    plt.figure(figsize=(10, 6))
    plt.semilogy(SNR_range, [max(t, 1e-7) for t in results['AWGN']], 
                 'bo-', linewidth=2, markersize=8, label='AWGN')
    plt.semilogy(SNR_range, [max(t, 1e-7) for t in results['Rayleigh']], 
                 'rs-', linewidth=2, markersize=8, label='Rayleigh (sans égal.)')
    plt.semilogy(SNR_range, [max(t, 1e-7) for t in results['Rayleigh_EQ']], 
                 'g^-', linewidth=2, markersize=8, label='Rayleigh (avec égal. ZF)')
    
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('TEB', fontsize=14)
    plt.title('Comparaison AWGN vs Rayleigh', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', alpha=0.3)
    plt.ylim([1e-6, 1])
    plt.tight_layout()
    plt.savefig('rapport/bonus_rayleigh.png', dpi=150)
    plt.show()


# ============================================================================
# BONUS : Variation du préfixe cyclique
# ============================================================================

def bonus_cp_variation():
    """Étudie l'effet de la longueur du préfixe cyclique."""
    print("\n" + "=" * 60)
    print("BONUS : Variation du préfixe cyclique")
    print("=" * 60)
    
    nFFTSize = DEFAULT_PARAMS['nFFTSize']
    M_qam = DEFAULT_PARAMS['M_qam']
    nSymbol_OFDM = 2**10
    L_channel = 8
    SNR_test = 30
    
    nbBit_qam = int(np.log2(M_qam))
    nBit = nFFTSize * nbBit_qam * nSymbol_OFDM
    
    L_values = [0, 2, 4, 8, 16]
    results = {'L': [], 'TEB': []}
    
    # Canal fixe
    h_real = np.random.randn(L_channel)
    h_imag = np.random.randn(L_channel)
    h = (h_real + 1j * h_imag) / np.sqrt(2)
    h = h / np.sqrt(np.sum(np.abs(h)**2))
    H = np.fft.fft(h, nFFTSize)
    
    print(f"\nSNR = {SNR_test} dB, L_canal = {L_channel}")
    print(f"\n{'L (CP)':<10} {'TEB':<15} {'Commentaire'}")
    print("-" * 40)
    
    for L_test in L_values:
        tMsgBin, tX, tXmat = bitMapping(nBit, M_qam, nFFTSize)
        tx, _ = ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L_test)
        
        rx = np.convolve(tx, h, 'full')[:len(tx)]
        rx = chan_awgn(rx, SNR_test)
        
        _, rXmat = removeIGandFFT(rx, nSymbol_OFDM, nFFTSize, L_test)
        rXmat_eq = rXmat / H[:, np.newaxis]
        _, rMsgBin = demapping2bit(rXmat_eq, M_qam)
        
        teb = np.sum(tMsgBin != rMsgBin) / len(tMsgBin)
        results['L'].append(L_test)
        results['TEB'].append(teb)
        
        comment = "ISI !" if L_test < L_channel else "OK"
        print(f"{L_test:<10} {teb:<15.4e} {comment}")
    
    # Tracé
    plt.figure(figsize=(8, 5))
    colors = ['red' if L < L_channel else 'green' for L in L_values]
    plt.bar(range(len(L_values)), results['TEB'], color=colors)
    plt.xticks(range(len(L_values)), [f'L={L}' for L in L_values])
    plt.xlabel('Longueur du préfixe cyclique', fontsize=14)
    plt.ylabel('TEB', fontsize=14)
    plt.title(f'Effet du CP (SNR={SNR_test}dB, L_canal={L_channel})', fontsize=16)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('rapport/bonus_cp_variation.png', dpi=150)
    plt.show()
    
    print(f"\n✓ Le CP doit être ≥ {L_channel} pour éviter l'ISI")


##############################################################################
###             Zone 4 : Zone Main
##############################################################################

if __name__ == "__main__":
    
    print_header()
    
    while True:
        print_menu()
        
        try:
            choice = input("\nVotre choix : ").strip()
            
            if choice == '0':
                print("\nMerci d'avoir utilisé ce TP ! Au revoir.\n")
                break
            elif choice == '1':
                question1_papr()
            elif choice == '2':
                question2_papr_vs_fft()
            elif choice == '3':
                question3_papr_vs_mqam()
            elif choice == '4':
                question4_verification()
            elif choice == '5':
                question5_spectrum()
            elif choice == '6':
                question6_rf_signal()
            elif choice == '7':
                reception_chain()
            elif choice == '8':
                teb_comparison()
            elif choice == '9':
                evm_vs_snr()
            elif choice == '10':
                bonus_rayleigh()
            elif choice == '11':
                bonus_cp_variation()
            else:
                print("\n⚠ Choix invalide. Veuillez entrer un nombre entre 0 et 11.")
        
        except KeyboardInterrupt:
            print("\n\nInterruption. Au revoir.\n")
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}")
            print("Veuillez réessayer.")
        
        input("\nAppuyez sur Entrée pour continuer...")