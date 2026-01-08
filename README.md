# 📡 Modélisation d'une Chaîne OFDM

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8+-green.svg)](https://matplotlib.org/)

Projet de Travaux Pratiques sur la **modélisation complète d'une chaîne de transmission OFDM** (Orthogonal Frequency-Division Multiplexing) incluant émission, réception et analyse de performances en canal AWGN et Rayleigh.

## 📋 Description

Ce projet implémente une chaîne OFDM complète avec :
- **Émission** : Mapping M-QAM, IFFT, ajout du préfixe cyclique, filtrage RRC
- **Réception** : Suppression du CP, FFT, égalisation, démapping
- **Canaux** : AWGN et Rayleigh multi-trajets
- **Métriques** : PAPR, TEB (Taux d'Erreur Binaire), EVM (Error Vector Magnitude)

## 🎯 Objectifs du TP

### Partie Émission
1. ✅ Calcul du PAPR du signal OFDM
2. ✅ Étude PAPR vs taille de FFT
3. ✅ Étude PAPR vs ordre de modulation M-QAM
4. ✅ Analyse du spectre et bande passante
5. ✅ Signal RF et PAPR en bande RF

### Partie Réception
6. ✅ Chaîne complète avec courbes TEB et EVM
7. ✅ Comparaison TEB pour différentes modulations
8. ✅ Indépendance de l'EVM par rapport à l'ordre de modulation

### Partie Bonus
9. ✅ Canal sélectif de Rayleigh avec égalisation Zero-Forcing
10. ✅ Rôle du préfixe cyclique (ISI)

## 🚀 Installation

### Prérequis
- Python 3.13+
- pip

### Étapes

1. **Cloner le dépôt**
```bash
git clone https://github.com/profirst30/modelisation-d-une-chaine-OFDM.git
cd modelisation-d-une-chaine-OFDM
```

2. **Créer un environnement virtuel**
```bash
python -m venv .venv
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## 📂 Structure du Projet

```
tp1_OFDM/
│
├── commonFunction.py              # Fonctions principales OFDM
├── main_tp.py                     # Script principal avec menu interactif
│
├── ofdmTranceiver.py             # Émetteur OFDM (Question 1)
├── ofdmTranceiver_reception.py   # Chaîne complète émission-réception
│
├── question2_papr_vs_fftsize.py  # PAPR vs taille FFT
├── question3_papr_vs_mqam.py     # PAPR vs M-QAM (N=64)
├── question4_papr_verification.py # PAPR vs M-QAM (N=128)
├── question5_spectrum.py          # Spectre et bande passante
├── question6_rf_signal.py         # Signal RF
│
├── question_teb_comparison.py     # Comparaison TEB
├── question4_5_evm_snr.py        # EVM vs SNR
│
├── bonus_rayleigh_cp.py          # Bonus : Canal Rayleigh + CP
│
├── rapport/                       # Figures et rapport LaTeX
│   ├── main.tex
│   └── *.png
│
└── README.md
```

## 💻 Utilisation

### Menu Interactif (Recommandé)

Lancer le script principal avec menu :
```bash
python main_tp.py
```

Menu disponible :
```
==================================================
                    MENU PRINCIPAL
--------------------------------------------------
  [1] Question 1 : Calcul du PAPR
  [2] Question 2 : PAPR vs Taille FFT
  [3] Question 3 : PAPR vs Ordre M-QAM (N=64)
  [4] Question 4 : Vérification PAPR (N=128)
  [5] Question 5 : Spectre et bande passante
  [6] Question 6 : Signal RF et PAPR
--------------------------------------------------
  [7] Réception : Chaîne complète avec TEB/EVM
  [8] Comparaison TEB : 4-QAM, 16-QAM, 64-QAM
  [9] EVM vs SNR (indépendance de M)
--------------------------------------------------
  [10] BONUS : Canal Rayleigh vs AWGN
  [11] BONUS : Variation du préfixe cyclique
--------------------------------------------------
  [0] Quitter
```

### Scripts Individuels

Chaque question peut aussi être exécutée séparément :
```bash
python ofdmTranceiver.py                    # Question 1
python question2_papr_vs_fftsize.py         # Question 2
python question3_papr_vs_mqam.py            # Question 3
python ofdmTranceiver_reception.py          # Réception complète
python bonus_rayleigh_cp.py                 # Bonus
```

## 📊 Résultats Clés

### PAPR
- PAPR typique : **11-12 dB** pour N=64, 16-QAM
- **Indépendant** de l'ordre de modulation M
- Saturation pour grandes tailles de FFT

### TEB et EVM
- EVM **indépendant** de M (dépend seulement du SNR)
- TEB diminue avec SNR, mais dépend fortement de M
- 64-QAM nécessite ~10 dB de SNR de plus que 4-QAM pour même TEB

### Canal Rayleigh
- Dégradation forte sans égalisation (TEB ≈ 50%)
- Égalisation ZF améliore les performances
- **Préfixe cyclique critique** : L ≥ L_canal pour éviter l'ISI

## 🔬 Fonctions Principales (`commonFunction.py`)

| Fonction | Description |
|----------|-------------|
| `bitMapping()` | Génération bits aléatoires + modulation M-QAM |
| `ifftAddIg()` | IFFT + ajout du préfixe cyclique |
| `rrcos()` | Filtrage Root Raised Cosine |
| `chan_awgn()` | Canal AWGN avec SNR réglable |
| `removeIGandFFT()` | Suppression CP + FFT |
| `demapping2bit()` | Démapping M-QAM → bits |
| `calculateEvm()` | Calcul de l'EVM (%) |

## 📈 Paramètres par Défaut

```python
nFFTSize = 64              # Taille FFT
M_qam = 16                 # Modulation 16-QAM
nSymbol_OFDM = 2^12        # Nombre de symboles OFDM
L = nFFTSize // 4          # Préfixe cyclique (16)
rolloff = 0.3              # Roll-off RRC
samples_per_symbol = 8     # Sur-échantillonnage
```

## 🛠️ Technologies Utilisées

- **Python 3.13**
- **NumPy** : Calculs numériques
- **Matplotlib** : Visualisations
- **SciPy** : Traitement du signal (filtres)
- **scikit-commpy** : Modulation M-QAM
- **LaTeX** : Rapport scientifique

## 📝 Rapport

Le rapport LaTeX complet est disponible dans `rapport/main.tex` avec :
- Analyses théoriques
- Résultats de simulation
- Graphiques et tableaux
- Conclusions et perspectives

