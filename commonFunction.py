import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, resample, deconvolve, get_window
from scipy.fft import fft, fftshift
from commpy.modulation import QAMModem
#from commpy.filters import rcosfilter,rrcosfilter
from commpy.channels import awgn
from commpy.utilities import upsample

#######################################################################################
def bitMapping(nBit, M_qam, nFFTSize):
    """
    Generate binary message tMsgBin and apply M-QAM Mapping    
    Args:
        nBit:       number of bits transmitted
        M_qam:      M-QAM modulation order
        nFFTSize:   FFT size (number of subcarriers)    
    Returns:
        tMsgBin:    1D array random binary bit message 
        tX:         1D array after applying qammod to symbol
        tXmat:      2D array reshape the tX with nFFTSize per column
    """

    # Generate binary message tMsgBin: A sequence of 0 and 1
    tMsgBin = (np.random.rand(nBit) > 0.5).astype(int)
    
    # M-QAM Mapping: use QAMModem() from commpy.modulation to generate tX
    # Read https://commpy.readthedocs.io/en/latest/generated/commpy.modulation.QAMModem.html
    modem = QAMModem(M_qam)
    tX = modem.modulate(tMsgBin)

    
    # Generate tXmat: use np.reshape() on tX to create tXmat matrix (series to parallel)
    tXmat = np.reshape(tX, (-1, nFFTSize)).T
    

    return tMsgBin, tX, tXmat

#######################################################################################

def ifftAddIg(tXmat, nSymbol_OFDM, nFFTSize, L):
    """
    Apply iFFT to each symbol in the transmitted OFDM signal 
    Add the cyclic prefix (IG) 
    
    Args:
        tXmat:          2D array reshape the tX with nFFTSize per column
        nSymbol_OFDM:   Number of OFDM symbols
        nFFTSize:       FFT size (number of subcarriers)
        L:              Number of samples in the guard interval

    Returns:
        tx:             1D array reshape the txMat in one column 
        txMat:          2D array after adding IG with nFFTSize+L per column
        tXmat:          2D array reshape the tX with nFFTSize per column
    """

    # Initialize tx as a zero array of size nSymbol_OFDM * (nFFTSize + L)
    tx = np.zeros(nSymbol_OFDM * (nFFTSize + L), dtype=complex)

    # Loop over the number of OFDM symbols
    for n in range(nSymbol_OFDM):
        # Extract the n-th OFDM symbol from tXmat
        ofdm_symbol = tXmat[:, n]
        
        # Apply IFFT of size nFFTSize using np.fft.ifft()
        ifft_symbol = np.fft.ifft(ofdm_symbol, nFFTSize)
        
        # Extract the cyclic prefix (CP) corresponding to the last L samples
        if L > 0:
            cp = ifft_symbol[-L:]
            # Add the CP at the beginning of the OFDM symbol : use np.concatenate()
            symbol_with_cp = np.concatenate([cp, ifft_symbol])
        else:
            # No CP case
            symbol_with_cp = ifft_symbol
        
        # Concatenate the symbols to form the output tx
        tx[n * (nFFTSize + L):(n + 1) * (nFFTSize + L)] = symbol_with_cp
        
    # Construct the matrix txMat using np.reshape()
    txMat = np.reshape(tx, (-1, nFFTSize + L)).T

    return tx,txMat

#######################################################################################
def rrcos(tx, rolloff, samples_per_symbol):
    
    """
    step 1 : UpSample tx
    step 2 : Design the root raised cosine (RRC) filter 
    for upsampling and pulse shaping in digital communication systems

    Args:
        tx :                    Original signal
        rolloff :               Rolloff factor (between 0 and 1)
        samples_per_symbol :    Output samples per symbol (upsampling factor)

    Returns:
        tSignal :      The resulting filtered signal
        rrcosFilter:   Root Raised Cosine Filter
    """

    ## UpSample tx : Use upsample() function from commpy.utilities
    tx_upsampled = upsample(tx, samples_per_symbol)
    

    ## Filter upsampled Signal    
    # Total filter length in samples
    span_in_symbols = 8     # Choose this span for the filter design
    num_taps = span_in_symbols * samples_per_symbol + 1  # Length of the filter in samples
    
    
    # Create the root raised cosine filter
    # Use rrcosfilter with Ts=1 and Fs=samples_per_symbol
    from commpy.filters import rrcosfilter
    time_idx, rrcosFilter = rrcosfilter(num_taps, rolloff, Ts=1, Fs=samples_per_symbol)
    
    
    # Filter our signal to generate tSignal
    # Use np.convolve with mode='same'
    tSignal = np.convolve(tx_upsampled, rrcosFilter, mode='same')

    return tSignal, rrcosFilter

#######################################################################################

def chan_awgn(signal, snr_db):
    """Add AWGN to a signal.    
    Parameters:
    signal (numpy array): The original signal.
    snr_db (float): Desired Signal-to-Noise Ratio in dB.
    
    Returns:
    numpy array: Signal with added AWGN.
    """
    # Use awgn function from commpy.channels
    signal_noisy = awgn(signal, snr_db)
    
    return signal_noisy

#######################################################################################

def removeIGandFFT(rx, nSymbol_OFDM, nFFTSize, L):
    """
    Remove the cyclic prefix (IG) and apply FFT to each symbol in the received OFDM signal.
    
    Args:
        rx: Received signal array (1D complex array)
        nSymbol_OFDM: Number of OFDM symbols
        nFFTSize: FFT size (number of subcarriers)
        L: Length of cyclic prefix (CP)
    
    Returns:
        rxMat: 2D array after removing the CP (cyclic prefix)
        rXmat: 2D array after applying FFT to each symbol
    """
    # Reshape the received signal rx to a matrix where each column is one OFDM symbol (with CP)
    # On utilise la même logique que ifftAddIg : reshape puis transpose
    rxMat = np.reshape(rx, (-1, nFFTSize + L)).T


    # Initialize the matrix rXmat to store the symbols after FFT
    rXmat = np.zeros((nFFTSize, nSymbol_OFDM), dtype=complex)


    ## Loop to remove CP from each symbol
    for n in range(nSymbol_OFDM):
        # Extract the n-th OFDM symbol with CP
        symbol_with_cp = rxMat[:, n]
        
        # Remove the CP (first L samples)
        symbol_without_cp = symbol_with_cp[L:]
        
        # Apply FFT to recover the frequency domain symbols
        rXmat[:, n] = np.fft.fft(symbol_without_cp, nFFTSize)
        
    return rxMat, rXmat

#######################################################################################

def demapping2bit(rXmat, M_qam):
    """
    Demaps the received OFDM signal from M-QAM symbols to a binary bit sequence.
    
    Parameters:
    rXmat: 2D array of received symbols (complex)
    M_qam: QAM modulation order (e.g., 16 for 16-QAM)
    
    Returns:
    rX: Flattened array of received QAM symbols
    rMsgBin: Demapped binary bit sequence
    """
    # Reshape rXmat to a 1D array rX
    rX = rXmat.T.flatten()

    # Create a QAM modem object with the specified M-QAM order
    modem = QAMModem(M_qam)


    # QAM demapping: Convert received QAM symbols to binary sequence (use demodulate())
    rMsgBin = modem.demodulate(rX, demod_type='hard')
    
    return rX, rMsgBin

#######################################################################################

def calculateEvm(reference, received):
    """
    Calculate the Error Vector Magnitude (EVM) in percentage.

    Parameters:
    reference (numpy.ndarray): Ideal reference symbols.
    received (numpy.ndarray): Received symbols.

    Returns:
    float: EVM in percentage.
    """
    # Compute the error vector
    error_vector = received - reference

    # Calculate RMS error
    rms_error = np.sqrt(np.mean(np.abs(error_vector)**2))

    # Calculate RMS of reference
    rms_reference = np.sqrt(np.mean(np.abs(reference)**2))

    # EVM in percentage
    evm = (rms_error / rms_reference) * 100
    
    return evm

#######################################################################################
#######################################################################################

def plotSpectrum(x, fs):
    
    """
    To plot the spectrum
    x    : signal
    fs   : sampling rate

    returns
    pxx : average fft absolute value
    freq : frequency array
    """
    win = 'flattop'  # The used window for FFT ans spectrum plot. Use flatttop
    Nblocks = 20   # number of blocks to perform average spectrum. Use 20.
    # Spectrum computation (two sided)
    N = len(x)
    fft_block_size = int(2**(np.ceil(np.log2(N/Nblocks))-1))
    window = get_window(win, fft_block_size)
    fft_input = np.zeros(fft_block_size, dtype='complex128')

    for n in range(1, Nblocks+1):
        idx = slice((n-1)*fft_block_size, n*fft_block_size)
        fft_block = fft(x[idx] * window) / fft_block_size
        fft_input += (fft_block * np.conj(fft_block))
    fft_input /= Nblocks
    
    # spectrum bettween [-fs/2 fs/2]
    pxx = np.abs(fft_input)
    pxx = fftshift(pxx)
    freq = (np.arange(0, fft_block_size) - fft_block_size/2) * fs / fft_block_size
    
    plt.figure
    plt.plot(freq, 10*np.log10(pxx))
    plt.grid()
    plt.xlabel('Frequency (Hz)'); plt.ylabel('Spectrum (dBm)')
    plt.show()



