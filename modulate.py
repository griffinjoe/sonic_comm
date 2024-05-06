############################## IMPORTS AND CONFIGS ################################
import numpy as np
import matplotlib.pyplot as plt

# SIMULATION PARAMETERS
Fc = 5*5*7*3*3 # Carrier Frequency, in Hz
Fs = 44100 # Sampling Frequency, in Hz
Fsym = 5*5*7 # Symbol rate, in Hz, 
# !!! ensure int(Fs / Fsym) == (Fs / Fsym) and int(Fc / Fsym) == (Fc / Fsym)a !!!
M = 4 # Modulation Order, QPSK means it is 4, i.e. 4 symbols
align_rate = 1 # samples lost per symbol
align_offset = 50 # samples, mod rate-shifted symbol width
phase_offset = 0 # rad
phase_rate_per_sym = 0 # rad / sym
phase_rate = phase_rate_per_sym * Fsym # rad / s

# User Interface
print("KEY SIMULATION PARAMETERS")
print(f'Align Rate: {align_rate}')
print(f'Align Offset: {align_offset}')
print(f'Phase Offset: {phase_offset}')
print(f'Phase Rate (Per Symbol): {phase_rate_per_sym}')

############################## SIMULATION ################################

# Square pulse for now, possibly add others later
# Pulse shape determines how energy leaks into sidebands
sps = int(Fs / Fsym)
pulse = np.ones((sps,)) # V or Pa, real or imaginary axis

# Energy per symbol uses pulse shape energy added across both carrier axes
Es = 2 * np.linalg.norm(pulse, ord = 2)**2

# Begin modulation
bitstream = np.random.randint(0, 2, (1000,))
antipodal = 2 * bitstream - 1
tx_constel = np.reshape(antipodal[0::2] + 1j * antipodal[1::2], (-1, 1))

# Basic pulse shaping is done by convolving the pulse against a zero-spaced impulse
#	train containing the symbols
# Convert from symbol rate to sample rate
rate_convert = np.zeros((tx_constel.shape[0] - 1, sps - 1))
block = np.hstack((tx_constel[:-1], rate_convert))
# Don't zero-pad the last symbol or convolution tail will zero-pad final result
impulse_train = np.concatenate((block.flatten(), tx_constel[-1]))
pulse_shaped = np.convolve(pulse, impulse_train)

# For symbol alignment testing without phase offsets, do symbol timing adjustment
#	here, before up-mixing.
rate_block = np.reshape(pulse_shaped, (-1, sps))[:, :(sps - align_rate)]
rate_shifted = np.reshape(rate_block, (-1,))
pulse_shifted = np.concatenate((rate_shifted[align_offset:],
								rate_shifted[:align_offset]), axis = 0)

# Non-integer-multiple pulse shaping is done in the frequency domain by multiplying
#	with the pulse shape's Fourier Transform with appropriate sample count

# Modulate up to carrier frequency
time = np.arange(pulse_shifted.shape[0]) / Fs
phase = phase_offset + phase_rate * time
phasor = np.exp(1j * phase)
up_carrier = np.exp(2j * np.pi * Fc * time) * phasor
in_phase = np.real(up_carrier)
quadrature = -np.imag(up_carrier) # Negative takes conjugate of carrier
# Conjugate comes from |x[n]|^2 = <x[n], x^*[n]> =/= <x[n], x[n]>

time = np.arange(pulse_shifted.shape[0]) / Fs
cplx_mod = up_carrier * pulse_shifted
freq = (np.arange(cplx_mod.shape[0]) * Fs / cplx_mod.shape[0]) - (Fs / 2)
T = cplx_mod.shape[0] / Fs
tx_spectrum = np.abs(np.fft.fftshift(np.fft.fft(cplx_mod)))**2 / (Fs**2 * T)
tx_spectrum_db = 10 * np.log10(tx_spectrum)

# Separate onto component carrier signals
#	This is required for the live demo, not required for the initial development.
tx_real = in_phase * np.real(pulse_shifted) + quadrature * np.imag(pulse_shifted)
