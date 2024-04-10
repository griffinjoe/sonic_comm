import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
Fc = 900 # Hz
Fs = 44100 # Hz
T = 1 / Fs # s
Fsym = 220.5 # Hz, until features are added ensure int(Fs / Fsym) == (Fs / Fsym)
M = 4 # Modulation order (unitless)
bitcount = 1024 # bits
assert(bitcount % np.log2(M) == 0)
# Different pulse shapes determine how signal energy leaks into sidebands
pulse_shape = 'square'
#pulse_shape = 'raised-cosine'
#pulse_shape = 'root-raised-cosine'
pulse = np.ones((int(Fs / Fsym),)) # V or Pa, real or imaginary axis

# Energy per symbol uses pulse shape energy added across both carrier axes
Es = 2 * np.sum(np.abs(pulse.shape[0])**2)

# Begin modulation
msg = np.random.randint(0, 2, bitcount)
antipodal = 2 * msg - 1
tx_constel = np.reshape(antipodal[0::2] + 1j * antipodal[1::2], (-1, 1))

# Basic pulse shaping is done by convolving the pulse against a zero-spaced impulse
#	train containing the symbols
# Convert from symbol rate to sample rate
rate_convert = np.zeros((tx_constel.shape[0] - 1, int(Fs / Fsym) - 1))
block = np.hstack((tx_constel[:-1], rate_convert))
# Don't zero-pad the last symbol or convolution tail will zero-pad final result
impulse_train = np.concatenate((block.flatten(), tx_constel[-1]))
pulse_shaped = np.convolve(pulse, impulse_train)

# Non-integer-multiple pulse shaping is done in the frequency domain by multiplying
#	with the pulse shape's Fourier Transform with appropriate sample count

# Modulate up to carrier frequency
up_carrier = np.exp(2j * np.pi * Fc * np.arange(pulse_shaped.shape[0]) / Fs)
in_phase = np.real(up_carrier)
quadrature = np.imag(up_carrier)

time = np.arange(pulse_shaped.shape[0]) / Fs
cplx_mod = up_carrier * pulse_shaped
freq = (np.arange(cplx_mod.shape[0]) * Fs / cplx_mod.shape[0]) - (Fs / 2)
T = cplx_mod.shape[0] / Fs
tx_spectrum = np.abs(np.fft.fftshift(np.fft.fft(cplx_mod)))**2 / (Fs**2 * T)
tx_spectrum_db = 10 * np.log10(tx_spectrum)

#plt.plot(freq, tx_spectrum_db)
plt.plot(time, np.real(cplx_mod))
plt.plot(time, np.imag(cplx_mod))
plt.grid(True)
plt.show()

# Separate onto component carrier signals
#	This is required for the live demo, not required for the phase lock feature
#	development.
tx_real = in_phase * np.real(pulse_shaped) + quadrature * np.imag(pulse_shaped)
