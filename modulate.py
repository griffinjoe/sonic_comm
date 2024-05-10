################################### IMPORTS/CONFIGS ###################################
from gen_data import np, plt, bitstream, M

# Transmission Parameters
Fc = 5*5*7*3*3 # Carrier Frequency in Hz
Fs = 44100 # Sampling frequency in Hz
Fsym = 5*5*7 # Symbol Rate in Hz, ensure int(Fs / Fsym) == (Fs / Fsym) and int(Fc / Fsym) == (Fc / Fsym)
sps = int(Fs / Fsym) # Samples Per Symbol

# Problem Parameters
align_rate = 1 # samples lost per symbol
align_offset = 0 # samples, mod rate-shifted symbol width
phase_offset = 0 # rad
phase_rate_per_sym = 0 # rad / sym
phase_rate = phase_rate_per_sym * Fsym # rad / s

# Reports to User
print('Alignment Rate:', align_rate)
print('Alignment Offset:', align_offset)
print('Phase Offset:', phase_offset)
print('Phase Rate (Per Symbol):', phase_rate_per_sym)

################################### PULSE SHAPING ###################################

# Square pulse for now, possibly add others later
# Pulse shape determines how energy leaks into sidebands
pulse = np.ones((sps,)) # V or Pa, real or imaginary axis

# Energy per symbol uses pulse shape energy added across both carrier axes
Es = 2 * np.linalg.norm(pulse, ord = 2)**2

# Antipodal Bitstream to Complex Constellation Symbols
antipodal = 2 * bitstream - 1
tx_constel = np.reshape(antipodal[0::2] + 1j * antipodal[1::2], (-1, 1)) # constellation stream

# Basic pulse shaping is done by convolving the pulse against a zero-spaced impulse
#	train containing the symbols
# Convert from symbol rate to sample rate
rate_convert = np.zeros((tx_constel.shape[0] - 1, sps - 1))
block = np.hstack((tx_constel[:-1], rate_convert))
# Don't zero-pad the last symbol or convolution tail will zero-pad final result
impulse_train = np.concatenate((block.flatten(), tx_constel[-1]))
pulse_shaped = np.convolve(pulse, impulse_train)

# Non-integer-multiple pulse shaping is done in the frequency domain by multiplying
#	with the pulse shape's Fourier Transform with appropriate sample count

# Timing Corruption
rate_block = np.reshape(pulse_shaped, (-1, sps))[:, :(sps - align_rate)]
rate_shifted = np.reshape(rate_block, (-1,))
pulse_shifted = np.concatenate((rate_shifted[align_offset:],
								rate_shifted[:align_offset]), axis = 0)

# Phase Corruption
time = np.arange(pulse_shifted.shape[0]) / Fs
phase = phase_offset + phase_rate * time 
phasor = np.exp(1j * phase)
################################### CARRIER MODULATION ###################################

# Modulate up to carrier frequency
up_carrier = np.exp(2j * np.pi * Fc * time) * phasor
in_phase = np.real(up_carrier)
quadrature = -np.imag(up_carrier) # Negative takes conjugate of carrier
# Conjugate comes from |x[n]|^2 = <x[n], x^*[n]> =/= <x[n], x[n]>

cplx_mod = up_carrier * pulse_shifted # Complex carrier wave
freq = (np.arange(cplx_mod.shape[0]) * Fs / cplx_mod.shape[0]) - (Fs / 2)
T = cplx_mod.shape[0] / Fs
tx_spectrum = np.abs(np.fft.fftshift(np.fft.fft(cplx_mod)))**2 / (Fs**2 * T)
tx_spectrum_db = 10 * np.log10(tx_spectrum)

# Separate onto component carrier signals
#	This is required for the live demo, not required for the initial development.
# This is the waveform sent over the channel. 
tx_real = in_phase * np.real(pulse_shifted) + quadrature * np.imag(pulse_shifted)

################################### PLOTTING ###################################
fig, axs = plt.subplots(3, 1, figsize=(10, 10))  # Create a figure and a grid of subplots

# Plot the real part of the pulse-shaped signal
axs[0].plot(time, np.real(pulse_shifted), label='Real Part')
axs[0].set_title('Phase/Timing Corrupted Pulse Phasor (Real)')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')
axs[0].legend()
axs[0].grid(True)

# Plot the imaginary part of the pulse-shaped signal
axs[1].plot(time, np.imag(pulse_shifted), label='Imaginary Part', color='red')
axs[1].set_title('Phase/Timing Corrupted Pulse Phasor (Imaginary)')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Amplitude')
axs[1].legend()
axs[1].grid(True)

# Plot the frequency spectrum of the transmitted signal
axs[2].plot(freq, tx_spectrum_db)
axs[2].set_title('Frequency Spectrum of Raw Carrier Waveform')
axs[2].set_xlabel('Frequency (Hz)')
axs[2].set_ylabel('Magnitude (dB)')
axs[2].grid(True)

plt.tight_layout()
plt.show()