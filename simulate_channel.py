################################### IMPORTS/CONFIGS ###################################

# Run modulate for required variables
# Import numpy and pyplot from modulate so library code isn't rerun
# Also import a few transmitter configuration variables to be used in demodulate
from modulate import np, plt, Fs, Fsym, Fc, Es, pulse, cplx_mod, tx_real, phase_offset, phase_rate_per_sym

SSNR_dB = 20 # Symbol Signal to Noise Ratio, dB
phase_rate = phase_rate_per_sym * Fsym # rad / s
# Channel fade indicates phase rate direction with spiral direction
fade_rate = 1 # loss / s
attenuation = 1

################################### CHANNEL EFFECTS ###################################

# Compute AWGN sample-wise variance
SSNR = np.power(10, SSNR_dB / 10) # Linear scale, unitless
No = Es / SSNR # V^2 or Pa^2, noise energy per symbol
noise_stddev = np.sqrt(No) # V or Pa, real or imaginary axis

# Add noise
noise_amps = noise_stddev * np.random.normal(0, 1, (cplx_mod.shape[0],))
noise_phasors = np.exp(1j * np.pi * np.random.rand(cplx_mod.shape[0]))
noise = noise_amps * noise_phasors

# Apply channel phasor
# Intentional double phase offset?
time = np.arange(cplx_mod.shape[0]) / Fs # s
phase = phase_offset + phase_rate * time
phasor = np.exp(1j * phase)

fade = attenuation * np.power(fade_rate, time)
#channel_output = np.real(noise) + np.imag(noise) + 2 * tx_real * fade * phasor
channel_output = noise + cplx_mod * fade * phasor

################################### PLOTTING ###################################
fig, axs = plt.subplots(1, 1, figsize=(10, 12))  

# Plot the final channel output
axs.plot(time, np.real(channel_output), color='blue')
axs.set_title('Carrier Waveform After Channel Effects')
axs.set_xlabel('Time (s)')
axs.set_ylabel('Amplitude')
axs.grid(True)

plt.tight_layout()
plt.show()
