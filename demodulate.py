'''
WARNING: demodulate is currently a suboptimal implementation of square filtering.
Demodulate is now handled by Will's KF or stable_demod (both work). 
Will's KF will pull from this library to demo a naive application of the matched
filter. 
'''

################################### IMPORTS/CONFIGS ###################################
# Run modulate and simulate_channel for required variables
# Import numpy and pyplot from simulate_channel so library code isn't rerun
from simulate_channel import np, plt, channel_output, pulse, Fs, Fc, Fsym, SSNR

################################### DOWNSAMPLING AND FILTER ###################################
sps = int(Fs / Fsym) # symbols per second
time = np.arange(channel_output.shape[0]) / Fs
down_carrier = np.exp(2j * np.pi * (-Fc) * time)

# Mix down to baseband
bb = channel_output * down_carrier

# Matched filter
pulse_normd = pulse / np.linalg.norm(pulse, ord = 2)**2
matched = np.convolve(pulse_normd, bb)

# Downsample to symbol points
symbol_offset = pulse.shape[0] - 1
symbol_outputs = matched[symbol_offset::sps]

################################### PLOTTING ###################################

# Matched filtering fails if we have significant phase drift within a symbol period
# Assuming small phase drift, we can produce a constellation plot
# Create figure and specify subplot layout
fig, axs = plt.subplots(3, 1, figsize=(10, 10))  # Update to use 3 rows now

# Plot the real part of the downconverted baseband signal
axs[0].plot(np.real(bb), label='Real Baseband', color='blue')
axs[0].plot(np.imag(bb), label='Imaginary Baseband', color='red')
axs[0].set_title('Raw Channel Corrupted Baseband Signal')
axs[0].set_xlabel('Sample Index')
axs[0].set_ylabel('Amplitude')
axs[0].legend()
axs[0].grid(True)

# Plot the matched filter output and sample points
axs[1].plot(np.real(matched), label='Real Matched Baseband', color='green')
axs[1].plot(np.imag(matched), label='Imaginary Matched Baseband', color='orange')
axs[1].scatter(np.arange(symbol_offset, len(matched), sps), np.real(matched)[symbol_offset::sps], color='darkred', label='Sampled Symbols', marker='o')
axs[1].set_title('Match Filtered Baseband and Sampling (No KF)')
axs[1].set_xlabel('Sample Index')
axs[1].set_ylabel('Amplitude')
axs[1].legend()
axs[1].grid(True)

# Plot the constellation diagram from the sampled outputs
axs[2].scatter(np.real(symbol_outputs), np.imag(symbol_outputs), color='purple', marker='.')
axs[2].set_title('Raw Constellation After Channel Effects')
axs[2].set_xlabel('Real')
axs[2].set_ylabel('Imaginary')
axs[2].grid(True)
axs[2].axhline(0, color='black', linewidth=0.5)
axs[2].axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()
