# Run modulate and simulate_channel for required variables
# Import numpy and pyplot from simulate_channel so library code isn't rerun
from simulate_channel import np, plt, channel_output, pulse, Fs, Fc, Fsym

time = np.arange(channel_output.shape[0]) / Fs
down_carrier = np.exp(2j * np.pi * (-Fc) * time)

# Mix down to baseband
bb = channel_output * down_carrier

# Matched filter
matched = np.convolve(pulse, bb)

# Downsample to symbols
symbol_offset = pulse.shape[0] - 1
symbol_outputs = matched[symbol_offset::pulse.shape[0]]

# Matched filtering fails if we have significant phase drift within a symbol period
# Assuming small phase drift, we can produce a constellation plot
plt.plot(np.real(symbol_outputs), np.imag(symbol_outputs), '.')
plt.grid(True)
plt.show()
