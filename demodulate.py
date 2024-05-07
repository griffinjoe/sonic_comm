# Run modulate and simulate_channel for required variables
# Import numpy and pyplot from simulate_channel so library code isn't rerun
from simulate_channel import np, plt, channel_output, pulse, Fs, Fc, Fsym, SSNR

sps = int(Fs / Fsym)
time = np.arange(channel_output.shape[0]) / Fs
down_carrier = np.exp(2j * np.pi * (-Fc) * time)

# Mix down to baseband
bb = channel_output * down_carrier

# Matched filter
pulse_normd = pulse / np.linalg.norm(pulse, ord = 2)**2
matched = np.convolve(pulse_normd, bb)

# Downsample to symbols
symbol_offset = pulse.shape[0] - 1
symbol_outputs = matched[symbol_offset::sps]

# Matched filtering fails if we have significant phase drift within a symbol period
# Assuming small phase drift, we can produce a constellation plot
#plt.subplot(2, 1, 1)
#plt.plot(np.real(bb))
#plt.grid(True)
#plt.subplot(2, 1, 2)
#plt.plot(np.real(matched)); plt.plot(np.arange(symbol_offset, len(matched), sps), np.real(symbol_outputs), 'o')
#plt.grid(True)

#plt.plot(np.real(symbol_outputs), np.imag(symbol_outputs), '.')
#plt.grid(True)
#
#plt.show()
