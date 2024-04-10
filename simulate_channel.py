# Run modulate for required variables
# Import numpy and pyplot from modulate so library code isn't rerun
# Also import a few transmitter configuration variables to be used in demodulate
from modulate import np, plt, Es, Fs, Fsym, cplx_mod, pulse, Fc

time = np.arange(cplx_mod.shape[0]) / Fs
SSNR_dB = 30 # Symbol Signal to Noise Ratio, dB
phase_offset = -2 * np.pi / 12 # rad
phase_rate = 2 * np.pi / 80 # rad / s
# Channel fade indicates phase rate direction with spiral direction
fade_rate = 0.9 # loss / s

SSNR = np.power(10, SSNR_dB / 10) # Linear scale, unitless
No = Es / SSNR # V^2 or Pa^2, noise energy per symbol
noise_stddev = np.sqrt(No / int(Fs / Fsym)) # V or Pa, real or imaginary axis

noise_amps = noise_stddev * np.random.normal(0, 1, (cplx_mod.shape[0],))
noise_phasors = np.exp(1j * np.pi * np.random.rand(cplx_mod.shape[0]))
noise = noise_amps * noise_phasors

phase = phase_offset + phase_rate * 2 * np.pi * time
phasor = np.exp(1j * phase)
fade = np.power(fade_rate, time)

channel_output = noise + cplx_mod * phasor * fade
