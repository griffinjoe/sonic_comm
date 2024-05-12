# Run modulate and simulate_channel for required variables
# Import numpy and pyplot from simulate_channel so library code isn't rerun
from simulate_channel import np, plt, channel_output, pulse, Fs, Fc, Fsym, SSNR
from ESN_lib import N_VeryLong, N, DTLTISystem

sps = int(Fs / Fsym)
time = np.arange(channel_output.shape[0]) / Fs
down_carrier = np.exp(2j * np.pi * (-Fc) * time)

# Mix down to baseband
bb = channel_output * down_carrier

# Matched filter
# Matched filter impulse response
pulse_normd = np.array(pulse, dtype = complex) / np.linalg.norm(pulse, ord = 2)**2

# Initialize the modem-state Kalman Filter, tracking phase and symbol alignment
# At model_state = 0, symbol indices are (pulse_shape[0] - 1) + n * pulse.shape[0]
align_shift = pulse.shape[0] - 1 # Reference point value, rad
align_rate = pulse.shape[0] # Reference point value, rad / sym
model_state = np.zeros((4, 1))
update_matrix = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
obs_matrix = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])

# Initial model confidence
Fd = Fc * 1.42 / (343 - 1.42) # Doppler shift due to human walking speed, Hz
model_cov = np.diag([1e-0 * sps / 2, 1e-2 * sps,
					 1e-0 * np.pi / 8, 1e-0 * Fd / Fsym ])**2
# Finding process noise covariance is a system identification problem
process_noise_cov = np.diag([1e-4 / Fsym, 1e-3 / Fsym,
							 1e-6 * Fc, 1e-6 * Fc])
# Alignment observation noise depends on symbol transitions
# Missing transitions make half-symbol observations untrustworthy
# Phase observation noise comes from SSNR
obs_noise_cov = np.diag([1, 1, # Half-symbol variance, to be recomputed at runtime
						 1e2]) # Phase variance, a function of SSNR

model_idx = round(model_state[0, 0])
ref1_idx = align_shift
sym1_idx = ref1_idx - model_idx
model_predict = update_matrix @ model_state
predict_idx = round(model_predict[0, 0])
ref2_idx = ref1_idx + align_rate
sym2_idx = ref2_idx - predict_idx

# Run only one symbol at a time through the matched filter
# The incremental approach allows phase estimator to modify filter input
# Modifiable input needs to be copied from bb to not accidentally also modify bb
mfilt_input = 4 * bb.copy() / (np.std(bb) * np.sqrt(2))
mfilt_output = np.array([], dtype = complex)
symbol_outputs = np.array([], dtype = complex)
symbol_times = np.array([])
mfilt = DTLTISystem(pulse_normd)

# Feed enough data to get first two symbols
mfilt_output = np.append(mfilt_output,
						 mfilt.append_input(mfilt_input[:sym2_idx + 1]))

# Monitor input length to prevent overflow
if mfilt.input_seq.shape[0] >= N_VeryLong:
	# Drop all memory of inputs too old for matched filter length
	mfilt.reset()
	recycle_input = mfilt.input_seq[-mfilt.h.shape[0]:].copy()
	mfilt.set_input(recycle_input)

symbol_outputs = np.append(symbol_outputs, mfilt_output[sym1_idx])
symbol_times = np.append(symbol_times, sym1_idx / Fs)

#scale_filt = DTLTISystem(np.power(0.5, np.arange(20) + 1))
#scale_filt.reset()
#scale = scale_filt.append_input(np.real(np.abs(symbol_outputs[-1:])))
#scale_memory = np.array([scale])

# For first symbol, no prediction information is available.
# Prediction with infinite variance is equivalent.
# Simply replacing model with measurement achieves same results.
model_predict[2, 0] = np.mod(np.angle(symbol_outputs[0]), np.pi / 2) - np.pi / 4
state_memory = model_state.copy()
innov_memory = np.zeros((3, 0))

# Define KF update operation function
def update_KF_model(innov, predict, update_matrix, obs_matrix, process_noise_cov,
					obs_noise_cov, predict_cov):
	net_variance = obs_matrix @ predict_cov @ obs_matrix.T + obs_noise_cov
	gain = predict_cov @ obs_matrix.T @ np.linalg.inv(net_variance)
	model_weight = np.identity(predict.shape[0]) - gain @ obs_matrix
	updated_state = predict + gain @ innov
	updated_cov = (model_weight @ predict_cov @ model_weight.T + 
				   gain @ obs_noise_cov @ gain.T)
	return (updated_state, updated_cov)

# Run Kalman Filter loop until no more input waveform is available
while sym2_idx < mfilt_input.shape[0]:

	symbol_outputs = np.append(symbol_outputs, mfilt_output[sym2_idx])
	symbol_times = np.append(symbol_times, sym2_idx / Fs)
	half_idx = round((sym1_idx + sym2_idx) / 2)
	half_sym = mfilt_output[half_idx]
	sym_diff = (symbol_outputs[-1] - symbol_outputs[-2]) / 2

	model_predict = update_matrix @ model_state
	predict_cov = update_matrix @ model_cov @ update_matrix.T + process_noise_cov
	# Each observation should be a half-symbol measurement scaled by the difference
	#	between preceding and following symbols.
	innov_real = np.sign(np.real(sym_diff)) * np.real(half_sym) * (sps / 2)
	innov_imag = np.sign(np.imag(sym_diff)) * np.imag(half_sym) * (sps / 2)
	innov_phase = np.mod(np.angle(symbol_outputs[-1]), np.pi / 2) - np.pi / 4
	innovation = np.array([[innov_real], [innov_imag], [innov_phase]])
#	print('model_predict =\n', model_predict.T)
#	print('innov =', innovation.T)
	innov_memory = np.append(innov_memory, innovation)
	# Use observation noise covariance to skip updates missing symbol transitions
	trust = np.array([np.real(sym_diff), np.imag(sym_diff)])**2
	obs_noise_cov = np.diag(np.append(1e2 / np.maximum(trust, 1e-30), 1e2))

#	print('obs_noise_cov =\n', obs_noise_cov)
	(model_state, model_cov) = update_KF_model(innovation, model_predict, 
											   update_matrix, obs_matrix,
											   process_noise_cov, obs_noise_cov,
											   predict_cov)
	
	ref1_idx = ref2_idx
	ref2_idx = ref2_idx + align_rate
	sym1_idx = sym2_idx
	sym2_idx = ref2_idx - round(model_state[0, 0])
	state_memory = np.concatenate((state_memory, model_state), axis = 1)
	state_memory[1, -1] = model_cov[0, 0]
	state_memory[3, -1] = model_cov[2, 2]
	if sym2_idx >= mfilt_input.shape[0]:
		break
	sample_times = np.arange(sym2_idx - sym1_idx) / (sym2_idx - sym1_idx)
	input_phasor = np.exp(-2j * np.pi * model_state[3, 0] * sample_times
						  - 1j * model_state[2, 0])

    # Frequency mismatch correction between matched filter input and matched filter itself
	mfilt_input[sym1_idx+1:sym2_idx+1] = (mfilt_input[sym1_idx+1:sym2_idx+1] *
										  input_phasor)# / scale[-1]
	mfilt_output = np.append(mfilt_output,
							 mfilt.append_input(mfilt_input[sym1_idx + 1:
															sym2_idx + 1]))
	# Monitor input length to prevent overflow
	if mfilt.input_seq.shape[0] >= N_VeryLong:
		# Drop all memory of inputs too old for matched filter length
		mfilt.reset()
		recycle_input = mfilt.input_seq[-mfilt.h.shape[0]:].copy()
		mfilt.set_input(recycle_input)


plt.subplot(2, 3, 1)
plt.plot(np.real(symbol_outputs), np.imag(symbol_outputs), '.')
plt.title('Constellation')
plt.grid(True)
plt.subplot(2, 3, 2)
plt.plot(symbol_times * Fs, state_memory[0, :])
plt.plot(symbol_times * Fs, state_memory[2, :])
plt.title('States')
plt.grid(True)
plt.subplot(2, 3, 3)
plt.plot(np.arange(mfilt_output.shape[0]), np.real(mfilt_output))
plt.plot(np.arange(mfilt_output.shape[0]), np.imag(mfilt_output))
plt.plot(symbol_times * Fs, np.real(symbol_outputs), 'o')
plt.plot(symbol_times * Fs, np.imag(symbol_outputs), 'o')
plt.title('Aligned Symbols')
plt.grid(True)
plt.subplot(2, 3, 4)
plt.plot(symbol_times * Fs, state_memory[1, :])
plt.plot(symbol_times * Fs, state_memory[3, :])
plt.title('Variances')
plt.grid(True)
#plt.subplot(2, 3, 5)
#plt.plot(symbol_times[:-1] * Fs, scale_memory)
#plt.grid(True)

ASM = np.array(list(f'{0x1ACFFC1D:0>32b}'), dtype = int)
ASM_antipodal = 2 * ASM - 1
ASM_constel = ASM_antipodal[0::2] + 1j * ASM_antipodal[1::2]
# Real-valued transmission
#ASM_cmp = np.convolve((ASM_constel[::-1]), symbol_outputs)
# Complex-valued transmission
ASM_cmp = np.convolve(np.conj(ASM_constel[::-1]), symbol_outputs)
frm_thresh = np.max(np.abs(ASM_cmp)) * 0.85
print('frm_thresh =', frm_thresh, ', frm_cnt =\n', np.sum(np.abs(ASM_cmp) > frm_thresh))
plt.subplot(2, 3, 6)
plt.plot(np.abs(ASM_cmp))
plt.grid(True)
plt.show()


