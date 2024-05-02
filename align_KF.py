# Run mod, sim channel, demod for required variables
# Import numpy and pyplot from demodulate so library code isn't rerun
from demodulate import np, plt, pulse, bb, Fs, Fsym

# Matched filter
sps = int(Fs / Fsym)
pulse_normd = pulse / np.linalg.norm(pulse, ord = 2)**2
matched = np.convolve(pulse_normd, bb)

# Run symbol timing Kalman Filter
# At model_state = 0, symbol indices are (pulse_shape[0] - 1) + n * pulse.shape[0]
ofst_shift = pulse.shape[0] - 1 # Reference point value, rad
model_shift = 0
ofst_rate = pulse.shape[0] # Reference point value, rad / sym
model_rate = 0
model_state = np.array([[model_shift], [model_rate]])
update_matrix = np.array([[1, 1], [0, 1]])
obs_matrix = np.array([[1, 0], [1, 0]])

# Initial model confidence
model_cov = 1e0 * np.identity(model_state.shape[0])
# Finding process noise covariance is a system identification problem
process_noise_cov = 1e-3 * np.identity(model_state.shape[0])
# Finding observation noise covariance is another system identification problem
#	probably related to channel characterization: high-SNR channels will have low
#	symbol noise and therefore low phase observation noise.
# obs_noise_cov = 1 * np.identity(obs_matrix.shape[0])

def update_align_model(obs, predict, update_matrix, obs_matrix, process_noise_cov,
					   obs_noise_cov, predict_cov):
	return (predict, predict_cov)

state_memory = model_state.copy()
ref1_idx = ofst_shift
ref2_idx = ref1_idx + ofst_rate
aligned_syms = [matched[ref1_idx]]
time = [ref1_idx / Fs]
innov_memory = []
t_idx = 0

while (ref2_idx - np.round(update_matrix @ model_state)[0, 0]) < matched.shape[0]:
	model_predict = update_matrix @ model_state
	model_idx = round(model_predict[0, 0])
	aligned_syms.append(matched[ref2_idx - model_idx])
	ref_half_idx = round((ref1_idx + ref2_idx) / 2)
	predict_half_idx = ref_half_idx - model_idx
	half_sym = matched[predict_half_idx]
	sym_diff = (matched[ref2_idx - model_idx] - matched[ref1_idx - model_idx]) / 2
#	print('half_sym =', half_sym)
#	print('sym_diff =', sym_diff)
	predict_cov = update_matrix @ model_cov @ update_matrix.T + process_noise_cov
	# Each observation should be a half-symbol measurement scaled by the difference
	#	between preceding and following symbols.
	innov_real = 0
	innov_imag = 0
	innovation = np.array([[innov_real], [innov_imag]])
#	print('innov =', innovation.T)
	innov_memory.append(innovation[:, 0])
	# Observation noise covariance should be scaled when symbols are close together
	# Repeated symbols don't give new info about timing alignment
	# Only symbol transitions with clear edges give timing alignment information, thus the half-symbols computed above
	obs_noise_cov = 1e3 * np.identity(obs_matrix.shape[0])
	(model_state, model_cov) = update_align_model(innovation, model_predict,
												  update_matrix, obs_matrix,
												  process_noise_cov, obs_noise_cov,
												  predict_cov)
	ref1_idx = ref2_idx
	ref2_idx = ref2_idx + ofst_rate
	t_idx = t_idx + 1
	state_memory = np.concatenate((state_memory, model_state), axis = 1)
	time.append((ref1_idx - model_idx) / Fs)


plt.subplot(2, 2, 1)
plt.plot(np.real(aligned_syms), np.imag(aligned_syms), '.')
plt.title('Constellation')
plt.grid(True)
plt.subplot(2, 2, 2)
plt.plot(np.array(time) * Fs, np.array(state_memory)[0, :])
plt.title('Offset State')
plt.grid(True)
plt.subplot(2, 2, 3)
plt.plot(np.arange(matched.shape[0]), np.real(matched))
plt.plot(np.arange(matched.shape[0]), np.imag(matched))
plt.plot(np.array(time) * Fs, np.real(aligned_syms), 'o')
plt.plot(np.array(time) * Fs, np.imag(aligned_syms), 'o')
plt.title('Aligned Symbols')
plt.grid(True)
plt.subplot(2, 2, 4)
plt.plot(np.array(time[:-1]) * Fs, np.real(np.array(innov_memory)))
plt.title('Innovation')
plt.grid(True)
plt.show()
