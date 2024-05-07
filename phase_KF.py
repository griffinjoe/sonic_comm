# Run mod, sim channel, demod for required variables
# Import numpy and pyplot from demodulate so library code isn't rerun
from demodulate import np, plt, symbol_outputs, Fs, Fsym#, phase_offset, phase_rate

state_memory = np.zeros((2, symbol_outputs.shape[0]))

# At model_shift = 0, symbol phases are oriented at: pi/4, 3pi/4, 5pi/4, 7pi/4
model_shift = 0 # Initial value, rad
model_rate = 0 # Initial value, rad / sym
model_state = np.array([[model_shift], [model_rate]])
update_matrix = np.array([[1, 1], [0, 1]])
obs_matrix = np.array([[1, 0]])

# Initial model confidence
model_cov = np.identity(model_state.shape[0])
# Finding process noise covariance is a system identification problem
process_noise_cov = 1e-2 * np.identity(model_state.shape[0])
# Finding observation noise covariance is another system identification problem
#	probably related to channel characterization: high-SNR channels will have low
#	symbol noise and therefore low phase observation noise.
obs_noise_cov = 1e2 * np.identity(obs_matrix.shape[0])

def update_phase_model(innov, predict, update_matrix, obs_matrix, process_noise_cov,
					   obs_noise_cov, predict_cov):
	net_variance = obs_matrix @ predict_cov @ obs_matrix.T + obs_noise_cov
	gain = predict_cov @ obs_matrix.T @ np.linalg.inv(net_variance)
	model_weight = (np.identity(predict.shape[0]) - gain @ obs_matrix)
	updated_state = predict + gain @ innov # x + K(y - Hx)
	updated_cov = (model_weight @ predict_cov @ model_weight.T + 
				   gain @ obs_noise_cov @ gain.T)
	return (updated_state, updated_cov)

time = np.arange(symbol_outputs.shape[0]) / Fsym
sample_phases = np.angle(symbol_outputs)
phase_aligned = symbol_outputs.copy()
for t_idx in np.arange(time.shape[0]):
	state_memory[:, t_idx] = model_state.flatten()
	t = time[t_idx]
	phi = sample_phases[t_idx]
	# The model prediction simply incorporates the phase rate into the phase
	model_predict = update_matrix @ model_state
	predict_cov = update_matrix @ model_cov @ update_matrix.T + process_noise_cov
	# Each observation should be a phase measurement from the model's nearest point
	#	to the measured constellation symbol.  The constellation has four points,
	#	so the nearest point is never more than pi/4 radians away.
	innov = np.mod(phi - obs_matrix @ model_predict, np.pi / 2) - np.pi / 4
	(model_state, model_cov) = update_phase_model(innov, model_predict, 
												  update_matrix, obs_matrix,
												  process_noise_cov, obs_noise_cov,
												  predict_cov)
	align_phasor = np.exp(-1j * obs_matrix @ model_state)
	phase_aligned[t_idx] = symbol_outputs[t_idx] * align_phasor
	# To prevent numerical overflow, reduce phase with a modulo operation
	# if np.abs(model_state[0, 0]) > np.pi / 2:
	#	model_state[0, 0] = np.mod(model_state[0, 0], np.pi / 2)

plt.subplot(2, 1, 1)
plt.plot(np.real(phase_aligned), np.imag(phase_aligned), '.')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time, state_memory[0])
#for lock_posn in np.arange(-3, 3):
#	plt.plot(time, (phase_offset + lock_posn * np.pi / 2) + time * phase_rate)
plt.grid(True)
plt.show()

# NOTE: Phase Lock Loop (PLL) slightly compresses symbol clusters along phase axis
# As a result, constellation points look stretched in the radial axis
# Fix this by turning down process noise covariance
