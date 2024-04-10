# Run mod, sim channel, demod for required variables
# Import numpy and pyplot from demodulate so library code isn't rerun
from demodulate import np, plt, symbol_outputs, Fsym

# At model_shift = 0, symbol phases are oriented at: pi/4, 3pi/4, 5pi/4, 7pi/4
model_shift = 0 # Modelled constellation rotation initialization value
model_rate = 2 * np.pi / 13 # Modelled constellation rotation rate initial value
model_state = np.array([[model_shift], [model_rate]])
update_matrix = np.array([[1, 1 / Fsym], [0, 1]])
obs_matrix = np.array([[1, 0]])

# Finding process noise covariance is a system identification problem
process_noise_cov = np.diag([1, 1])
# Finding observation noise covariance is another system identification problem
#	probably related to channel characterization: high-SNR channels will have low
#	symbol noise and therefore low phase observation noise.
obs_noise_cov = np.diag([1, 1])

# @Will, you should rewrite the following function to implement a Kalman Filter
def update_phase_model(obs, predict, update_matrix, obs_matrix, process_noise_cov,
					   obs_noise_cov):
	return predict

time = np.arange(symbol_outputs.shape[0]) / Fsym
symbol_phases = np.angle(symbol_outputs)
phase_aligned = symbol_outputs
for t_idx in np.arange(time.shape[0]):
	t = time[t_idx]
	phi = symbol_phases[t_idx]
	# The model prediction simply incorporates the phase rate into the phase
	model_predict = update_matrix @ model_state
	# Each observation should be a phase measurement from the model's nearest point
	#	to the measured constellation symbol.  The constellation has four points,
	#	so the nearest point is never more than pi/8 radians away.
	obs = np.mod(phi - model_predict[0], np.pi / 4) - np.pi / 8
	model_state = update_phase_model(obs, model_predict, update_matrix, obs_matrix,
									 process_noise_cov, obs_noise_cov)
	align_phasor = np.exp(-1j * model_state[0])
	phase_aligned[t_idx] = symbol_outputs[t_idx] * align_phasor

plt.plot(np.real(phase_aligned), np.imag(phase_aligned), '.')
plt.grid(True)
plt.show()
