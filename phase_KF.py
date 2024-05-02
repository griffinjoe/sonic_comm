################################### IMPORTS/CONFIGS #################################
# Run mod, sim channel, demod for required variables
# Import numpy and pyplot from demodulate so library code isn't rerun
from demodulate import np, plt, symbol_outputs, Fsym

################################### INITIALIZATIONS #################################

# STATE DYNAMICS
update_matrix = np.array([[1, 1], [0, 1]]) # Matrix A in Will'S notation
obs_matrix = np.array([[1, 0]]) # Matrix B in Will'S Notation

# NOISE DYNAMICS

# Finding process noise covariance is a system identification problem
process_noise_cov = np.diag([1, 1])

# Finding observation noise covariance is another system identification problem
# 	probably related to channel characterization: high-SNR channels will have low
# 	symbol noise and therefore low phase observation noise.
obs_noise_cov = np.array([[1]])  

# ESTIMATORS 
initial_shift = 0 # Initial value for phase offset
initial_rate =  np.pi / 64 # Initial value for phase rate
predicted_state = np.array([[initial_shift], [initial_rate]])
# Using assumed covariance parameter as initial estimate
predicted_covariance = np.diag([1, 1])

# OTHER ALGEBRA

# ID matrix gets used in update for covariance estimate; defining
#	it here saves some clutter later.
id = np.identity(predicted_covariance.shape[0]) 

################################### KALMAN FILTER #################################
'''
- Kalman Filter to update hidden state estimates based on observations.
- Includes both predict and update step.
- Changes global predictions in place. 

Inputs:
phi (np.ndarray): Demodulator's estimated phase shift

Returns:
NULL
'''
def KF_Process(phi):
	
	# Scope handling
	global predicted_state, predicted_covariance, obs_matrix, update_matrix, process_noise_cov, obs_noise_cov

	# Prediction Step
	predicted_state = update_matrix.dot(predicted_state) 
	predicted_covariance = update_matrix.dot(predicted_covariance).dot(update_matrix.T) + process_noise_cov

	# Compute Innovation and Kalman Gain
	raw_innovation = phi - obs_matrix.dot(predicted_state) 
	adjusted_innovation = np.mod(raw_innovation, np.pi / 2) - np.pi / 4
	inverse = np.linalg.inv((obs_matrix.dot(predicted_covariance).dot(obs_matrix.T) + obs_noise_cov))
	projection = predicted_covariance.dot(obs_matrix.T)
	kalman_gain = projection.dot(inverse)

	# Update Step 
	predicted_state = predicted_state + kalman_gain.dot(adjusted_innovation)
	predicted_covariance = (id - kalman_gain.dot(obs_matrix)).dot(predicted_covariance)

	return

if __name__ == "__main__":

	################################### SIMULATION #################################

	time = np.arange(symbol_outputs.shape[0]) / Fsym
	symbol_phases = np.angle(symbol_outputs)
	phase_aligned = symbol_outputs

	for t_idx in np.arange(time.shape[0]):

		# Demodulator's Estimate of Phase Shift
		t = time[t_idx] 
		phi = symbol_phases[t_idx] 

		# Run Kalman Filter and update predictions
		KF_Process(phi)

		# Map phase constellation to visualization
		align_phasor = np.exp(-1j * (predicted_state[0]))
		phase_aligned[t_idx] = symbol_outputs[t_idx] * align_phasor

	################################### PLOTTING #################################

	# Plot the constellation after phase KF corrects the phase output
	plt.plot(np.real(phase_aligned), np.imag(phase_aligned), '.')
	plt.title('Aligned Constellation Points')
	plt.xlabel('Real')
	plt.ylabel('Imaginary')
	plt.grid(True)
	plt.show()

	

	