################################### IMPORTS/CONFIGS ###################################

# Run mod, sim channel, demod for required variables
# Import numpy and pyplot from demodulate so library code isn't rerun
from demodulate import np, plt, pulse, bb, Fs, Fsym

################################### INITIALIZATIONS ###################################

# Matched filter
sps = int(Fs / Fsym)
pulse_normd = pulse / np.linalg.norm(pulse, ord = 2)**2
matched = np.convolve(pulse_normd, bb) # filtered version of the received signal

# Symbol Timing Constants
# At model_state = 0, symbol indices are (pulse_shape[0] - 1) + n * pulse.shape[0]
ofst_shift = pulse.shape[0] - 1 # Reference point value, rad
ofst_rate = pulse.shape[0] # Reference point value, rad / sym

# System Predictions
initial_shift = 0.0
initial_rate = 0.0
predicted_state = np.array([[initial_shift], [initial_rate]])
predicted_covariance = 1e0 * np.identity(predicted_state.shape[0])

# Index Predictions
predicted_idx = 0

# System Dynamics
update_matrix = np.array([[1, 1], [0, 1]]) # Process Evolution
obs_matrix = np.array([[1, 0]]) # Observation Evolution
process_noise_cov = 1e-3 * np.identity(predicted_state.shape[0]) # Process Noise
obs_noise_cov = 1 * np.identity(obs_matrix.shape[0]) # Observation Noise
# Finding observation noise covariance is another system identification problem
#	probably related to channel characterization: high-SNR channels will have low
#	symbol noise and therefore low phase observation noise.

# Simulation Variables
ref1_idx = ofst_shift # Symbol Lower Bound
ref2_idx = ref1_idx + ofst_rate # Symbol Upper Bound

# Simulation Cache
state_memory = predicted_state.copy() # Want to analyze how predicted state evolves
innov_memory = [] # Want to be able to analyze how innovation evolves
aligned_syms = [matched[ref1_idx]]

# Aligned symbol point
time = [ref1_idx / Fs] # Use sampling frequency to convert to time axis
t_idx = 0

# Useful Algebraic Structures
id = np.identity(predicted_covariance.shape[0]) 

################################### KALMAN FILTER ###################################
'''
- Kalman Filter to update hidden state estimates based on observations.
- Includes both predict and update step.
- Changes global predictions in place. 

Inputs:
obs (np.ndarray): Observed timing offset

Returns:
NULL
'''
def KF_Process(obs):

	# Python Scope Alignment
	global predicted_state, predicted_covariance, update_matrix, obs_matrix, process_noise_cov, obs_noise_cov, id
	
	# Prediction Step
	predicted_state = update_matrix.dot(predicted_state)
	predicted_covariance = update_matrix.dot(predicted_covariance).dot(update_matrix.T) + process_noise_cov

	# Innovation and Kalman Gain
	innovation = obs - obs_matrix.dot(predicted_state)
	inv = np.linalg.inv(obs_matrix.dot(predicted_covariance).dot(obs_matrix.T) + obs_noise_cov)
	kalman_gain = predicted_covariance.dot(obs_matrix.T).dot(inv)

	# Update State
	predicted_state += kalman_gain.dot(innovation)
	predicted_covariance = (id - kalman_gain.dot(obs_matrix)).dot(predicted_covariance)

	# Cache Innovation for Debugging
	innov_memory.append(innovation)

	return

################################### SIMULATION ###################################

# While loop breaks when next symbol boundary is out of sample, indicating done processing
while ((ref2_idx - np.round(predicted_state)[0, 0]) < matched.shape[0]):

	# Predict Timing Offset and adjust sampling
	predicted_idx = round(predicted_state[0, 0]) # Predicted timing offset 
	aligned_syms.append(matched[ref2_idx - predicted_idx]) # Log Adjusted sample point

	# Compute adjusted half index
	ref_half_idx = round((ref1_idx + ref2_idx) / 2) # Raw half index
	predict_half_idx = ref_half_idx - predicted_idx # Adjusted half index
	half_sym = matched[predict_half_idx] # Adjusted Half Symbol 
	sym_diff = (matched[ref2_idx - predicted_idx] - matched[ref1_idx - predicted_idx]) / 2 # Adjusted Symbol Difference

	# Each observation should be a half-symbol measurement scaled by the difference
	#	between preceding and following symbols.
	obs = np.array([np.abs(matched[ref_half_idx])]) # Observe magnitude of raw half symbol

	# Observation noise covariance should be scaled when symbols are close together
	# Repeated symbols don't give new info about timing alignment
	# Only symbol transitions with clear edges give timing alignment information, thus the half-symbols computed above
	if np.abs(sym_diff) > 1e-2:
		obs_noise_cov = 1 * np.identity(obs_matrix.shape[0]) # symbols different, trust observation
	else: 
		obs_noise_cov = 1e3 * np.identity(obs_matrix.shape[0]) # symbols similar, do not trust observation

	# Kalman Filter Updated Estimations
	KF_Process(obs)

	# Update for next iteration
	ref1_idx = ref2_idx
	ref2_idx += ofst_rate
	t_idx = t_idx + 1
	state_memory = np.concatenate((state_memory, predicted_state), axis = 1)
	time.append((ref1_idx - predicted_idx) / Fs)

################################### PLOTTING ###################################
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
plt.plot(np.array(time[:-1]) * Fs, np.array(innov_memory).reshape(-1))
plt.title('Innovation')
plt.grid(True)
plt.show()
