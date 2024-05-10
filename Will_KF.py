################################### IMPORTS/CONFIGS ###################################

# Run mod, sim channel, demod for required variables
# Import numpy and pyplot from demodulate so library code isn't rerun
from demodulate import np, plt, symbol_outputs, Fs, Fc, Fsym, pulse, sps # phase_offset, phase_rate

Fd = Fc * 1.42 / (343 - 1.42) # Doppler shift due to human walking speed, Hz

################################### SYSTEM DYNAMICS ###################################

update_matrix = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
obs_matrix = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])

# Finding process noise covariance is a system identification problem
process_noise_cov = np.diag([1e-4 / Fsym, 1e-3 / Fsym,
							 1e-6 * Fc, 1e-6 * Fc])
# Alignment observation noise depends on symbol transitions
# Missing transitions make half-symbol observations untrustworthy
# Phase observation noise comes from SSNR
obs_noise_cov = np.diag([1, 1, # Half-symbol variance, to be recomputed at runtime
						 1e2]) # Phase variance, a function of SSNR

# Initial Predictions
# At model_state = 0, symbol indices are (pulse_shape[0] - 1) + n * pulse.shape[0]
model_state = np.array([0],[0],[0],[0])
# Initial model confidence
model_cov = np.diag([1e-0 * sps / 2, 1e-2 * sps,
					 1e-0 * np.pi / 8, 1e-0 * Fd / Fsym ])**2

align_shift = pulse.shape[0] - 1 # Reference point value, rad
align_rate = pulse.shape[0] # Reference point value, rad / sym
model_idx = round(model_state[0, 0])
ref1_idx = align_shift
sym1_idx = ref1_idx - model_idx
model_predict = update_matrix @ model_state
predict_idx = round(model_predict[0, 0])
ref2_idx = ref1_idx + align_rate
sym2_idx = ref2_idx - predict_idx

if __name__ == '__main__':
    print("Working!")
