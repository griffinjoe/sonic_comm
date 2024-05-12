######################################### IMPORTS / CONFIGS #########################################

# Run modulate and simulate_channel for required variables
# Import numpy and pyplot from simulate_channel so library code isn't rerun
from simulate_channel import np, plt, channel_output, pulse, Fs, Fc, Fsym, SSNR
from ESN_lib import N_VeryLong, N, DTLTISystem
import demodulate # demodulate, while outdated, contains some informative plots

# Useful Scalars 
sps = int(Fs / Fsym) # Samples per Symbol
Fd = Fc * 1.42 / (343 - 1.42) # Doppler shift due to human walking speed, Hz. Distorts carrier.

# Logs for plotting
time = np.arange(channel_output.shape[0]) / Fs # Time axis converted to seoncds
mfilt_output = np.array([], dtype = complex) # Stores matched filter output
symbol_outputs = np.array([], dtype = complex) # Stores predicted symbols
symbol_times = np.array([]) # Stores predicted symbol times
state_memory = np.array([[0], [0], [0], [0]]) # Stores hidden state predictions
innov_memory = np.zeros((3, 0)) # Stores innovation history

# Kalman Filter Function
def KF_Process(innov):

	# Scope Alignment
    global model_state, model_cov, update_matrix, process_noise_cov, obs_matrix, obs_noise_cov, state_memory
    
    # Prediction Step
    model_state = update_matrix @ model_state
    model_cov = update_matrix @ model_cov @ update_matrix.T + process_noise_cov
    
    # Compute Kalman Gain/Weights
    net_variance = obs_matrix @ model_cov @ obs_matrix.T + obs_noise_cov
    net_precision = np.linalg.inv(net_variance)
    kalman_gain = model_cov @ obs_matrix.T @ net_precision
    model_weight = np.identity(model_state.shape[0]) - kalman_gain @ obs_matrix

	# Update Step, Covariance Matrix uses Joseph Form for stability
    model_state = model_state + kalman_gain @ innov
    model_cov = (model_weight @ model_cov @ model_weight.T + 
                   kalman_gain @ obs_noise_cov @ kalman_gain.T)
	
	# Log states
    state_memory = np.concatenate((state_memory, model_state), axis = 1)
    state_memory[1, -1] = model_cov[0, 0]
    state_memory[3, -1] = model_cov[2, 2]

######################################### SYSTEM DYNAMICS #########################################

# LTI Matrices
update_matrix = np.array([
						[1, 1, 0, 0], 
						[0, 1, 0, 0], 
						[0, 0, 1, 1], 
						[0, 0, 0, 1]
						])

obs_matrix = np.array([
					[1, 0, 0, 0], 
					[1, 0, 0, 0], 
					[0, 0, 1, 0]
					])

# Finding process noise covariance is a system identification problem
process_noise_cov = np.diag([1e-4 / Fsym, 
							 1e-3 / Fsym,
							 1e-6 * Fc, 
							 1e-6 * Fc])

obs_noise_cov = np.diag([1,
						 1, # Holder for half-symbol variance, will be recomputed at runtime
						 1e2]) # Phase variance, a function of SSNR

# Initialize Hidden State/Uncertainty Prediction 
model_state = np.array([[0], # Timing Offset
						[0], # Timing Offset Rate
						[0], # Phase Offset
						[0]]) # Phase Offset Rate

# Good variance initialization is a system identification problem
model_cov = np.diag([1e-0 * sps / 2, 
					 1e-2 * sps,
					 1e-0 * np.pi / 8, 
					 1e-0 * Fd / Fsym ])**2

# Predict first and second symbol offset based on initial conditions
model_idx = round(model_state[0, 0]) # First predicted offset
model_predict = update_matrix @ model_state
predict_idx = round(model_predict[0, 0]) # Second predicted offset

# First and Second Symbols observations will be assigned based on pulse width. 
align_shift = pulse.shape[0] - 1 # Assign first symbol to end of pulse
align_rate = pulse.shape[0] # Distance between symbols is pulse width

# Make initial filter corrections
ref1_idx = align_shift # First "observed" symbol time
sym1_idx = ref1_idx - model_idx # Corrected first symbol time, based on predicted offset
ref2_idx = ref1_idx + align_rate # Second "observed" symbol time
sym2_idx = ref2_idx - predict_idx # Corrected second symbol time, based on predicted offset

######################################### PREPROCESSING #########################################

# Raw demodulation
down_carrier = np.exp(2j * np.pi * (-Fc) * time) # Complex Conjugate of Carrier Wave
bb = channel_output * down_carrier # Retrieved baseband from down carrier
mfilt_input = 4 * bb.copy() / (np.std(bb) * np.sqrt(2)) # Normalized baseband for numerical stability
pulse_normd = np.array(pulse, dtype = complex) / np.linalg.norm(pulse, ord = 2)**2 # Matched Filter impulse response
mfilt = DTLTISystem(pulse_normd) # Class initialization

# Convolve impulse response with input up to predicted second sample time, log filtered output
mfilt_output = np.append(mfilt_output,
						 mfilt.append_input(mfilt_input[:sym2_idx + 1]))

# Overflow prevention
if mfilt.input_seq.shape[0] >= N_VeryLong:
	mfilt.reset() # Clear memory
	recycle_input = mfilt.input_seq[-mfilt.h.shape[0]:].copy() # Truncate input
	mfilt.set_input(recycle_input) # Input is now only relevant previous samples and unprocessed ones

# First symbol is sampled from filtered baseband at first timing predictions
symbol_outputs = np.append(symbol_outputs, mfilt_output[sym1_idx]) # Record symbol
symbol_times = np.append(symbol_times, sym1_idx / Fs) # Record when the symbol was taken

# Given first symbol, no prediction information is available.
# "Cold Start": assign first phase offset prediction as observed deviation from a symbol point
model_predict[2, 0] = np.mod(np.angle(symbol_outputs[0]), np.pi / 2) - np.pi / 4 
    
######################################### SIMULATION #########################################

# Run Kalman Filter loop until no more input waveform is available
while sym2_idx < mfilt_input.shape[0]:

	# Log the symbol at the front end of the symbol window
	symbol_outputs = np.append(symbol_outputs, mfilt_output[sym2_idx])
	symbol_times = np.append(symbol_times, sym2_idx / Fs)

	# Find half symbol in the window and compute symbol difference
	half_idx = round((sym1_idx + sym2_idx) / 2)
	half_sym = mfilt_output[half_idx]
	sym_diff = (symbol_outputs[-1] - symbol_outputs[-2]) / 2

	# Each timing innovation should be a half-symbol measurement scaled by the time
	#	between preceding and following symbols.
	innov_real = np.sign(np.real(sym_diff)) * np.real(half_sym) * (sps / 2)
	innov_imag = np.sign(np.imag(sym_diff)) * np.imag(half_sym) * (sps / 2)
	innov_phase = np.mod(np.angle(symbol_outputs[-1]), np.pi / 2) - np.pi / 4 # Offset from quadrant symbol point
	innovation = np.array([[innov_real], [innov_imag], [innov_phase]])
	innov_memory = np.append(innov_memory, innovation) # Log innovation

	# Use observation noise covariance to skip updates missing symbol transitions
	trust = np.array([np.real(sym_diff), np.imag(sym_diff)])**2 # Near 0 if no symbol change
	obs_noise_cov = np.diag(np.append(1e2 / np.maximum(trust, 1e-30), 1e2))

	# Run Kalman Filter
	KF_Process(innovation)
	
	# Update reference window
	ref1_idx = ref2_idx
	ref2_idx = ref2_idx + align_rate

	# Adjust symbol window based on estimated offset, in-sample validity check
	sym1_idx = sym2_idx
	sym2_idx = ref2_idx - round(model_state[0, 0])
	if sym2_idx >= mfilt_input.shape[0]:
		break

	# Adjust the corrupted baseband using the information from the KF
	sample_times = np.arange(sym2_idx - sym1_idx) / (sym2_idx - sym1_idx)
	# Adjustment phasor, adjust for current phase offset and propagate over sampling window
	input_phasor = np.exp(-2j * np.pi * model_state[3, 0] * sample_times
						  - 1j * model_state[2, 0])

    # Use current phase offset estimate to adjust window and propagate offset rate estimate. 
	mfilt_input[sym1_idx+1:sym2_idx+1] = (mfilt_input[sym1_idx+1:sym2_idx+1] *
										  input_phasor)
	# Apply pulse filter to adjusted window, then log the output
	mfilt_output = np.append(mfilt_output,
							 mfilt.append_input(mfilt_input[sym1_idx + 1:
															sym2_idx + 1]))
	
	# Overflow prevention
	if mfilt.input_seq.shape[0] >= N_VeryLong:
		mfilt.reset() # Clear memory
		recycle_input = mfilt.input_seq[-mfilt.h.shape[0]:].copy() # Recycle relevant samples
		mfilt.set_input(recycle_input) # Truncate memory

######################################### PLOTTING #########################################
plt.figure(figsize=(18, 12))

# 1. Constellation Diagram
plt.subplot(2, 3, 1)
plt.scatter(np.real(symbol_outputs), np.imag(symbol_outputs), color='blue', marker='.')
plt.title('Symbol Constellation')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.axis('square')
plt.grid(True)

# 2. States - Time evolution of state estimates
plt.subplot(2, 3, 2)
plt.plot(symbol_times * Fs, state_memory[0, :], label='Timing Offset', marker='o')
plt.plot(symbol_times * Fs, state_memory[2, :], label='Phase Offset', marker='x')
plt.title('State Estimates Over Time')
plt.xlabel('Time')
plt.ylabel('State Value')
plt.legend()
plt.grid(True)

# 3. Aligned Symbols - Matched filter output and detected symbols, spans full width
plt.subplot(2, 1, 2)  # This places it across the full width of the figure
plt.plot(np.arange(mfilt_output.shape[0]), np.real(mfilt_output), color = 'green', label='Filtered Baseband (Real)')
plt.plot(np.arange(mfilt_output.shape[0]), np.imag(mfilt_output), color = 'orange', label='Filtered Baseband (Imag)')
plt.scatter(symbol_times * Fs, np.real(symbol_outputs), color='darkred', label='Real Symbols', marker='o')
plt.scatter(symbol_times * Fs, np.imag(symbol_outputs), color='darkred', label='Imag Symbols', marker='x')
plt.title('Matched Filter Output and Aligned Symbols')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 5. Frame Sync Detection - Using Acquisition Sequence Mark (ASM)
ASM = np.array(list(f'{0x1ACFFC1D:0>32b}'), dtype = int)
ASM_antipodal = 2 * ASM - 1
ASM_constel = ASM_antipodal[0::2] + 1j * ASM_antipodal[1::2]
ASM_cmp = np.convolve(np.conj(ASM_constel[::-1]), symbol_outputs)
frm_thresh = np.max(np.abs(ASM_cmp)) * 0.85
plt.subplot(2, 3, 3)
plt.plot(np.abs(ASM_cmp), label='Frame Sync Correlation')
plt.axhline(y=frm_thresh, color='r', linestyle='--', label='Threshold')
plt.title('Frame Synchronization Detection')
plt.xlabel('Sample Index')
plt.ylabel('Correlation Strength')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

