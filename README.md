SIM BASICS:

- Run "python3 stable_demod.py" to get full simulation and some diagonostic plots
- Run "python3 Will_KF.py" to get Will's version of the simulation

OVERVIEW:

- gen_data.py creates the raw bitstream and will report the number of symbols being sent over the channel. 
- modulate.py produces the modulated carrier waveform. It will show its frequency representation and the baseband wave components
- simulate_channel.py injects noise into the carrier waveform and will show what the receiver sees
- demodulate is outdated but will show what a naive attempt to estimate symbols without a Kalman Filter results in
- stable_demod and Will_KF both run the Kalman Filter and show the results
