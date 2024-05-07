import numpy as np
import matplotlib.pyplot as plt

# constant for long response
N_VeryLong = 1000
# Constant for ESN training
N = 500

class DTLTISystem:

    def __init__(self, h=np.asarray([1])):
        self.set_impulse_response(h)
        self.reset()

    def reset(self):
        # used for single steps
        self.current_pt = 0
        # length of the output sequence to report
        self.length = 0
        self.input_seq = np.asarray([], dtype = self.h.dtype)
        self.output_seq = np.asarray([], dtype = self.h.dtype)

    def set_impulse_response(self, h):
        self.h = h.copy()


    def get_freq_response(self, plot= False):
        self.H = np.fft.fft(self.h)
        if plot:
            plt.figure()
            plt.plot(np.abs(self.H))
            plt.title("Frequency Response")
            plt.show()
        return self.H

    def set_input (self, input_seq):

        assert (input_seq.shape[0] <= N_VeryLong)
        self.input_seq = input_seq.reshape([-1,]).copy()

    def fresh_exec(self):

        '''
        run system with the entire input sequence and zero in memory.
        '''

        ilength = self.input_seq.shape[0]
        hlength = self.h.shape[0]

        self.length = ilength + hlength -1

        self.current_pt = ilength
        self.output_seq = np.zeros([self.length,])

        for i in range(ilength):
            self.output_seq[i:i+hlength] += self.h * self.input_seq[i]

        if self.length > N_VeryLong:
            self.output_seq = self.output_seq[:N_VeryLong]

        return (self.output_seq[:self.current_pt])



    def run(self, input_seq, plot= False):
        self.set_input (input_seq)
        self.fresh_exec()

        if plot:
            plt.figure()
            plt.plot(self.output_seq)
            plt.show()


    def append_input (self, input_seq):

        '''
        incremental run of system with appended input sequence
        starting with the current memory
        move self.current_pt
        return the new output up to the new current_pt
        keep future samples in the memory of self.output_seq,
        '''

        assert (input_seq.shape[0] <= N_VeryLong)
        self.input_seq = np.append(self.input_seq, input_seq)

        i_length = input_seq.shape[0]
        h_length = self.h.shape[0]
        o_length = self.output_seq.shape[0]

        self.length = self.input_seq.shape[0] + self.h.shape[0]-1


        self.output_seq = np.append(self.output_seq, np.zeros(self.length - o_length,))

        start_pt = self.current_pt

        for i in range(i_length):
            x = input_seq[i]
            self.output_seq[self.current_pt : self.current_pt+h_length] += x * self.h
            self.current_pt += 1

        return(self.output_seq[start_pt: self.current_pt])

	
class DTStateSpace(DTLTISystem):
    '''
    Implement a discrete time state-space model
    
    Parameters
        * A, B, C, D, Vn, Vw,
    
            for model 
        
            q[n+1] = A q[n] + B x[n] + \sqrt{Vn} Normal(0,1)
            y[n]   = C q[n] + D x[n] + \sqrt{Vw} Normal(0,1)
        
        
        * state_dim
        * init_state
        * q
        * state_seq
            record the initial state, current state, history of state 
            each state variable is real valued with state_dim dimensions
            
        * input_seq
            record the input sequence x[n], 
            also specifies the length of the simulation
        * output_seq
        * output_length
            output_length by default is the same as the input seq, but can
            be set longer, for example when evaluating impulse response.
            
    Methods 
        * set_input()
            hard copy of hte input sequence

    
    '''

    def __init__(self, A, B, C, D, Vn= 0, Vw=0):
        
        self.state_dim = A.shape[0]
        
        self.A = A
        self.B = B.reshape([-1, 1]).copy()
        self.C = C.reshape([1, -1]).copy()
        self.D = np.asarray(D).reshape([1,1])
        self.Vn = Vn
        self.Vw = Vw
        
        assert self.A.shape == (self.state_dim, self.state_dim)
        assert self.B.shape == (self.state_dim, 1)
        assert self.C.shape == (1, self.state_dim)
        assert self.D.shape == (1, 1)
        
        self.reset()

        h = self.findImpulseResponse()
        super().__init__(h)
        
    def reset(self):
        
        self.state_seq = np.zeros([self.state_dim, 1])
        self.set_init_state(np.zeros([self.state_dim, 1]))
        self.q = np.zeros([self.state_dim, 1])   # the current state
        super().reset()
    
        
    def set_init_state (self, init_state):
        assert init_state.shape == (self.state_dim, 1) 
        self.state_seq = init_state.copy()
        self.q = init_state.copy() # last update by xx
            
    
    def findImpulseResponse(self, length = 200):
        
        '''
        Record output with an impulse as input, upto 200 samples. 
        '''

        impulse = np.zeros([10,])
        impulse[0] = 1
        self.SP_run(impulse, output_length = length)

        impulse_response = self.output_seq.copy()
        self.reset()
        return impulse_response
    
        
    def SP_SingleStep(self, x):
        '''
        running directly from a state space model, can be compared with the 
        super().append_input() as controlled run of the system while keeping 
        the memory
        '''        
        self.input_seq = np.append(self.input_seq, x)
        
        pt = self.current_pt
        y = self.C @ self.q + self.D * x + np.random.normal(0, self.Vw) 
        
        self.current_pt += 1
        self.length += 1
        
        self.output_seq = np.append(self.output_seq, y)
        
        self.q = self.A @ self.q + self.B * x \
                + np.random.normal(0, self.Vn, size = self.q.shape)
        
        self.state_seq = np.append(self.state_seq, self.q, 1)
        return y
    
    def SP_run(self, input_seq, output_length = -1):
        
        if output_length == -1:
            output_length = N_VeryLong
        

        
        for i in range (output_length):
            if i < input_seq.shape[0]:
                x = input_seq[i]
            else:
                x = 0
            y = self.SP_SingleStep(x)


class basicESN:

    def __init__(self, input, output, pole, delay, lr):

        # dimension
        self.input = input
        self.pole = pole
        self.delay = delay
        self.output = output

        # parameters
        node = self.pole + self.delay + 1  # all poles, all delay modules, and a direct pass
        self.Win = np.zeros((node, self.input))
        self.Wesn = np.zeros((pole, pole), dtype=complex)
        self.Wout = np.zeros((output, node))

        # train
        self.lr = lr
        self.conj_state = None
        self.delay_memory = None

    def sort_complex_and_real(self, arr):
        complex_numbers = [x for x in arr if np.iscomplex(x)]
        real_numbers = [x for x in arr if not np.iscomplex(x)]

        # check whether each complex pole has its conjugate part
        complex_set = set(complex_numbers)
        for c in complex_numbers:
            if np.conj(c) not in complex_set:       # illegal if no conjugate
                return -1
        complex_numbers.sort(key=lambda x: (x.real, -abs(x.imag)))

        # merge the complex- and real- number sets
        real_numbers.sort()
        sorted_array = np.array(complex_numbers + real_numbers)
        return sorted_array

    def set_Wesn(self, Wesn):
        # reservoir weights
        ordered_Wesn = self.sort_complex_and_real(Wesn)
        if isinstance(ordered_Wesn, int):
            print("ESN error: illegal poles")
            sys.exit()
        [row, col] = np.diag_indices_from(self.Wesn)
        self.Wesn[row, col] = ordered_Wesn

    def set_Win(self, Win):
        # input weights
        self.Win = Win

    def set_Wout(self, Wout):
        # output weights
        if Wout.ndim < 2:
            Wout = np.reshape(Wout, (-1, len(Wout)))
        self.Wout = Wout
	
    def get_obs_coeffs(self):
        # Complex-valued modal observation coefficients in complex-conjugate pairs
        p = self.pole
        t = np.array(self.Wout[:, :p].T, dtype = complex)
        t[0:p:2, 0] = ((-self.Wout[0, 1:p:2] - 1j * self.Wout[0, 0:p:2]) * (self.Wesn[0:p:2, 0:p:2]**2))
        t[1:p:2, 0] = np.conj(t[0:p:2, 0])
        return t

    def _forward(self, current_input, before_state):
        node = self.pole + self.delay + 1
        after_inputlayer = np.dot(self.Win, current_input)
        preactive_state = np.zeros(node)

        num_complexpair = int(np.sum(np.imag(np.diagonal(self.Wesn)) != 0) / 2)

        # 1. complex poles
        for n in range(num_complexpair):
            preactive_state[2*n] = -2 * np.real(self.conj_state[2*n] * self.Wesn[2*n, 2*n] ** 2 + self.Wesn[2*n, 2*n] * after_inputlayer[2*n])
            preactive_state[2*n+1] = -2 * np.imag(self.conj_state[2*n+1] * self.Wesn[2*n+1, 2*n+1] ** 2 - np.conj(self.Wesn[2*n+1, 2*n+1]) * after_inputlayer[2*n+1])

            self.conj_state[2*n] = self.conj_state[2*n] * self.Wesn[2*n, 2*n] + after_inputlayer[2*n]
            self.conj_state[2*n+1] = self.conj_state[2*n+1] * self.Wesn[2*n+1, 2*n+1] + after_inputlayer[2*n+1]
        # 2. real poles
        for n in range(2*num_complexpair, self.pole):
            preactive_state[n] = before_state[n] * self.Wesn[n, n].real + after_inputlayer[n]
        # 3. delay modules
        for n in range(self.delay):
            preactive_state[n + self.pole] = self.delay_memory[n, n]
            # update delay memory
            self.delay_memory[n, 1:] = self.delay_memory[n, :-1]
            self.delay_memory[n, 0] = after_inputlayer[n + self.pole]
        # 4. direct pass
        preactive_state[node - 1] = after_inputlayer[node - 1]

        # output
        current_state = preactive_state  # linear activation
        current_output = np.dot(self.Wout, current_state)
        return current_state, current_output

    def fresh_exec(self, inputs):
        # running the ESN once with a zero state
        node = self.pole + self.delay + 1
        self.conj_state = np.zeros(node, dtype=complex)             # clear status of complex pole
        self.delay_memory = np.zeros((self.delay, self.delay))      # clear delay memory

        # if T==1, transform (size_input,) into (size_input,1):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        esn_states = np.zeros((node, inputs.shape[1]))
        esn_outputs = np.zeros((self.output, inputs.shape[1]))

        # zero state
        esn_states[:, 0], esn_outputs[:, 0] = self._forward(inputs[:, 0], np.zeros(esn_states[:, 0].shape))
        # after starting
        for n in range(1, inputs.shape[1]):
            esn_states[:, n], esn_outputs[:, n] = self._forward(inputs[:, n], esn_states[:, n - 1])

        return esn_states, esn_outputs

    def train_Wout(self, inputs, labels, plot_loss):
        # dimension of inputs: (self.input, length of sequence, experiment times)
        # dimension of labels: (self.output, length of sequence, experiment times)
        error_list = np.zeros(labels.shape[2])

        for epoch in range(labels.shape[2]):
            input, label = inputs[:, :, epoch], labels[:, :, epoch]
            [sample, output] = self.fresh_exec(input)

            # gradient descend
            gradient = 2*np.dot(output - label, sample.T)
            norm_gradient = gradient / np.linalg.norm(gradient)    # normalization
            self.Wout -= norm_gradient * self.lr

            # display performance
            train_error = np.sqrt(np.mean((label - np.dot(self.Wout, sample)) ** 2))
            # print('epoch: %d, training loss: %f\n' % (epoch, train_error))
            error_list[epoch] = train_error

            if np.mod(epoch, 200) == 0:
                self.lr /= 10

        if plot_loss:
            plt.figure()
            plt.plot(error_list, color='k')
            plt.xlabel('Epoches')
            plt.ylabel('l_2 loss')
            plt.show()
