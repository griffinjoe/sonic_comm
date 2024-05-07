import numpy as np
import matplotlib.pyplot as plt
import binascii

M = 4
bitcount = 1000 # bits
assert(bitcount % np.log2(M) == 0)
preamble_len = 200
dense_flips = False
while not dense_flips:
	preamble_1s = np.ones((int(preamble_len / 2),))
	preamble_0s = np.zeros((int(preamble_len / 2),))
	preamble = np.random.permutation(np.concatenate((preamble_1s, preamble_0s)))
	preamble_block = np.reshape(preamble, (-1, 10, int(np.log2(M))))
	preamble_variances = np.var(preamble_block, axis = 1)
	dense_flips = np.all(np.min(preamble_variances) > 0)

# Attached synchronization marker
ASM = np.array([list(f'{0x1ACFFC1D:0>32b}')], dtype = int)

# Error correction code specifications
n = 8
k = 4
#P = np.random.randint(0, 2, (k, n - k))
P = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0]])
G = np.concatenate((np.identity(k), P), axis = 1)
H = np.concatenate((np.mod(-P.T, 2), np.identity(n - k)), axis = 1)

cw_cnt = (bitcount - preamble_len) / (ASM.shape[1] + n)

# Encode random data, attach ASM, prepend preamble
msg_blk = np.random.randint(0, 2, (k, 2 * int(cw_cnt / 2)))
byte_blk = np.array(np.reshape(msg_blk.T, (-1, 8)), dtype = str)
dec_list = [int(''.join(byte_row), 2) for byte_row in byte_blk]
print(''.join(list(map(chr, dec_list))))

#msg_string = '''You should not complete the 6.3010 course evaluation.'''
#bytelist = list(map(ord, msg_string))
#bitstring = ''.join([format(byte, '#010b')[2:] for byte in bytelist])
#msg_blk = np.reshape(np.array(list(bitstring), dtype = int), (-1, k)).T
# No error correction feature is currently implemented
# The data is encoded, so this should be manageable at the receiver end
cw_blk = np.array(np.mod(G.T @ msg_blk, 2), dtype = int)
asm_blk = ASM.T @ np.ones((1, msg_blk.shape[1]))
frm_blk = np.concatenate((asm_blk, cw_blk), axis = 0)
datastream = np.array(np.reshape(frm_blk.T, (-1,)), dtype = int)
bitstream = np.concatenate((preamble, datastream))
print('Symbol_count =\n', len(bitstream) / 2)
