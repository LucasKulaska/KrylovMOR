import numpy as np
from scipy.sparse import csc_matrix, bmat

from mor.solver import Solver, sec_to_hour

data_K = np.loadtxt('example/plate and cavity/K.csv', delimiter=',')
data_M = np.loadtxt('example/plate and cavity/M.csv', delimiter=',')
data_lambda_A = np.loadtxt('example/plate and cavity/lambda_A.csv', delimiter=',')

data_H = np.loadtxt('example/plate and cavity/H.csv', delimiter=',')
data_Q = np.loadtxt('example/plate and cavity/Q.csv', delimiter=',')

data_S = np.loadtxt('example/plate and cavity/S.csv', delimiter=',')
data_R = np.loadtxt('example/plate and cavity/R.csv', delimiter=',')

ngl_a = int(np.max(data_H[:,0:2]))
ngl_s = int(np.max(data_K[:,0:2]))

F = np.zeros(ngl_a+ngl_s)
F[3541-1] = 1

K = csc_matrix((data_K[:,2], (data_K[:,0]-1, data_K[:,1]-1)), shape=(ngl_s, ngl_s))
M = csc_matrix((data_M[:,2], (data_M[:,0]-1, data_M[:,1]-1)), shape=(ngl_s, ngl_s))
lambda_A = csc_matrix((data_lambda_A[:,2], (data_lambda_A[:,0]-1, data_lambda_A[:,1]-1)), shape=(ngl_s, ngl_s))
eta = 0.008
K = K*(1 + 1j*eta) + lambda_A

H = csc_matrix((data_H[:,2], (data_H[:,0]-1, data_H[:,1]-1)), shape=(ngl_a, ngl_a))
Q = csc_matrix((data_Q[:,2], (data_Q[:,0]-1, data_Q[:,1]-1)), shape=(ngl_a, ngl_a))

S = csc_matrix((data_S[:,2], (data_S[:,0]-1, data_S[:,1]-1)), shape=(ngl_s, ngl_a))
R = csc_matrix((data_R[:,2], (data_R[:,0]-1, data_R[:,1]-1)), shape=(ngl_a, ngl_s))

C = csc_matrix(([0], ([0], [0])), shape=(ngl_a+ngl_s, ngl_a+ngl_s))

Kc = bmat([[K, S], [None, H]])
Mc = bmat([[M, None], [-R, Q]])

freq = np.arange(1,1000, 1)

#%% Reduced Order Model (ROM)
rom = Solver(Kc, C, Mc, F, freq, moment_matching = 100,
                                 tol_error = 1e-1,
                                 num_freqs = 2)

rom.projection()
solution_rom = rom.get_solution()
freq_error, error = rom.get_error()
expansion_frequencies = rom.get_expension_frequencies()

rom.plot(3541)

