import numpy as np
from time import time
from scipy.linalg import solve, svd
from scipy.sparse.linalg import splu
from numpy.linalg import norm
from math import pi, ceil
from statistics import median_low
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from mor.mgs import get_orthonormal
from mor.soar import Soar


def sec_to_hour(t):
    day = t // (24 * 3600)
    t = t % (24 * 3600)
    hour = t // 3600
    t %= 3600
    minutes = t // 60
    t %= 60
    seconds = t
    if day > 0:
        return ("%d dia(s) %dh %dm %ds" % (day, hour, minutes, seconds))
    elif hour > 0:
        return ("%dh %dm %ds" % (hour, minutes, seconds))
    elif minutes > 0:
        return ("%dm %ds" % (minutes, seconds))
    return ("%ds" % (seconds))

class Solver:
    def __init__(self, stiffness, damping, mass, force, freq, **kwargs):
        self.K = stiffness
        self.C = damping
        self.M = mass
        self.b = force
        self.freq = np.sort(np.unique(freq))

        if len(self.b.shape) > 1:
            self.force_by_freq = {f: self.b[:, i] for i, f in enumerate(self.freq)}
            self.freq_dependence = True
        else:
            self.freq_dependence = False

        # Config
        self.moment_matching = kwargs.get('moment_matching', 10)
        self.tol_svd = kwargs.get('tol_svd', 1e-6)
        self.svd = kwargs.get('svd', False)
        self.tol_error = kwargs.get('tol_error', 1e-2)
        self.num_freqs = kwargs.get('num_freqs', 5)
        self.error_comp = kwargs.get('error_comp', True)
        self.error_force = kwargs.get('error_force', False)

        if self.num_freqs == 1:
            self.exp_freqs  = self.freq_central
        else:
            steps = self.freq_step * np.arange(-self.num_freqs+1,1)
            self.exp_freqs  = self.freq_central + steps

            for i, freq in enumerate(self.exp_freqs):
                index = np.abs(self.freq - freq).argmin()
                self.exp_freqs[i] = self.freq[index]

        self.dict_basis = {}
        self.exp_dimension = {}
        self.solution = {}
        self.dict_K_inv = {}
        self.dict_H = {}
        self.dict_U = {}
        self.proj_error = {}
        self.last_count = 0
    
    @property
    def count_freqs(self):
        return len(self.freq)
    
    @property
    def freq_step(self):
        return (max(self.freq) - min(self.freq)) / (2*(self.num_freqs))

    @property
    def new_moments(self):
        return ceil(0.4 * self.moment_matching)
    
    @property
    def index_central(self):
        return np.where(self.freq == self.freq_central)[0][0]

    @property
    def number_dof(self):
        return self.K.shape[0]

    @property
    def M_array(self):
        return self.M.toarray()
    
    @property
    def C_array(self):
        return self.C.toarray()

    @property
    def freq_central(self):
        return median_low(self.freq)
    
    def force(self, f):
        if self.freq_dependence:
            return self.force_by_freq[f]
        else:
            return self.b

    def get_dimensions(self):
        try:
            return [0 if self.exp_dimension.get(f) is None else self.exp_dimension.get(f) for f in self.exp_freqs]
        except:
            return 0 if self.exp_dimension.get(self.exp_freqs) is None else self.exp_dimension.get(self.exp_freqs)

    def svd_on_basis(self, basis):
        # Basis reduction via Singular Value Decomposition
        u, s, vh = svd(basis)
        # Take the singular values that are relevant
        index = s > self.tol_svd

        index_u = np.zeros(u.shape[1], dtype = 'bool')
        index_u[:len(index)] = index

        U, Sigma, Vh = u[:, index_u], np.diag( s[index] ), vh[index,:]
        W  = U @ Sigma @ Vh
        return W

    def next_frequency(self, f = None):
        if f is None:
            return self.freq_central
        else:
            index = np.where(self.freq == f)[0][0]
            if index == 0:
                return self.freq[self.index_central + 1]
            elif f > self.freq_central:
                if f == max(self.freq):
                    return None
                else:
                    return self.freq[index + 1]
            elif f <= self.freq_central:
                return self.freq[index - 1]
    
    def strategy(self, f):

        if self.num_freqs == 1:
            self.exp_freqs = f
            # Free memory
            old_exp_freq = list(self.dict_K_inv.keys()).remove(self.freq_central)
            if old_exp_freq is None:
                old_exp_freq = []
            for f in old_exp_freq:
                del self.dict_K_inv[f]
                del self.dict_basis[f]
                del self.dict_H[f]
                del self.dict_U[f]
            
        else:
            central = self.freq_central
            step = self.freq_step

            if f <= central:
                self.exp_freqs = self.exp_freqs[self.exp_freqs < f]
                if np.any(self.exp_freqs):
                    self.exp_freqs = np.r_[self.exp_freqs,f]
                else:
                    self.exp_freqs = np.array([f])
                while len(self.exp_freqs) < self.num_freqs:
                    min_f = min(self.exp_freqs)
                    new_f = min_f - step
                    index = np.abs(self.freq - new_f).argmin()
                    new_f = self.freq[index]
                    if new_f < min(self.freq):
                        new_f = min(self.freq)
                    self.exp_freqs = np.r_[self.exp_freqs,new_f]
                
                # Free Memory
                old_exp_freqs = np.fromiter(self.dict_K_inv.keys(), dtype=float)
                mask1 = old_exp_freqs != self.freq_central
                mask2 = old_exp_freqs > f
                for old_freq in old_exp_freqs[mask1 & mask2]:
                    del self.dict_K_inv[old_freq]
                    del self.dict_basis[old_freq]
                    del self.dict_H[old_freq]
                    del self.dict_U[old_freq]
                    
            else:
                self.exp_freqs = self.exp_freqs[self.exp_freqs > f]
                index = self.index_central
                if f == self.freq[index + 1]:
                    self.exp_freqs = np.r_[self.exp_freqs,self.freq_central]
                elif np.any(self.exp_freqs):
                    self.exp_freqs = np.r_[self.exp_freqs,f]
                else:
                    self.exp_freqs = np.array([f])
                while len(self.exp_freqs) < self.num_freqs:
                    max_f = max(self.exp_freqs)
                    new_f = max_f + step
                    index = np.abs(self.freq - new_f).argmin()
                    new_f = self.freq[index]
                    if new_f > max(self.freq):
                        new_f = max(self.freq)
                    self.exp_freqs = np.r_[self.exp_freqs,new_f]
                
                # Free Memory
                old_exp_freqs = np.fromiter(self.dict_K_inv.keys(), dtype=float)
                mask = old_exp_freqs < f
                for old_freq in old_exp_freqs[mask]:
                    del self.dict_K_inv[old_freq]
                    del self.dict_basis[old_freq]
                    del self.dict_H[old_freq]
                    del self.dict_U[old_freq]
            
            dif = self.num_freqs - len(np.unique(self.exp_freqs))
            if dif != 0:
                new_fs = np.random.uniform( min(self.exp_freqs), max(self.exp_freqs), dif)
                for new_f in new_fs:
                    index = np.abs(self.freq - new_f).argmin()
                    new_f = self.freq[index]
                    self.exp_freqs = np.unique(np.r_[self.exp_freqs,new_f])

    def expansion_basis(self):
        dimensions = self.get_dimensions()
        try:
            min_dimension = min(dimensions)
        except:
            min_dimension = dimensions

        if isinstance(self.exp_freqs, np.ndarray ):
            for f in self.exp_freqs:
                if f in self.dict_basis and self.exp_dimension.get(f) <= min_dimension + self.new_moments:
                    # Increase basis dimension if it's necessary
                    omega = 2 * pi * f
                    K_dynamic = - omega**2 * self.M + 1j * omega * self.C + self.K
                    C_dynamic = 2 * 1j * omega * self.M + self.C
                    K_inv = self.dict_K_inv[f]
                    A = lambda v: K_inv.solve( C_dynamic @ v )
                    B = lambda v: K_inv.solve( self.M @ v )
                    
                    V, H, U, u = self.dict_basis[f], self.dict_H[f], self.dict_U[f], self.solution[f]
                    V, H, U, _, _ = Soar.procedure(A, B, u, n = self.new_moments, V = V, H = H, U = U)
                    self.dict_basis.update({f : V})
                    self.exp_dimension.update({f : V.shape[1]})
                    self.dict_H.update({f : H})
                    self.dict_U.update({f : U})
                else:
                    omega = 2 * pi * f
                    K_dynamic = - omega**2 * self.M + 1j * omega * self.C + self.K
                    C_dynamic = 2 * 1j * omega * self.M + self.C
                    K_inv = splu(-K_dynamic.tocsc()) 
                    self.dict_K_inv.update({f : K_inv})

                    A = lambda v: K_inv.solve( C_dynamic @ v )
                    B = lambda v: K_inv.solve( self.M @ v )
                    u = - K_inv.solve( self.force(f) )
                    V, H, U, _, _ = Soar.procedure(A, B, u, n = self.moment_matching)

                    self.dict_basis.update({f : V})
                    self.exp_dimension.update({f : V.shape[1]})
                    self.dict_H.update({f : H})
                    self.dict_U.update({f : U})
                    self.solution.update({f : u})
                    self.proj_error.update({f : 0})
        
        elif isinstance(self.exp_freqs, np.generic ):
            f = self.exp_freqs
            if f in self.dict_basis and self.exp_dimension.get(f) <= min_dimension + self.new_moments:
                # Increase basis dimension if it's necessary
                omega = 2 * pi * f
                K_dynamic = - omega**2 * self.M + 1j * omega * self.C + self.K
                C_dynamic = 2 * 1j * omega * self.M + self.C
                K_inv = self.dict_K_inv[f]
                A = lambda v: K_inv.solve( C_dynamic @ v )
                B = lambda v: K_inv.solve( self.M @ v )
                
                V, H, U, u = self.dict_basis[f], self.dict_H[f], self.dict_U[f], self.solution[f]
                V, H, U, _, _ = Soar.procedure(A, B, u, n = self.new_moments, V = V, H = H, U = U)
                self.dict_basis.update({f : V})
                self.exp_dimension.update({f : V.shape[1]})
                self.dict_H.update({f : H})
                self.dict_U.update({f : U})
            else:
                omega = 2 * pi * f
                K_dynamic = - omega**2 * self.M + 1j * omega * self.C + self.K
                C_dynamic = 2 * 1j * omega * self.M + self.C
                K_inv = splu(-K_dynamic) 
                self.dict_K_inv.update({f : K_inv})

                A = lambda v: K_inv.solve( C_dynamic @ v )
                B = lambda v: K_inv.solve( self.M @ v )
                u = - K_inv.solve( self.force(f) )
                V, H, U, _, _ = Soar.procedure(A, B, u, n = self.moment_matching)

                self.dict_basis.update({f : V})
                self.exp_dimension.update({f : V.shape[1]})
                self.dict_H.update({f : H})
                self.dict_U.update({f : U})
                self.solution.update({f : u})
                self.proj_error.update({f : 0})
        else:
            raise ValueError("Invalid expansion frequency values.")


    def basis(self):
        self.expansion_basis()
        dimensions = self.get_dimensions()
        try:
            dim_high, dim_low = min(dimensions), min(dimensions) - self.new_moments
        except:
            dim_high, dim_low = dimensions, dimensions - self.new_moments
        basis_high, basis_low = None, None

        if isinstance(self.exp_freqs, np.ndarray ):
            for f in self.exp_freqs:
                V = self.dict_basis[f]
                if basis_high is None or basis_low is None:
                    basis_high = V[:, :dim_high]
                    basis_low = V[:, :dim_low]
                else:
                    basis_high = np.c_[basis_high, V[:, :dim_high]]
                    basis_low = np.c_[basis_low, V[:, :dim_low]]
            if self.svd:
                W_high = self.svd_on_basis(basis_high)
                W_low = self.svd_on_basis(basis_low)
            else:
                W_high = get_orthonormal(basis_high)
                W_low = get_orthonormal(basis_low)
            return W_high, W_low
        elif isinstance(self.exp_freqs, np.generic ):
            f = self.exp_freqs
            V = self.dict_basis[f]
            if basis_high is None or basis_low is None:
                basis_high = V[:, :dim_high]
                basis_low = V[:, :dim_low]
            else:
                basis_high = np.c_[basis_high, V[:, :dim_high]]
                basis_low = np.c_[basis_low, V[:, :dim_low]]
            return basis_high, basis_low
        else:
            raise ValueError("Invalid expansion frequency values.")

    
    def projection_step(self, freq, K_rom, C_rom, M_rom, b_rom, W ):
        omega = 2 * pi * freq
        K_dynamic_rom = - omega**2 * M_rom + 1j*omega * C_rom + K_rom
        u_rom = solve(K_dynamic_rom, b_rom)
        u_proj = W @ u_rom
        return u_proj        
    
    def reduced_matrices(self, W, f):
        W_hermit = W.conj().T

        K_rom = W_hermit @ self.K @ W
        C_rom = W_hermit @ self.C @ W
        M_rom = W_hermit @ self.M @ W
        b_rom = W_hermit @ self.force(f)

        return K_rom, C_rom, M_rom, b_rom
    
    def stagnation(self):
        if not self.solution:
            # No solution calculated
            return False
        count = self.count_freqs - len(self.solution.keys())
        factor = (count - self.last_count) / self.count_freqs
        self.last_count = count
        if factor < 0.1:
            return True
        else:
            return False

    def projection(self):
        # Init parameters
        start = time()
        error = None
        f = self.next_frequency()
        while f is not None:
            if error is None or error > self.tol_error:
                # Init -> error is None
                # Bad approximation -> error greater than tolerance
                if self.stagnation():
                    self.strategy(f)
                else:
                    pass
                W_high, W_low = self.basis()
                K_high, C_high, M_high, b_high = self.reduced_matrices(W_high, f)
                K_low, C_low, M_low, b_low = self.reduced_matrices(W_low, f)
            
            if f in self.dict_basis:
                error = 0
                f = self.next_frequency(f)
            else:
                u_high = self.projection_step(f, K_high, C_high, M_high, b_high, W_high)
                u_low = self.projection_step(f, K_low, C_low, M_low, b_low, W_low)
                if self.error_comp:
                    error = self.calculate_error_complementary(u_high, u_low)
                else:
                    error = self.calculate_error_force(f, u_high)
                
                if error < self.tol_error:
                    self.proj_error.update({f : error})
                    self.solution.update({f : u_high})
                    f = self.next_frequency(f)
        end = time()
        self.t_rom = end - start
        

    def get_solution(self):
        sol = [value for (key, value) in sorted(self.solution.items())]
        return np.array(sol).T

    def get_error(self):
        error = np.array([value for (_, value) in sorted(self.proj_error.items())])
        
        freq_error = np.delete(self.freq, np.where(error == 0))
        error = np.delete(error, np.where(error == 0))
        return freq_error, error

    def get_expension_frequencies(self):
        return np.fromiter(self.exp_dimension.keys(), dtype=float)

    def calculate_error_force(self, freq, u_proj):
        omega = 2 * pi * freq
        b_mass = - omega**2 * self.M @ u_proj
        b_damp = 1j * omega * self.C @ u_proj
        b_elas = self.K @ u_proj
        b_proj = b_mass + b_damp + b_elas
        error = norm(self.force(freq) - b_proj) / norm(self.force(freq))
        return error
    
    def plot(self, dof):
        solution_rom = self.get_solution()
        freq_error, error = self.get_error()
        expansion_frequencies = self.get_expension_frequencies()

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["legend.loc"] = 'lower left'
        plt.rc('xtick',labelsize=16)
        plt.rc('ytick',labelsize=16)
        fig = plt.figure(figsize=[10,8], tight_layout=True)
        gs = GridSpec(7, 1, figure=fig)

        ax0 = plt.subplot(gs[:3, :])

        plt.semilogy(self.freq, np.abs(solution_rom[dof, :]), color = [0,0,0], linewidth=1.5)
        plt.grid(True)
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax0.set_xlim([0,max(self.freq)+1])
        ax0.set_ylabel(("Admittance [m/N]"), fontsize = 16, fontweight = 'bold')
        ax0.legend(['ROM: ' + sec_to_hour(self.t_rom)], fontsize = 16)

        ax1 = plt.subplot(gs[3:5, :])
        plt.plot(self.freq, np.angle(solution_rom[dof, :]), color = [0,0,0], linewidth=2)
        plt.grid(True)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_ylim([-np.pi, np.pi])
        ax1.set_xlim([0,max(self.freq)+1])
        plt.yticks([-np.pi, 0, np.pi], ['-$\pi$', '0', '$\pi$'])
        ax1.set_ylabel(("Phase [rad]"), fontsize = 16, fontweight = 'bold')

        ax2 = plt.subplot(gs[5:8,0], sharex=ax0)
        plt.semilogy(freq_error, error, color = [0,0,0], linewidth=2)
        plt.axhline(self.tol_error, color = [0.3,0.3,0.3], linewidth=1, linestyle = (0, (5, 1, 1, 1)))
        plt.semilogy(expansion_frequencies, np.ones_like(expansion_frequencies)*self.tol_error, linewidth = 0, marker=(5, 1), markersize=8, color = [0.1,0.1,0.1])
        plt.grid(True)
        ax2.set_xlabel(("Frequency [Hz]"), fontsize = 16, fontweight = 'bold')
        ax2.set_ylabel(("Relative error [-]"), fontsize = 16, fontweight = 'bold')
        ax2.set_xlim([0,max(self.freq)+1])
        ax2.legend(['Estimated', 'Tolerance', 'Expansion points [{}]'.format(len(expansion_frequencies))], fontsize = 16)

        plt.show()

    
    @staticmethod
    def calculate_error_complementary(u_high, u_low):
        return norm(u_high - u_low) / norm(u_high)
