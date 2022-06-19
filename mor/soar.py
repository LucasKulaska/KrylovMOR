import numpy as np
from numpy.linalg import solve, norm
from numpy import conj, dot
from copy import deepcopy

class Soar:
    tolerance = 1e-12
    def __init__(self, A, B, u):
        pass

    @staticmethod
    def procedure(A, B, u, n, V = None, H = None, U=None):
        '''
        Q = SOAR(A, B, q1, n)
        computes an orthonormal basis Q of the second-order Krylov subspace:
            span{ Q }   = G_n(A,B,v_0).
                        = {v_0, v_1, v_2, ..., v_{n-1}}
            with    v_0 = q1 / norm(q1)
                    v_1 = A @ v_0
                    v_j = A @ v_{j-1} + B @ v_{j-2}
        using space-efficient SOAR procedure.

        Parameters:
        A, B :   N-dimensional square matrices
        q1   :   starting vector, of size N-by-1
        n    :   dimension of second-order Krylov subspace.

        Returns:
        Q   :   N-by-n matrix whose the vector columns form an orthonormal basis
                of the second-order Krylov subspace G_n(A,B,v_0).
        P   :   N-by-n matrix.
        T   :   n-by-n upper Hessenberg matrix.

        The following compact Krylov decomposition holds:
            A @ Q + B @ P = Q @ T + t_{n+1,n} * q_{n+1} @ e_n.T,
                        Q = P @ T + t_{n+1,n} * P_{n+1} @ e_n.T

        where, e_n is the nth column of n-dimensional identity matrix.

        This python code is adapted from [2].

        References:
        [1] Yangfeng Su, and Zhaojun Bai, SOAR: a second-order Arnoldi 
            method for the solution of the quadratic eigenvalue problem,
            SIAM J. Matrix Anal. Appl., 2005, 26(3): 640-659.
        
        [2] Ding Lu, Fudan Univ. TOAR: A Two-level Orthogonal ARnoldi procedure.
            http://www.unige.ch/~dlu/toar.html

        Author:
        Lucas Kulakauskas, UFSC Brazil, 2020/03/28. '''
        # Initialize
        N = len(u)
        q = u / norm(u)

        # Allocate memory
        if V is None and H is None and U is None:
            # Non initialized  process
            j_0 = 0
            V = np.zeros([N,n], dtype = 'complex')
            V[:,0] = q
            H = np.zeros([n,n-1], dtype = 'complex')
            g = np.zeros(N)
            U = g
        else:
            # Re-init process
            j_0 = V.shape[1] -1
            e_j = np.zeros(j_0)
            e_j[j_0-1] = 1
            v_aux = solve( H[1:j_0+1,0:j_0], e_j )
            g = V[:,:j_0] @ v_aux
            V = np.c_[V, np.zeros([N,n-1], dtype = 'complex')]
            H = np.c_[H, np.zeros([H.shape[0], n-1], dtype = 'complex')]
            H = np.r_[H, np.zeros([n-1, H.shape[1]], dtype = 'complex')]

        P = None
        deflation = [] # Deflation index list.

        for j in j_0 + np.arange(n-1):
            # Recurrence role
            r = A(V[:,j]) + B(g)
            norm_init = norm(r)
            basis = V[:,:j+1]

            ## Modified Gram Schmidt procedure
            # first orthogonalization
            coef = r.T @ basis.conj()
            if isinstance(coef,complex) or isinstance(coef,float):
                r -= coef * basis
            else:
                r -= basis @ coef
            # Saving coefficients
            H[:j+1, j] = coef
            
            # Re-orthogonalization, if needed.
            if norm(r) < 0.7 * norm_init:
                coef = r.T @ basis.conj()
                if isinstance(coef,complex) or isinstance(coef,float):
                    r -= coef * basis
                else:
                    r -= basis @ coef
                H[:j+1, j] += coef
            r_norm = norm(r)

            # Check for breakdown
            if r_norm > Soar.tolerance:
                H[j+1,j] = r_norm
                V[:,j+1] = r / r_norm
                e_j = np.zeros(j+1)
                e_j[j] = 1
                v_aux = solve( H[1:j+2,0:j+1], e_j )
                g = V[:,:j+1] @ v_aux
            else:
                # Deflation reset
                H[j+1,j] = 1
                V[:,j+1] = np.zeros(N)
                e_j = np.zeros(j+1)
                e_j[j] = 1
                v_aux = solve( H[1:j+2,0:j+1], e_j )
                g = V[:,:j+1] @ v_aux

                # Deflation verification
                print("Deflation")
                f_proj = deepcopy(g)
                if P is None:
                    P = f_proj
                else:
                    for p in P:
                        coef_f = dot( g, conj(p) ) / dot( p, conj(p) ) 
                        f_proj = f_proj - p * coef_f
                    if norm(f_proj) > Soar.tolerance:
                        deflation.append(j)
                    else:
                        print('SOAR lucky breakdown.')
                        return V, H, P, deflation
                    P = np.c_[P, g]
            U = np.c_[U, g]
        
        n = n + j_0
        e_n = np.zeros([n-1,1])
        e_n[n-2] = 1
        r = V[:, -1].reshape((-1,1)) * H[-1,-1]
        g = U[:, -1].reshape((-1,1)) * H[-1,-1]

        aux1 = norm(V.conj().T @ V - np.eye(n))
        aux2 = norm( A(V[:,:n-1]) + B(U[:,:n-1]) - V[:,:n-1] @ H[:n-1,:n-1] - r @ e_n.T )
        aux3 = norm( V[:,:n-1] - U[:,:n-1] @ H[:n-1,:n-1] - g @ e_n.T )
        if aux1 > 1e-10 or aux2 > 1e-10 or aux3 > 1e-10:
            pass
            # print("SOAR instability.")
        
        return V, H, U, P, deflation

if __name__ == "__main__":
    # Example and test setup
    N = 100
    n = 30
    M1 = np.random.rand(N,N)
    M2 = np.random.rand(N,N)
    u = np.random.rand(N)

    A = lambda v: M1 @ v
    B = lambda v: M2 @ v

    V, H, U, P, deflation = Soar.procedure(A, B, u, n = n)

    e_n = np.zeros([n-1,1])
    e_n[n-2] = 1
    r = V[:, -1].reshape((-1,1)) * H[-1,-1]
    g = U[:, -1].reshape((-1,1)) * H[-1,-1]

    print("The following values must be zero:")
    print( norm( V.conj().T @ V - np.eye(n) ) )
    print( norm( A(V[:,:n-1]) + B(U[:,:n-1]) - V[:,:n-1] @ H[:n-1,:n-1] - r @ e_n.T ) )
    print( norm( V[:,:n-1] - U[:,:n-1] @ H[:n-1,:n-1] - g @ e_n.T ) )
