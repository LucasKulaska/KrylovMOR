import numpy as np
from numpy.linalg import norm


def get_orthonormal(X, tol = 1e-10, Y = None):
    for i, x in enumerate(X.T):
        if Y is None:
            Y = x / norm(x)
        else:
            # Initiate projection loop
            norm_init = norm(x)
            coef = x.T @ Y.conj()
            if isinstance(coef,complex) or isinstance(coef,float):
                x -= coef * Y
            else:
                x -= Y @ coef
            
            # reorthogonalization if need.
            if norm(x) < 0.7*norm_init:
                coef = x.T @ Y.conj()
                if isinstance(coef,complex) or isinstance(coef,float):
                    x -= coef * Y
                else:
                    x -= Y @ coef
            
            # Orthogonal condition
            if norm(x) < tol * norm_init:
                pass
                # print('The', str(i)+'th', 'vector is orthogonal to the basis.')
            else:
                # New basis' element normalized.
                if Y is None:
                    Y = x / norm(x)
                else:
                    Y = np.c_[Y,  x / norm(x)]
    return Y
    
def new_element(Y,v):
    norm_init = norm(v)

    # first orthogonalization
    for y in Y.T:
        coef = y.T @ v
        v -= coef * y
    
    # reorthogonalization if need.
    if norm(v) < 0.7*norm_init:
        for y in Y.T:
            coef = y.T @ v
            v -= coef * y
    
    if norm(v) < 1e-12:
        print('The given vector is orthogonal to the given basis.')
        return Y
    return np.c_[Y,v]

if __name__ == "__main__":
    N = 100
    n = 30
    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    u = np.random.rand(N)

    # Krylov space
    V = np.zeros([N,n])
    V[:,0] = u
    V[:,1] = A @ u

    for i in range(2,n):
        V[:,i] = A @ V[:,i-1] + B @ V[:,i-2]

    Q = get_orthonormal(V)

    print(norm ( Q.T @ Q - np.eye(N) ))
