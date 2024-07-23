import numpy as np
import cvxpy as cp

def generate_theta_mxs(n):
    Thetas = [np.identity(n)] 
    for i in range(0, n-1):
        Thetas += [np.triu(np.roll(Thetas[i], +1, axis=1))]
    return Thetas



def generate_gamma_mxs(n, Thetas = None):
    if Thetas == None:
        Thetas = generate_theta_mxs(n)
    Gammas = np.transpose(Thetas)
    return np.vstack((Gammas, np.flip(Gammas)[1:]))



def generate_phi_mxs(n, Thetas = None, Gammas = None):
    if Thetas == None or Gammas == None:
        Thetas = generate_theta_mxs(n)
        Gammas = generate_gamma_mxs(n, Thetas)

    Phi_zero = 1/2*(Gammas[0] + np.eye(n))
    Phis = [Phi_zero]
    Phis += [1/4*(Gammas[k] + Thetas[k] + np.flip(Thetas[k])) for k in range(1, n)]
    Phis = np.vstack((Phis, 1/4*Gammas[n:]))

    return (Phis, Thetas, Gammas)



def generate_lambda_mxs(n, Thetas = None, Gammas = None):
    if Thetas == None or Gammas == None:
        Thetas = generate_theta_mxs(n)
        Gammas = generate_gamma_mxs(n, Thetas)

    Lambda_nil = 1/2*np.eye(n)
    Lambda_one = 1/4*(Thetas[1] + np.flip(Thetas[1]))

    Lambdas = [Lambda_nil, Lambda_one]
    Lambdas += [1/4*(-Gammas[k-2] + Thetas[k] + np.flip(Thetas[k])) for k in range(2, n)]
    Lambdas = np.vstack((Lambdas, -1/4*Gammas[(n-2):]))
    
    return (Lambdas, Thetas, Gammas)


class GramPairRep:
    def __init__(self, n, Q = None, S = None):
        if n % 2 == 1: 
            raise ValueError("This class requires the degree to be even.")
        self._n = n
        m = n // 2
        if Q is None: 
            self._Q = cp.Variable((m+1, m+1), symmetric=True)
        else: 
            self._Q = Q
        
        if S is None:
            self._S = cp.Variable((m, m), symmetric=True)
        else:
            self._S = S

        (self._Phis, _, _) = generate_phi_mxs(m+1)
        (self._Lambdas, _, _) = generate_lambda_mxs(m)



    def __sub__(self, other):
        return GramPairRep(self._n, self._Q - other.get_Q(), self._S - other.get_S())


    def get_Q(self, as_var = True):
        if as_var:
            return self._Q 
        else:
            return self._Q.value
    


    def get_S(self, as_var = True):
        if as_var:
            return self._S 
        else:
            return self._S.value



    def get_coord(self, k, as_var = True):
        if as_var:
            return cp.trace(self._Phis[k] @ self._Q) + cp.trace(self._Lambdas[k] @ self._S)
        else:
            return np.trace(self._Phis[k] @ self._Q.value) + np.trace(self._Lambdas[k] @ self._S.value)
        


    def get_coordinate_vector(self, as_var = True):
        return [self.get_coord(k, as_var) for k in range(0, self._n)] # could be optimized when as_var is false

    

    def get_semidefinite_constraints(self): 
        return [self._Q >> 0, self._S >> 0]



