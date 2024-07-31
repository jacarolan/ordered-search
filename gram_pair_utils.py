import numpy as np
import cvxpy as cp
import scipy.sparse as sparse
from memory_profiler import profile


@profile
class MaskMatrixFactory:
    def __init__(self, n):
        self._Thetas = {}
        self._Gammas = {}
        self._Phis_even_rules = {}
        self._Phis_odd_rules = {}
        self._Lambdas_even_rules = {}
        self._Lambdas_odd_rules = {}

        m = n // 2
        offset = (n % 2)
        self.get_phis(m + 1, offset)
        self.get_lambdas(m + offset, offset)

        # self._intify(m, offset)
        # self._sparsify(m, offset)

    def _intify(self, m, offset):
        if offset:
            self._Phis_odd_rules[m+1] = [mx.astype(int) for mx in self._Phis_odd_rules[m+1]]
        else: 
            self._Phis_even_rules[m+1] = [mx.astype(int) for mx in self._Phis_even_rules[m+1]]
        
        if offset:
            self._Lambdas_odd_rules[m + offset] = [mx.astype(int) for mx in self._Lambdas_odd_rules[m + offset]]
        else: 
            self._Lambdas_even_rules[m + offset] = [mx.astype(int) for mx in self._Lambdas_even_rules[m + offset]]


        
    def _sparsify(self, m, offset):
        if offset:
            self._Phis_odd_rules[m+1] = [sparse.csr_matrix(mx) for mx in self._Phis_odd_rules[m+1]]
        else: 
            self._Phis_even_rules[m+1] = [sparse.csr_matrix(mx) for mx in self._Phis_even_rules[m+1]]
        
        if offset:
            self._Lambdas_odd_rules[m + offset] = [sparse.csr_matrix(mx) for mx in self._Lambdas_odd_rules[m + offset]]
        else: 
            self._Lambdas_even_rules[m + offset] = [sparse.csr_matrix(mx) for mx in self._Lambdas_even_rules[m + offset]]



    def get_thetas(self, n): 
        if n not in self._Thetas:
            self._Thetas[n] = self._generate_theta_mxs(n)
        return self._Thetas[n]



    def get_gammas(self, n):
        if n not in self._Gammas:
            self._Gammas[n] = self._generate_gamma_mxs(n)
        return self._Gammas[n]
    


    def get_phis(self, n, use_odd_rules):
        if use_odd_rules:
            return self._get_phis_use_odd_rules(n)
        else:
            return self._get_phis_use_even_rules(n)



    def get_lambdas(self, n, use_odd_rules):
        if use_odd_rules:
            return self._get_lambdas_use_odd_rules(n)
        else:
            return self._get_lambdas_use_even_rules(n)



    def _get_phis_use_odd_rules(self, n):
        if n not in self._Phis_odd_rules: 
            self._Phis_odd_rules[n] = self._generate_phi_mxs_use_odd_rules(n)
        return self._Phis_odd_rules[n]

    

    def _get_phis_use_even_rules(self, n):
        if n not in self._Phis_even_rules: 
            self._Phis_even_rules[n] = self._generate_phi_mxs_use_even_rules(n)
        return self._Phis_even_rules[n]



    def _get_lambdas_use_odd_rules(self, n):
        if n not in self._Lambdas_odd_rules: 
            self._Lambdas_odd_rules[n] = self._generate_lambda_mxs_use_odd_rules(n)
        return self._Lambdas_odd_rules[n]



    def _get_lambdas_use_even_rules(self, n):
        if n not in self._Lambdas_even_rules: 
            self._Lambdas_even_rules[n] = self._generate_lambda_mxs_use_even_rules(n)
        return self._Lambdas_even_rules[n]



    def _generate_theta_mxs(self, n):
        Thetas = [np.identity(n)] 
        for i in range(0, n-1):
            Thetas += [np.triu(np.roll(Thetas[i], +1, axis=1))]
        return Thetas



    def _generate_gamma_mxs(self, n):
        Thetas = self.get_thetas(n)
        Gammas = np.transpose(Thetas)
        return np.vstack((Gammas, np.flip(Gammas)[1:]))



    def _generate_phi_mxs_use_odd_rules(self, n):
        Thetas = self.get_thetas(n)
        Gammas = self.get_gammas(n)

        Phi_nil = 1/4 * (Thetas[0] + np.flip(Thetas[0]))
        Phis = [Phi_nil]
        Phis += [1/4 * (Gammas[k-1] + Thetas[k] + np.flip(Thetas[k])) for k in range(1, n)]
        Phis = np.vstack((Phis, 1/4 * Gammas[(n-1):]))

        return Phis



    def _generate_phi_mxs_use_even_rules(self, n):
        Thetas = self.get_thetas(n)
        Gammas = self.get_gammas(n)

        Phi_nil = 1/2*(Gammas[0] + np.eye(n))
        Phis = [Phi_nil]
        Phis += [1/4*(Gammas[k] + Thetas[k] + np.flip(Thetas[k])) for k in range(1, n)]
        Phis = np.vstack((Phis, 1/4*Gammas[n:]))

        return Phis



    def _generate_lambda_mxs_use_odd_rules(self, n):
        Thetas = self.get_thetas(n)
        Gammas = self.get_gammas(n)

        Lambda_nil = 1/4 * (Thetas[0] + np.flip(Thetas[0]))

        Lambdas = [Lambda_nil]
        Lambdas += [1/4*(-Gammas[k-1] + Thetas[k] + np.flip(Thetas[k])) for k in range(1, n)]
        Lambdas = np.vstack((Lambdas, -1/4*Gammas[(n-1):]))
        
        return Lambdas



    def _generate_lambda_mxs_use_even_rules(self, n):
        Thetas = self.get_thetas(n)
        Gammas = self.get_gammas(n)

        Lambda_nil = 1/2*np.eye(n)
        Lambda_one = 1/4*(Thetas[1] + np.flip(Thetas[1]))

        Lambdas = [Lambda_nil, Lambda_one]
        Lambdas += [1/4*(-Gammas[k-2] + Thetas[k] + np.flip(Thetas[k])) for k in range(2, n)]
        Lambdas = np.vstack((Lambdas, -1/4*Gammas[(n-2):]))
        
        return Lambdas


class GramPairRep:
    def __init__(self, n: int, mask_factory: MaskMatrixFactory, Q = None, S = None):
        self._n = n
        self._mask_factory = mask_factory;
        
        self._m = n // 2
        self._offset = (n % 2)

        if Q is None: 
            self._Q = cp.Variable((self._m+1, self._m+1), symmetric=True)
        else: 
            self._Q = Q
        
        if S is None:
            self._S = cp.Variable((self._m + self._offset, self._m + self._offset), symmetric=True)
        else:
            self._S = S
      


    def __sub__(self, other):
        if self._n != other.get_n(): 
            raise ValueError("Polynomials must be of the same degree.")
        
        return GramPairRep(self._n, self._mask_factory, self._Q - other.get_Q(), self._S - other.get_S())



    def get_n(self):
        return self._n



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
        Phis = self._mask_factory.get_phis(self._m + 1, self._offset)
        Lambdas = self._mask_factory.get_lambdas(self._m + self._offset, self._offset)
        
        if as_var:
            return cp.trace(Phis[k] @ self._Q) + cp.trace(Lambdas[k] @ self._S)
        else:
            return np.trace(Phis[k] @ self._Q.value) + np.trace(Lambdas[k] @ self._S.value)
        


    def get_coordinate_vector(self, as_var = True):
        return [self.get_coord(k, as_var) for k in range(0, self._n+1)] # could be optimized when as_var is false

    

    def get_semidefinite_constraints(self): 
        return [self._Q >> 0, self._S >> 0]



