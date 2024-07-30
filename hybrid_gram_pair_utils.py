import numpy as np
import cvxpy as cp
import scipy.sparse as sparse

class MatrixProvider:
    def __init__(self, n):
        self._n = n 
        self._m = n // 2

        self._set_B1()
        self._set_B2()
        self._set_W()

    def _B1_entry(self, k, l):
        return np.cos(np.pi * k * l / self._n)
    
    def _B2_entry(self, k, l):
        return np.sin(np.pi * (k+1) * l / self._n)
    
    def _set_W(self):
        A = np.fromfunction(self._B1_entry, (self._n+1, self._n+1))
        d = np.ones(self._n+1)
        d[0] = 1/2
        d[self._n] = 1/2
        D = sparse.diags(d)

        e = np.ones(self._n+1)
        e[0] = 2
        E = sparse.diags(e)

        self._W = 1 / self._n * E @ D @ A @ D
        self._W_tr = self._W

    def _set_B1(self):
        self._B1 = np.fromfunction(self._B1_entry, (self._m+1, self._n+1))
        self._B1_tr = np.transpose(self._B1)

    def _set_B2(self):
        self._B2 = np.fromfunction(self._B2_entry, (self._m, self._n+1))
        self._B2_tr = np.transpose(self._B2)

    def get_B1(self):
        return (self._B1, self._B1_tr)
    
    def get_B2(self):
        return (self._B2, self._B2_tr)
    
    def get_W(self):
        return (self._W, self._W_tr)



class GramPairRep:
    def __init__(self, n: int, matrix_provider: MatrixProvider, Q = None, S = None):
        self._n = n
        self._matrix_provider = matrix_provider

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

    def _set_B1(self):
        self._B1 = np.fromfunction(self._B1_entry, (self._m+1, self._n+1))
        self._B1_tr = np.transpose(self._B1)

    def _set_B2(self):
        self._B2 = np.fromfunction(self._B2_entry, (self._m, self._n+1))
        self._B2_tr = np.transpose(self._B2)

    def __sub__(self, other):
        if self._n != other.get_n(): 
            raise ValueError("Polynomials must be of the same degree.")
        
        return GramPairRep(self._n, self._matrix_provider, self._Q - other.get_Q(), self._S - other.get_S())



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
        coord_vec = self.get_coordinate_vector(as_var)
        return coord_vec[k]



    def get_coordinate_vector(self, as_var = True, skip_dct = False):
        (B1, B1_tr) = self._matrix_provider.get_B1()
        (B2, B2_tr) = self._matrix_provider.get_B2()
        (W, W_tr) = self._matrix_provider.get_W()
            
        if as_var:
            if skip_dct:
                return cp.diag(B1_tr @ self._Q @ B1) + cp.diag(B2_tr @ self._S @ B2)
            else:
                return W_tr @ (cp.diag(B1_tr @ self._Q @ B1) + cp.diag(B2_tr @ self._S @ B2))
        else:
            if skip_dct:
                return cp.diag(B1_tr @ self._Q.value @ B1) + cp.diag(B2_tr @ self._S.value @ B2)
            else:
                return W_tr @ (np.diag(B1_tr @ self._Q.value @ B1) + np.diag(B2_tr @ self._S.value @ B2))



    def get_semidefinite_constraints(self): 
        return [self._Q >> 0, self._S >> 0]
        