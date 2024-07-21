import numpy as np

def extract_first_eigenvector(mx):
    
    eigvals, eigvecs = np.linalg.eig(mx)
    return eigvecs[:,0]