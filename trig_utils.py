import numpy as np
from scipy.linalg import toeplitz

def laurent_poly_to_toeplitz_mx(coeffs):
    n = len(coeffs)
    assert n % 2 == 1, "The coefficient vector must have odd length."
    deg = (n-1)//2; 
    degrees = np.arange(deg+1)
    first_row = [coeffs[deg+j] for j in degrees]
    first_column = [coeffs[deg-j] for j in degrees]
    return toeplitz(first_row, first_column)