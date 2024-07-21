import numpy as np
from scipy.linalg import toeplitz

def laurent_poly_to_toeplitz_mx(coeffs):
    n = len(coeffs)
    assert n % 2 == 1, "The coefficient vector must have odd length."
    deg = (n-1)//2; 
    degrees = np.arange(deg+1)
    first_row = [coeffs[deg-j] for j in degrees]
    first_column = [coeffs[deg+j] for j in degrees]
    return toeplitz(first_row, first_column)

def laurent_poly_to_circulant_mx(coeffs):
    n = len(coeffs)
    assert n % 2 == 1, "The coefficient vector must have odd length."
    deg = (n-1)//2; 
    degrees = np.arange(1,deg+1)
    first_column = [coeffs[deg]] + [coeffs[deg+j] + coeffs[deg-(deg+1-j)] for j in degrees]
    first_row = [coeffs[deg]] + [coeffs[deg-j] + coeffs[deg+deg+1-j] for j in degrees]
    return toeplitz(first_row, first_column)


def laurent_poly_to_circulant_mx_signed(coeffs):
    n = len(coeffs)
    assert n % 2 == 1, "The coefficient vector must have odd length."
    deg = (n-1)//2; 
    degrees = np.arange(1,deg+1)
    first_column = [coeffs[deg]] + [coeffs[deg+j] - coeffs[deg-(deg+1-j)] for j in degrees]
    first_row = [coeffs[deg]] + [coeffs[deg-j] - coeffs[deg+deg+1-j] for j in degrees]
    return toeplitz(first_row, first_column)


def symmetric_laurent_poly_to_circulant_mx(coeffs):
    n = coeffs.shape[0]
    degrees = np.arange(1,n)
    first_row = [coeffs[0]] + [coeffs[j] + coeffs[n-j] for j in degrees]
    return symmetric_toeplitz_matrix(first_row)


def symmetric_laurent_poly_to_circulant_mx_signed(coeffs):
    n = coeffs.shape[0]
    degrees = np.arange(1,n)
    first_row = [coeffs[0]] + [coeffs[j] - coeffs[n-j] for j in degrees]
    return symmetric_toeplitz_matrix(first_row)


def symmetric_toeplitz_matrix(first_row):
    n = len(first_row)
    matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            matrix[i][j] = first_row[j - i]
            matrix[j][i] = first_row[j - i]

    return matrix
