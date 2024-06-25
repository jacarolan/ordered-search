import numpy as np
import numpy.linalg as la

# Constructing the HNS adversary matrix
def adv_HNS(n):
    # Construct the HNS matrix
    HNS = np.zeros((2 * n, 2 * n))
    for i in range(2 * n):
        for j in range(2 * n):
            if i != j and abs(i - j) < n // 2:
                HNS[i, j] = 1 / (np.pi * abs(i - j))
    return HNS

# Construct the D matrix
# Assumes ordering like 1100, 0110, 0011, 1001
def D(n, i):
    D = np.zeros((2 * n, 2 * n))
    for j in range(2 * n):
        for k in range(2 * n):
            if (j + i) // n != (k+i) // n:
                D[j, k] = 1
    return D

for i in range(4):
    print(D(2, i))