import numpy as np
import math 



def generate_chebyshev_coordinates(N):
    coords = np.zeros((N,N))

    coords[0,0] = 1
    coords[:,1] = np.roll(coords[:,0], 1)
    
    for n in range(2, N):
        coords[:,n] = 2*np.roll(coords[:,n-1], 1) - coords[:,n-2]
    
    return coords



def generate_hermite_coordinates(N):
    coords = np.zeros((N,N))

    coords[0,0] = 1
    coords[:,1] = 2*np.roll(coords[:,0], 1)
    
    for n in range(2, N):
        coords[:,n] = 2*np.roll(coords[:,n-1], 1) - 2*(n-1)*coords[:,n-2]
    
    return coords



def generate_basis_ch_mx_chebyshev_to_hermite(N):
    basis_ch_mx_chebyshev_to_standard = generate_chebyshev_coordinates(N)
    basis_ch_mx_hermite_to_standard = generate_hermite_coordinates(N)

    return np.matmul(np.linalg.inv(basis_ch_mx_hermite_to_standard), basis_ch_mx_chebyshev_to_standard)



def generate_basis_ch_mx_chebyshev_to_custom(N):
    coords = np.zeros((N,N))

    for i in range(math.ceil(N/2)): 
        v = np.zeros(N);
        v[i] = 1.0
        v[N-1-i] = 1.0 
        coords[:,i] = v
    for i in range(math.floor(N/2)): 
        v = np.zeros(N);
        v[i] = -1.0
        v[N-1-i] = 1.0 
        coords[:, math.ceil(N/2) + i] = v
    return np.linalg.inv(coords)