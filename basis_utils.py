import numpy as np
np.set_printoptions(suppress=True)



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



def generate_basis_ch_mx_chebyshev_to_custom(n):
    mx = [];
    for i in range(n // 2 + 1): 
        v = np.zeros(n);
        v[i] = 1.0
        v[n-1-i] = 1.0 
        mx += [v]
    for i in range(n // 2): 
        v = np.zeros(n);
        v[i] = -1.0
        v[n-1-i] = 1.0 
        mx += [v]
    return np.linalg.inv(np.transpose(np.asmatrix(mx)))