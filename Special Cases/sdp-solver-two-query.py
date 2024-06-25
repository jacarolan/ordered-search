# Import packages.
import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt

# Trace of the off-diagonal elements of a matrix
# In particular, the i-th super-diagonal (or -i-th sub-diagonal) elements are summed
def tr_off_diag_cp(M, i):
    N = M.shape[0]
    if i > 0:
        return cp.sum([M[j, j+i] for j in range(N-i)])
    elif i < 0:
        return cp.sum([M[j-i, j] for j in range(N+i)])
    else:
        return cp.trace(M)
    
# Trace of the off-diagonal elements of a matrix
# In particular, the i-th super-diagonal (or -i-th sub-diagonal) elements are summed
def tr_off_diag(M, i):
    N = M.shape[0]
    if i > 0:
        return np.sum([M[j, j+i] for j in range(N-i)])
    elif i < 0:
        return np.sum([M[j-i, j] for j in range(N+i)])
    else:
        return np.trace(M)

# Trace of the off-diagonal elements of a matrix, signed by t
# In particular, returns a list of sums of the i-th super-diagonal (or -i-th sub-diagonal) elements, each multiplied by (-1)^t
def T(Q, t):
    N = Q.shape[0]
    return [tr_off_diag_cp(Q, i) + (-1)**t * tr_off_diag_cp(Q, i-N) for i in range(1, N-1)]


# Generate the simplest ordered search SDP
N = 6
q = 2

Q = [[], [], []]

Q[0] = np.ones((N, N)) / N
Q[2] = np.eye(N)/N

# Define and solve the CVXPY problem.
# Create a symmetric matrix variable.
Q[1] = cp.Variable((N,N), symmetric=True)
# The operator >> denotes matrix inequality.
constraints = [Q[1] >> 0]

# Add the constraints for the first query
for t in range(1, 3):
    constraints += [
        tr_off_diag(Q[t], i) + (-1)**t * tr_off_diag(Q[t], i-N) == tr_off_diag(Q[t-1], i) + (-1)**t * tr_off_diag(Q[t-1], i-N) for i in range(1, N-1)
    ]

constraints += [
    cp.trace(Q[1]) == 1
]

prob = cp.Problem(cp.Minimize(cp.trace(Q[1])),
                  constraints)
prob.solve()

# Print result.
print("The optimal value is", prob.value)
print("A solution is")
print(Q[1].value)

# Compute the solution polynomials

P = [[], [], []]

Q[1] = Q[1].value
# Q[2] = Q[2].value

for i in range(q + 1):
    P[i] = [tr_off_diag(Q[i], j) for j in range(-N+1, N)]

print("The solution polynomials are:\n" + str(P))

# Returns the value of symmetric laurent polynomial, assumes length of P is odd
def poly_val(P, x):
    d = len(P)
    mid_d = (d-1) // 2
    return sum([P[i] * x**(i-mid_d) for i in range(d)])

# Returns a list of values of symmetric laurent polynomial, assumes length of P is odd
def poly_val(P, xs):
    d = len(P)
    mid_d = (d-1) // 2
    vals = []
    for x in xs:
        vals += [sum([P[i] * x**(i-mid_d) for i in range(d)])]
    return np.array(vals)

thetas = np.linspace(0, 2 * np.pi, 100)
plt.plot(thetas, poly_val(P[0], np.exp(1j * thetas)), label='P_0')
plt.plot(thetas, poly_val(P[1], np.exp(1j * thetas)), label='P_1')
plt.plot(thetas, poly_val(P[2], np.exp(1j * thetas)), label='P_2')
plt.legend()
plt.show()