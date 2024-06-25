# Import packages.
import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt

# Parameters for the ordered search SDP
# THIS CODE ASSUMES EVEN N!!!
N = 56                    # Instance size (MUST BE EVEN!!!)
q = 3                     # Number of queries
epsilon = 1e-6            # Solver precision (default is not high enough)
plot_polys = False        # Plots analogous to Figure 1 in [arxiv:0608161]
plot_poly_coeffs = False  # Plots analogous to Figure 2 in [arxiv:0608161]
# Note that plots will always be saved, flags determined if they are shown

############################################################
################ Subroutine Definitions ####################
############################################################

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

# Returns a list of values of symmetric laurent polynomial, assumes length of P is odd
def poly_val(P, xs):
    d = len(P)
    mid_d = (d-1) // 2
    vals = []
    for x in xs:
        vals += [sum([P[i] * x**(i-mid_d) for i in range(d)])]
    return np.array(vals)

############################################################
###################### SDP Solving #########################
############################################################
I = np.eye(N // 2)
J = np.fliplr(I)
E = np.ones((N // 2, N // 2))
A = [[] for _ in range(q + 1)]
A[0] = E / N
A[q] = I / N
B = [[] for _ in range(q + 1)]
B[0] = E / N
B[q] = 0 * I

# Define and solve the CVXPY problem.
constraints = []
for i in range(1, q):
    A[i] = cp.Variable((N // 2,N // 2), symmetric=True)
    constraints += [2 * cp.trace(A[i]) == 1]
    B[i] = cp.Variable((N // 2,N // 2))
    constraints += [J @ B[i].T @ J == B[i]]
    # The operator >> denotes matrix inequality.
    constraints += [A[i] + B[i] @ J >> 0]
    constraints += [A[i] - B[i] @ J >> 0]

for t in range(1, q + 1):
    Q_t = cp.bmat([[A[t], B[t]], [J @ B[t] @ J, J @ A[t] @ J]])
    Q_t_prev = cp.bmat([[A[t - 1], B[t - 1]], [J @ B[t - 1] @ J, J @ A[t - 1] @ J]])
    constraints += [
        tr_off_diag(Q_t, i) + (-1)**t * tr_off_diag(Q_t, i-N) == tr_off_diag(Q_t_prev, i) + (-1)**t * tr_off_diag(Q_t_prev, i-N) for i in range(1, N)
    ]

print("Solving SDP")

prob = cp.Problem(cp.Minimize(0),
                  constraints)
prob.solve(eps=epsilon, solver=cp.CVXOPT)

print("Finished solving!")

# Print whether a solution was found
if prob.value < 2:
    print("SDP was feasible")
else:
    print("SDP was infeasible")
    exit()


Q = [[] for _ in range(q + 1)]
Q[0] = np.ones((N, N)) / N
Q[q] = np.eye(N, N) / N
# Q[i] = np.block([[A[i], B[i]], [J @ B[i] @ J, J @ A[i] @ J]])

# Compute the solution polynomials
P = [[] for _ in range(q + 1)]
for i in range(1, q):
    # Q[i] = Q[i].value
    Q[i] = np.block([[A[i].value, B[i].value], [J @ B[i].value @ J, J @ A[i].value @ J]])
for i in range(q + 1):
    P[i] = [tr_off_diag(Q[i], j) for j in range(-N+1, N)]

# Plot the solution polynomials
thetas = np.linspace(0, 2 * np.pi, N * 6 + 100)
for i in range(q + 1):
    plt.plot(thetas, poly_val(P[i], np.exp(1j * thetas)), label='P_' + str(i))
plt.legend()
fig_name = "SDP_polynomial_values_N=" + str(N) + "_q=" + str(q) + ".png"
if plot_polys:
    plt.show()
plt.savefig("Plots/symmetrized/" + fig_name)
print("Saving polynomial values plot in Plots/symmetrized/" + fig_name)
plt.clf()

# Plot the solution polynomial coefficients
degs = np.arange(-N+1, N)
for i in range(q+1):
    plt.plot(degs, P[i], label='P_' + str(i))
plt.legend()
fig_name = "SDP_polynomial_coeffs_N=" + str(N) + "_q=" + str(q) + ".png"
if plot_poly_coeffs:
    plt.show()
plt.savefig("Plots/symmetrized/" + fig_name)
print("Saving polynomial coefficients plot in Plots/symmetrized/" + fig_name)