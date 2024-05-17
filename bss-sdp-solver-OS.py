# Import packages.
import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt

# Parameters for the ordered search SDP
N = 55                    # Instance size
q = 3                     # Number of queries
fuzzy_end = True          # Whether the final gram matrix is I or N/(N-1) I - 1/(N-1) J
epsilon = 1e-8            # Solver precision (default is not high enough)
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

Q = [[] for _ in range(q + 1)]
Q[0] = np.ones((N, N)) / N
Q[q] = np.eye(N) / N
if fuzzy_end:
    Q[q] = (N/(N-1) * np.eye(N) - 1/(N-1) * np.ones((N, N))) / N

# Define and solve the CVXPY problem.
constraints = []
for i in range(1, q):
    Q[i] = cp.Variable((N,N), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints += [Q[i] >> 0, cp.trace(Q[i]) == 1]

for t in range(1, q + 1):
    constraints += [
        tr_off_diag(Q[t], i) + (-1)**t * tr_off_diag(Q[t], i-N) == tr_off_diag(Q[t-1], i) + (-1)**t * tr_off_diag(Q[t-1], i-N) for i in range(1, N)
    ]

prob = cp.Problem(cp.Minimize(cp.trace(Q[1])),
                  constraints)
prob.solve(eps=epsilon)

# Print whether a solution was found
if prob.value < 2:
    print("SDP was feasible")
else:
    print("SDP was infeasible")
    exit()

# Compute the solution polynomials
P = [[] for _ in range(q + 1)]
for i in range(1, q):
    Q[i] = Q[i].value
for i in range(q + 1):
    P[i] = [tr_off_diag(Q[i], j) for j in range(-N+1, N)]

# Plot the solution polynomials
thetas = np.linspace(0, 2 * np.pi, N * 6 + 100)
for i in range(q + 1):
    plt.plot(thetas, poly_val(P[i], np.exp(1j * thetas)), label='P_' + str(i))
plt.legend()
fig_name = "SDP_polynomial_values_N=" + str(N) + "_q=" + str(q) + ("_fuzzy_end"  if fuzzy_end else "") + ".png"
if plot_polys:
    plt.show()
plt.savefig("Plots/" + ("fuzzy_end/"  if fuzzy_end else "") + fig_name)
print("Saving polynomial values plot in Plots/" + ("fuzzy_end/"  if fuzzy_end else "") + fig_name)
plt.clf()

# Plot the solution polynomial coefficients
degs = np.arange(-N+1, N)
for i in range(q+1):
    plt.plot(degs, P[i], label='P_' + str(i))
plt.legend()
fig_name = "SDP_polynomial_coeffs_N=" + str(N) + "_q=" + str(q) + ("_fuzzy_end"  if fuzzy_end else "") + ".png"
if plot_poly_coeffs:
    plt.show()
plt.savefig("Plots/"+ ("fuzzy_end/"  if fuzzy_end else "") + fig_name)
print("Saving polynomial coefficients plot in Plots/" + ("fuzzy_end/"  if fuzzy_end else "") + fig_name)