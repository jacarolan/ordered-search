# Import packages.
import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt

# Parameters for the ordered search SDP
N = 6                     # Instance size
q = 2                     # Number of queries
epsilon = 1e-8            # Solver precision (default is not high enough)
plot_polys = False        # Plots analogous to Figures 1 and 2 in [arxiv:0608161]
save_plots = True         # Note that plots are saved only if they are not shown

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
    
    
# Flip the polynomial coefficients of a laurent polynomial
def flip_poly(P):
    d = len(P)
    mid_d = (d-1) // 2
    new_coeffs = [0] * d
    for i in range(mid_d):
        new_coeffs[i] = P[mid_d - i - 1]
        new_coeffs[mid_d + 1 + i] = P[d - i - 1]
    return np.array(new_coeffs)

# Returns a list of values of symmetric laurent polynomial, assumes length of P is odd
def poly_val(P, xs):
    d = len(P)
    mid_d = (d-1) // 2
    vals = []
    for x in xs:
        vals += [sum([P[i] * x**(i-mid_d) for i in range(d)])]
    return np.array(vals)

# Determines whether two polynomials are equal on the unit circle, up to precision eps
def poly_eq(P1, P2, eps=1e-6):
    d = len(P1)
    thetas = np.linspace(0, 2 * np.pi, 1000)
    vals1 = poly_val(P1, np.exp(1j * thetas))
    vals2 = poly_val(P2, np.exp(1j * thetas))
    return np.all(np.abs(vals1 - vals2) < eps)


############################################################
###################### SDP Solving #########################
############################################################

Q = [[] for _ in range(q + 1)]
Q[0] = np.ones((N, N)) / N
Q[q] = np.eye(N) / N

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

# Compute the solution polynomials
P = [[] for _ in range(q + 1)]
P_rev = [[] for _ in range(q + 1)]
P_sum = [[] for _ in range(q + 1)]
P_diff = [[] for _ in range(q + 1)]
for i in range(1, q):
    Q[i] = Q[i].value
for i in range(q + 1):
    P[i] = [tr_off_diag(Q[i], j) for j in range(-N+1, N)]
    P_rev[i] = flip_poly(P[i])
    P_sum[i] = P[i] + P_rev[i]
    P_diff[i] = P[i] - P_rev[i]

# Determine equality relations among the polynomials
# Note that not every equality relation is checked...
# One could check more things if desired
for i in range(q + 1):
    for j in range(i +1, q + 1):
        if poly_eq(P_sum[i], P_sum[j]):
            print("P_sum_" + str(i) + " and P_sum_" + str(j) + " are equal")
        if poly_eq(P_diff[i], P_diff[j]):
            print("P_diff_" + str(i) + " and P_diff_" + str(j) + " are equal")
        if poly_eq(P_diff[i], P_sum[j]):
            print("P_diff_" + str(i) + " and P_sum_" + str(j) + " are equal")
        if poly_eq(P_sum[i], P_diff[j]):
            print("P_sum_" + str(i) + " and P_diff_" + str(j) + " are equal")

# Plot the solution polynomials
thetas = np.linspace(0, 2 * np.pi, N * 6 + 100)
for i in range(q + 1):
    plt.plot(thetas, poly_val(P[i], np.exp(1j * thetas)), label='P_' + str(i))
plt.legend()
fig_name = "SDP_polynomial_values_N=" + str(N) + "_q=" + str(q) + ".png"
plt.title("SDP_polynomial_values_N=" + str(N) + "_q=" + str(q))
if plot_polys:
    plt.show()
elif save_plots:
    plt.savefig("Plots/" + fig_name)
print("Saving polynomial values plot in Plots/" + fig_name)
plt.clf()

# Plot the solution polynomial coefficients
degs = np.arange(-N+1, N)
for i in range(q+1):
    plt.plot(degs, P[i], label='P_' + str(i))
plt.legend()
fig_name = "SDP_polynomial_coeffs_N=" + str(N) + "_q=" + str(q) + ".png"
plt.title("SDP_polynomial_coeffs_N=" + str(N) + "_q=" + str(q))
if plot_polys:
    plt.show()
elif save_plots:
    plt.savefig("Plots/" + fig_name)
print("Saving polynomial coefficients plot in Plots/" + fig_name)
plt.clf()


# Plot the solution polynomial reversals
thetas = np.linspace(0, 2 * np.pi, N * 6 + 100)
for i in range(q + 1):
    plt.plot(thetas, poly_val(P_rev[i], np.exp(1j * thetas)), label='P_' + str(i))
plt.legend()
fig_name = "SDP_reversed_polynomial_values_N=" + str(N) + "_q=" + str(q) + ".png"
plt.title("SDP_reversed_polynomial_values_N=" + str(N) + "_q=" + str(q))
if plot_polys:
    plt.show()
elif save_plots:
    plt.savefig("Plots/" + fig_name)
print("Saving polynomial values plot in Plots/" + fig_name)
plt.clf()

# Plot the sum of solution polynomial and their reversals
thetas = np.linspace(0, 2 * np.pi, N * 6 + 100)
for i in range(q + 1):
    plt.plot(thetas, poly_val(P[i] + P_rev[i], np.exp(1j * thetas)), label='P_' + str(i))
plt.legend()
fig_name = "SDP_sum_polynomial_values_N=" + str(N) + "_q=" + str(q) + ".png"
plt.title("SDP_sum_polynomial_values_N=" + str(N) + "_q=" + str(q))
if plot_polys:
    plt.show()
elif save_plots:
    plt.savefig("Plots/" + fig_name)
print("Saving polynomial values plot in Plots/" + fig_name)
plt.clf()

# Plot the difference of solution polynomial and their reversals
thetas = np.linspace(0, 2 * np.pi, N * 6 + 100)
for i in range(q + 1):
    plt.plot(thetas, poly_val(P[i] - P_rev[i], np.exp(1j * thetas)), label='P_' + str(i))
plt.legend()
fig_name = "SDP_diff_polynomial_values_N=" + str(N) + "_q=" + str(q) + ".png"
plt.title("SDP_diff_polynomial_values_N=" + str(N) + "_q=" + str(q))
if plot_polys:
    plt.show()
elif save_plots:
    plt.savefig("Plots/" + fig_name)
print("Saving polynomial values plot in Plots/" + fig_name)
plt.clf()