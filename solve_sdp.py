# Import packages.
import argparse
import timeit
import math 
import os.path
from os import makedirs

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt

import basis_utils  
import trig_utils


parser = argparse.ArgumentParser()
parser.add_argument("query_count", help="The number of queries.", type=int)
parser.add_argument("instance_size", help="The size of the OSP instance.", type=int)
parser.add_argument("--solver", help="Choose which solver to use.", type=str)
parser.add_argument("--repeats", help="Number of times the solver should be run for profiling purposes. Defaults to 1.", type=int)
parser.add_argument("--accuracy", help="Enter an integer m for 1e-m accuracy.", type=int)
parser.add_argument("--skip-save", help="Add flag to skip saving solutions to disk.", action='store_true')
parser.add_argument("--generate-plots", help="Add flag to generate plots.", action='store_true')
args = parser.parse_args()

if args.instance_size % 2 != 0:
    raise ValueError("This program requires the instance size to be even.")

solver = None
if args.solver == "CVXOPT":
    solver = cp.CVXOPT
if args.solver == "SCS":
    solver = cp.SCS
if solver == None: 
    solver = cp.MOSEK

rep_count = 1
if args.repeats != None:
    rep_count = args.repeats


epsilon = 1e-6            # Solver precision (default is not high enough)
if args.accuracy != None:
    epsilon = 1/10**args.accuracy


# Parameters for the ordered search SDP
# THIS CODE ASSUMES EVEN N!!!
N = args.instance_size    # Instance size (MUST BE EVEN!!!)
q = args.query_count      # Number of queries
EXPORTS_DIR = "exports/"   # relative path to directory for exports
COEFFS_EXPORT_SUBDIR = "coefficients/" 
PLOTS_EXPORT_SUBDIR = "plots/" 
MATRIX_PLOTS_SUBDIR = "matrix_plots/"
MX_EXPORT_SUBDIR = "matrices/"
# Note that plots will always be saved, flags determined if they are shown

print("Invoked with size params " + str(q) + " " + str(N) + ".")
print("Using solver " + str(solver) + ".")
print("Running solve " + str(rep_count) + " time(s).")

############################################################
################ Subroutine Definitions ####################
############################################################

Rs = [np.identity(N)] 
for i in range(0, N-1):
    Rs += [np.tril(np.roll(Rs[i], -1, axis=1))]

def cp_tr(M, i):
    return cp.trace(Rs[i] @ M)

def my_tr(M, i):
    return np.trace(Rs[i] @ M)

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
U = np.block([[I, I], [J, -J]])/math.sqrt(2)
A[0] = E / N
A[q] = I / N
B = [[] for _ in range(q + 1)]
B[0] = E / N
B[q] = 0 * I

# Define and solve the CVXPY problem.
constraints = []
for i in range(1, q):
    A[i] = cp.Variable((N // 2, N // 2), symmetric=True)
    B[i] = cp.Variable((N // 2, N // 2))

    constraints += [2 * cp.trace(A[i]) == 1]
    constraints += [J @ B[i].T @ J == B[i]]

    constraints += [A[i] + B[i] @ J >> 0]
    constraints += [A[i] - B[i] @ J >> 0]


for t in range(1, q + 1):
    Q_curr = cp.bmat([[A[t], B[t]], [J @ B[t] @ J, J @ A[t] @ J]])
    Q_prev = cp.bmat([[A[t - 1], B[t - 1]], [J @ B[t - 1] @ J, J @ A[t - 1] @ J]])

    constraints += [
        cp_tr(Q_curr, i) + (-1)**t * cp_tr(Q_curr, N-i) == cp_tr(Q_prev, i) + (-1)**t * cp_tr(Q_prev, N-i) for i in range(1, N)
        # cp.trace((Rs[i] + (-1)**t * Rs[N-i]) @ (Q_curr - Q_prev)) ==  0  for i in range(1, N) # slightly more efficient? 
    ] 

# Q1 = cp.bmat([[A[1], B[1]], [J @ B[1] @ J, J @ A[1] @ J]])
# constraints += [
#     cp_tr(Q1, 20) >= -0.262291*0.3
# ]

print("Number of constraints is " + str(len(constraints)))

print("Solving SDP")
prob = cp.Problem(cp.Minimize(0),
                  constraints)
def solve():
    prob._cache.param_prog = None
    prob.solve(eps=epsilon, solver=solver, verbose=False)

elapsed = timeit.timeit("solve()", globals=globals(), number=rep_count)
print("Finished. Time elapsed: " + str(round(elapsed, 2)))


BENCHMARK_FILE_HEADER = ["rep_count", "q", "N", "t_elapsed"]
benchmark_line = list(map(str, [rep_count, q, N, round(elapsed, 2)]))

benchmark_filepath = EXPORTS_DIR + "benchmarks/default.csv"
if not os.path.isfile(benchmark_filepath): 
    makedirs(EXPORTS_DIR + "benchmarks/", exist_ok=True)
    with open(benchmark_filepath, "w") as benchmark_file:
        benchmark_file.write(" ".join(BENCHMARK_FILE_HEADER) + "\n")
        benchmark_file.close()

with open(benchmark_filepath, "a") as benchmark_file:
    info = [rep_count, q, N, round(elapsed, 2)]
    benchmark_file.write(" ".join(benchmark_line) + "\n")
    benchmark_file.close()

if prob.value < 2:
    print("SDP was feasible")
else:
    print("SDP was infeasible")
    exit()


if args.generate_plots or not args.skip_save:
    Q = [[] for _ in range(q + 1)]
    Q[0] = np.ones((N, N)) / N
    Q[q] = np.eye(N, N) / N
    # Q[i] = np.block([[A[i], B[i]], [J @ B[i] @ J, J @ A[i] @ J]])

    # Compute the solution polynomials
    P = np.empty([q+1, 2*(N-1)+1])
    for i in range(1, q):
        # Q[i] = Q[i].value
        Q[i] = np.block([[A[i].value, B[i].value], [J @ B[i].value @ J, J @ A[i].value @ J]])
    for i in range(q + 1):
        P[i, (N-1):] = [my_tr(Q[i], j) for j in range(0, N)]
        P[i, :(N-1)] = np.flip(P[i, N:])

    basis_ch_mx_chebyshev_to_custom = np.linalg.inv(basis_utils.generate_basis_ch_mx_custom_to_chebyshev(N-1))
    basis_ch_mx_chebyshev_to_kernels = np.linalg.inv(basis_utils.generate_basis_ch_mx_kernels_to_chebyshev(N))
    basis_ch_mx_chebyshev_to_hermite = basis_utils.generate_basis_ch_mx_chebyshev_to_hermite(N)

    custom_coords = np.matmul(P[:, N:], np.transpose(basis_ch_mx_chebyshev_to_custom))
    kernel_coords = np.matmul(P[:, (N-1):], np.transpose(basis_ch_mx_chebyshev_to_kernels))
    hermite_coords = np.matmul(P[:, (N-1):], np.transpose(basis_ch_mx_chebyshev_to_hermite))

makedirs(EXPORTS_DIR + MX_EXPORT_SUBDIR, exist_ok=True)
for i in range(q+1):
    plt.matshow(U*Q[i]*U)
    plt.savefig(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + MATRIX_PLOTS_SUBDIR + "q" + str(q) + "_N" + str(N) + "_Q" + str(i) + ".png")
    plt.clf()

for i in range(q+1):
    np.savetxt(EXPORTS_DIR + MX_EXPORT_SUBDIR + "q" + str(q) + "_N" + str(N) + "_Q" + str(i) + ".txt", Q[i], fmt="%+1.3f")

for i in range(q+1):
    mx = trig_utils.laurent_poly_to_toeplitz_mx(P[i])
    np.savetxt(EXPORTS_DIR + MX_EXPORT_SUBDIR + "q" + str(q) + "_N" + str(N) + "_Toeplitz_Q" + str(i) + ".txt", mx, fmt="%+1.3f")

if not args.skip_save:
    txt_file_name = "polynomial_coeffs_" + str(q) + "_" + str(N) + ".txt"
    txt_file_name_custom_basis = "polynomial_coeffs_" + str(q) + "_" + str(N) + "_custom_basis.txt"
    txt_file_name_kernels_basis = "polynomial_coeffs_" + str(q) + "_" + str(N) + "_kernels_basis.txt"
    txt_file_name_hermite_basis = "polynomial_coeffs_" + str(q) + "_" + str(N) + "_hermite_basis.txt"

    makedirs(EXPORTS_DIR + COEFFS_EXPORT_SUBDIR, exist_ok=True)
    np.savetxt(EXPORTS_DIR + COEFFS_EXPORT_SUBDIR + txt_file_name, P[:, (N-1):], fmt="%+1.6f")
    np.savetxt(EXPORTS_DIR + COEFFS_EXPORT_SUBDIR + txt_file_name_custom_basis, custom_coords, fmt="%+1.4f,")
    np.savetxt(EXPORTS_DIR + COEFFS_EXPORT_SUBDIR + txt_file_name_kernels_basis, np.transpose(kernel_coords), fmt="%+1.6f,")
    np.savetxt(EXPORTS_DIR + COEFFS_EXPORT_SUBDIR + txt_file_name_hermite_basis, hermite_coords, fmt="%+1.6f,")

if args.generate_plots:
    makedirs(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR, exist_ok=True)
    # Plot the solution polynomials
    thetas = np.linspace(0, 2 * np.pi/20, N * 6 + 100)
    for i in range(1, q + 1):
        plt.plot(thetas, poly_val(P[i]-P[i-1], np.exp(1j * thetas)), label='D_' + str(i) + ' = P_' + str(i) + " - P_" + str(i-1))
    plt.legend()
    fig_name = "SDP_polynomial_values_" + str(q) + "_" + str(N) + ".png"
    # if plot_polys: TODO 
    #     plt.show()
    plt.savefig(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)
    print("Saving polynomial values plot in " + EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)
    plt.clf()

    # Plot the solution polynomial coefficients
    degs = np.arange(0, N)
    for i in range(0,q+1):
        plt.plot(degs, P[i,(N-1):], label='P_' + str(i))
    plt.legend()
    fig_name = "SDP_polynomial_coeffs_" + str(q) + "_" + str(N) + ".png"
    # if plot_poly_coeffs: TODO 
    #     plt.show()
    plt.savefig(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)
    plt.clf()

    # Plot the solution polynomial coefficients in custom basis 
    degs = np.arange(1, N)
    for i in range(q+1):
        plt.scatter(degs, custom_coords[i], label='P_' + str(i))    
    plt.legend()
    fig_name = "SDP_polynomial_coeffs_" + str(q) + "_" + str(N) + "_custom_basis.png"
    plt.savefig(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)
    plt.clf()

    # Plot the solution polynomial coefficients in Hermite basis 
    degs = np.arange(0, N)
    for i in range(q+1):
        plt.scatter(degs, kernel_coords[i], label='P_' + str(i))    
    plt.legend()
    fig_name = "SDP_polynomial_coeffs_" + str(q) + "_" + str(N) + "_kernels_basis.png"
    plt.savefig(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)
    plt.clf()

    # Plot the solution polynomial coefficients in Hermite basis 
    degs = np.arange(0, N)
    for i in range(q+1):
        plt.plot(degs, hermite_coords[i], label='P_' + str(i))    
    plt.legend()
    fig_name = "SDP_polynomial_coeffs_" + str(q) + "_" + str(N) + "_hermite_basis.png"
    plt.savefig(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)
    
    print("Saving polynomial coefficients plot in " + EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)