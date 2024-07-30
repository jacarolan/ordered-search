# Import packages.
import argparse
import timeit
import math 
import os.path
from os import makedirs
import resource

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt

from hybrid_gram_pair_utils import GramPairRep, MatrixProvider


parser = argparse.ArgumentParser()
parser.add_argument("query_count", help="The number of queries.", type=int)
parser.add_argument("instance_size", help="The size of the OSP instance.", type=int)
parser.add_argument("--solver", help="Choose which solver to use.", type=str)
parser.add_argument("--repeats", help="Number of times the solver should be run for profiling purposes. Defaults to 1.", type=int)
parser.add_argument("--accuracy", help="Enter an integer m for 1e-m accuracy.", type=int)
parser.add_argument("--skip-save", help="Add flag to skip saving solutions to disk.", action='store_true')
parser.add_argument("--generate-plots", help="Add flag to generate plots.", action='store_true')
args = parser.parse_args()

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


epsilon = 1e-8            # Solver precision (default is not high enough)
if args.accuracy != None:
    epsilon = 1/10**args.accuracy


# Parameters for the ordered search SDP
N = args.instance_size    # Instance size (MUST BE EVEN!!!)
q = args.query_count      # Number of queries
EXPORTS_DIR = "exports/gram_pair/"   # relative path to directory for exports
COEFFS_EXPORT_SUBDIR = "coefficients/" 
PLOTS_EXPORT_SUBDIR = "plots/" 
MATRIX_PLOTS_SUBDIR = "matrix_plots/"
MX_EXPORT_SUBDIR = "matrices/"
# Note that plots will always be saved, flags determined if they are shown

print("Invoked with size params " + str(q) + " " + str(N) + ".")
print("Using solver " + str(solver) + ".")
print("Running solve " + str(rep_count) + " time(s).")

# Returns a list of values of symmetric laurent polynomial, assumes length of P is odd
def eval_on_grid(coords, thetas):
    vals = []
    for theta in thetas:
        vals += [coords[0] + sum([coords[j] * math.cos(theta*j) for j in range(1, len(coords))])]
    return np.array(vals)
    

fejer_kernel_coordinates = np.fromiter((1-j/N for j in range(N)), dtype=float)
identity_coordinates = np.zeros(N)
identity_coordinates[0] = 1

provider = MatrixProvider(N-1)

Polys = [GramPairRep(N-1, provider) for _ in range(q+1)]

constraints = []
for j in range(q+1):
    constraints += Polys[j].get_semidefinite_constraints()

constraints += [
    Polys[0].get_coordinate_vector() == fejer_kernel_coordinates,
    Polys[q].get_coordinate_vector() == identity_coordinates
]


for t in range(0, q):
    poly_curr = Polys[t].get_coordinate_vector(skip_dct=True)
    poly_next = Polys[t+1].get_coordinate_vector(skip_dct=True)
    diff = poly_next - poly_curr 

    constraints += [
        diff[0] == 0,
        diff[1:] == (-1)**t * diff[::-1][:-1]
    ] 


print("Solving SDP")
prob = cp.Problem(cp.Maximize(1),
                  constraints)
def solve():
    prob._cache.param_prog = None
    prob.solve(eps=epsilon, solver=solver, verbose=True)

elapsed = timeit.timeit("solve()", globals=globals(), number=rep_count)
print("Finished. Time elapsed: " + str(round(elapsed, 2)))
print("Maximal memory usage was " + str(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/10**9, 2)) + " gigabytes")

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

if prob.status == 'optimal':
    print(">>> SDP was FEASIBLE <<<")
else:
    print(">>> SDP was INFEASIBLE <<<")
    exit()


poly_coordinates = []
for t in range(q+1):
    poly_coordinates += [Polys[t].get_coordinate_vector(False)]
poly_coordinates = np.array(poly_coordinates)

makedirs(EXPORTS_DIR + COEFFS_EXPORT_SUBDIR, exist_ok=True)
np.savetxt(EXPORTS_DIR + COEFFS_EXPORT_SUBDIR + "coeffs_new.txt", np.transpose(poly_coordinates), fmt="%+1.6f")


if args.generate_plots:
    makedirs(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR, exist_ok=True)

    # Plot the solution polynomial coefficients
    degs = np.arange(1, N)
    for t in range(0, q+1):
        plt.plot(degs, poly_coordinates[t,1:], label='q_' + str(t))
    plt.legend()
    fig_name = "poly_coeffs_" + str(q) + "_" + str(N) + ".png"
    plt.savefig(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)
    plt.clf()

    # Plot the solution polynomials
    thetas = np.linspace(-np.pi, np.pi, 2*N+1+100)
    for t in range(0, q + 1):
        plt.plot(thetas, np.round(eval_on_grid(poly_coordinates[t], thetas), 4), label='q_' + str(t))
    plt.legend()
    fig_name = "poly_values_" + str(q) + "_" + str(N) + ".png"
    plt.savefig(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)
    plt.clf()

    degs = np.arange(1, N)
    for t in range(1,q+1):
        plt.plot(degs, poly_coordinates[t, 1:] / poly_coordinates[t-1, 1:], label='q_' + str(t) + ' / q_' + str(t-1))
    plt.legend()
    fig_name = "SDP_polynomial_coeffs_ratios_" + str(q) + "_" + str(N) + ".png"
    plt.savefig(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)
    plt.clf()