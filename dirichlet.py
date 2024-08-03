# Import packages.
import argparse
import timeit
import math 
import os.path
from os import makedirs
import resource
from memory_profiler import profile 

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("query_count", help="The number of queries.", type=int)
parser.add_argument("instance_size", help="The size of the OSP instance.", type=int)
parser.add_argument("--solver", help="Choose which solver to use.", type=str)
parser.add_argument("--repeats", help="Number of times the solver should be run for profiling purposes. Defaults to 1.", type=int)
parser.add_argument("--accuracy", help="Enter an integer m for 1e-m accuracy.", type=int)
parser.add_argument("--skip-save", help="Add flag to skip saving solutions to disk.", action='store_true')
parser.add_argument("--verbose", help="Add flag to run CVXPY in verbose mode.", action='store_true')
parser.add_argument("--presolve", help="Add flag to presolve with SCS.", action='store_true')
parser.add_argument("--generate-plots", help="Add flag to generate plots.", action='store_true')
args = parser.parse_args()

solver = None
if args.solver == "CVXOPT":
    solver = cp.CVXOPT
if args.solver == "SCS":
    solver = cp.SCS
if args.solver == "CLARABEL":
    solver = cp.CLARABEL
if solver == None: 
    args.solver = "MOSEK"
    solver = cp.MOSEK

rep_count = 1
if args.repeats != None:
    rep_count = args.repeats


epsilon = 1e-8            # Solver precision (default is not high enough)
if args.accuracy != None:
    epsilon = 1/10**args.accuracy


# Parameters for the ordered search SDP
# THIS CODE ASSUMES EVEN N!!!
N = args.instance_size    # Instance size (MUST BE EVEN!!!)
q = args.query_count      # Number of queries
EXPORTS_DIR = "exports/dirichlet/"   # relative path to directory for exports
COEFFS_EXPORT_SUBDIR = "coefficients/" 
PLOTS_EXPORT_SUBDIR = "plots/" 
MATRIX_PLOTS_SUBDIR = "matrix_plots/"
MX_EXPORT_SUBDIR = "matrices/"
# Note that plots will always be saved, flags determined if they are shown

print("Invoked with size params " + str(q) + " " + str(N) + ".")
print("Using solver " + str(solver) + ".")
print("Running solve " + str(rep_count) + " time(s).")

# Assumes N is odd!!!!


m = N // 2
tau = np.pi * 2/N

def eval(func, grid):
    return np.array([func(w) for w in grid])

def fejer(n, w):
    if w == 0:
        return n
    else: 
        return 1/n * (math.sin(n*w/2)/math.sin(w/2))**2

def dirichlet(n, w):
    if w == 0:
        return 1
    else:
        return 1/(2*n+1) * math.sin((2*n+1)/2*w)/math.sin(w/2)    

def phi_at(w):
    res = np.zeros(2 * m + 1)
    for j in range(-m, m+1):
        res[j+m] = dirichlet(m, w + j * tau)
    return res

def phi_grid(w):
    res = np.zeros(2 * m + 1)
    for j in range(-m, m+1):
        res[j+m] =w + j * tau
    return res

def eval_on_grid(Q, grid):
    vals = np.zeros(len(grid))
    for j in range(len(grid)):
        phi = phi_at(grid[j])
        vals[j] = phi.T @ Q @ phi
    return vals 

def quad(A, v):
    return v.T @ A @ v


grid = np.zeros(2*N-1)
for j in range(-N+1, N):
    grid[N-1+j] = j * tau / 2

phi_grids = eval(phi_grid, grid)



roots_of_one = np.array([tau * j for j in range(-m, m+1)]) # N elems 
roots_of_minus_one = np.array([tau/2 + tau * j for j in range(-m, m)]) # N-1 elems

phis_at_roots_of_one = eval(phi_at, roots_of_one)
phis_at_roots_of_minus_one = eval(phi_at, roots_of_minus_one)

fejer_at_roots_of_minus_one = eval(lambda w: fejer(N, w), roots_of_minus_one)
fejer_at_roots_of_one = eval(lambda w: fejer(N, w), roots_of_one)

def init_constraints(N):
    Qs = [cp.Variable((N, N), symmetric=True) for _ in range(q + 1)]
    constraints = [Q >= 0 for Q in Qs]

    constraints += [fejer_at_roots_of_one[j] == quad(Qs[0], phis_at_roots_of_one[j]) for j in range(len(roots_of_one))] # 0th polynomial 
    constraints += [fejer_at_roots_of_minus_one[j] == quad(Qs[0], phis_at_roots_of_minus_one[j]) for j in range(len(roots_of_minus_one))]
    
    constraints += [1 == quad(Qs[q], phis_at_roots_of_one[j]) for j in range(len(roots_of_one))] # qth polynomial 
    constraints += [1 == quad(Qs[q], phis_at_roots_of_minus_one[j]) for j in range(len(roots_of_minus_one))]

    for t in range(1,q+1):
        if t % 2 == 0:
            constraints += [quad(Qs[t], phis_at_roots_of_one[j]) == quad(Qs[t-1], phis_at_roots_of_one[j]) for j in range(len(roots_of_one))]
        else: 
            constraints += [quad(Qs[t], phis_at_roots_of_minus_one[j]) == quad(Qs[t-1], phis_at_roots_of_minus_one[j]) for j in range(len(roots_of_minus_one))]

    return (constraints, Qs)


def init_problem(constraints):
    return cp.Problem(cp.Maximize(1),
                  constraints)

(constraints, Qs) = init_constraints(N)
prob = init_problem(constraints)

def solve():
    prob.solve(solver=solver, verbose=bool(args.verbose))

elapsed = timeit.timeit("solve()", globals=globals(), number=rep_count)
print("Finished. Time elapsed: " + str(round(elapsed, 2)))
max_memory = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/10**9, 2)
print("Maximal memory usage was " + str(max_memory) + " gigabytes")

print("roots of one", roots_of_one)
print("roots of minus one", roots_of_minus_one)

if prob.status == 'infeasible':
    print(">>> SDP was INFEASIBLE <<<")
    exit()
    

if args.generate_plots:
    # Plot the solution polynomials
    thetas = np.linspace(-np.pi/10, np.pi/10, 2*N+1+100)
    for t in range(0, q + 1):
        Q = Qs[t].value
        plt.plot(thetas, eval_on_grid(Q, thetas), label='q_' + str(t))
    plt.legend()
    fig_name = "poly_values_" + str(q) + "_" + str(N) + "_zoomed.png"
    plt.savefig(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)
    plt.clf()

    thetas = np.linspace(-np.pi, np.pi, 2*N+1+100)
    for t in range(0, q + 1):
        Q = Qs[t].value

        evals = eval_on_grid(Q, thetas)
        min = np.min(evals)
        print("minimum of the", t, "th polynomial:", np.round(min, 3))
        
        plt.plot(thetas, evals, label='q_' + str(t))
    plt.legend()
    fig_name = "poly_values_" + str(q) + "_" + str(N) + ".png"
    plt.savefig(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)
    plt.clf()

#     degs = np.arange(0, N)
#     for i in range(1,q+1):
#         ratio = poly_coordinates[i] / poly_coordinates[i-1]
#         for j in range(len(ratio)):
#             if abs(ratio[j]) > 1:
#                 ratio[j] = 1 * np.sign(ratio[j])
#         plt.plot(degs, ratio, label='q_' + str(i) + ' / q_' + str(i-1))
#     plt.legend()
#     fig_name = "SDP_polynomial_coeffs_ratios_" + str(q) + "_" + str(N) + ".png"
#     plt.savefig(EXPORTS_DIR + PLOTS_EXPORT_SUBDIR + fig_name)
#     plt.clf()