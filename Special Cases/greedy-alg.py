# A playground for different angles used by the 1-round greedy algorithm
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("N", help="The instance size.", type=int)
parser.add_argument("k", help="The number of queries.", type=int)
args = parser.parse_args()
N = args.N
k = args.k

### Playground for the one-query algorithm
# Phases are considered relative to the greedy algorithm, e.g. all 0 represents greedy
def phase(n):
    return 0

# Entropy of a list of probabilities
def entropy(p_list):
    entropy = 0
    for p in p_list:
        entropy -= p * np.log2(p)
    return entropy

# Probability of outcome x using the algorithm with phases given by phase(m)
def prob(x, N):
    return (np.abs(np.sum([np.exp(2j * np.pi * (2 * n + 1) * x / (2 * N) + phase(n) * 1j) / np.sin(np.pi * (2 * n + 1) / (2 * N)) for n in range(N)])) / (N ** 1.5)) ** 2

# prob_list = [prob(x, N) for x in range(N)]
# ent = entropy(prob_list)
# base_ent = np.log2(N)
# print("Entropy: ", ent)
# print("Information: ", base_ent - ent)
# plt.plot(prob_list)
# plt.show()
ratios = [prob(0, i) / prob(1, i) for i in range(10, N, 10)]
print(ratios)
plt.scatter([i for i in range(10, N, 10)], ratios)
# plt.yscale('log')
# plt.xscale('log')
plt.show()

# ALL OF THE BELOW IS BUGGY!
# ### Analysis of the many query algorithm
# # We will consider a list of coefficients in the momentum basis to represent the state, which can be FFT'ed to the position basis

# # The odd transition matrix, has rows given by the p's and columns given by q's
# def odd_t_mat(p, q):
#     if p % 2 == 1 and q % 2 == 0:
#         return 1 / N * ( 1 / np.tan(np.pi * (p - q) / (2 * N)) + 1j)
    
# # The even transition matrix, has rows given by the p's and columns given by q's
# def even_t_mat(p, q):
#     if p % 2 == 0 and q % 2 == 1:
#         return 1 / N * ( 1 / np.tan(np.pi * (p - q) / (2 * N)) + 1j)
    

# def generic_f(shape, elementwise_f):
#     fv = np.vectorize(elementwise_f)
#     return np.fromfunction(fv, shape)

# Me = generic_f((2 * N, 2 * N), even_t_mat)
# Me[Me == None] = 0
# Mo = generic_f((2 * N, 2 * N), odd_t_mat)
# Mo[Mo == None] = 0

# # print(N)
# # print(Me)
# # print(Mo)

# # For a system in state s, compute the next state of the greedy algorithm from making the q-th query. Depends on parity of q.
# def next_state(s, q):
#     if q % 2 == 0:
#         return np.matmul(Me, s)
#     else:
#         return np.matmul(Mo, s)
    
# init_state = np.zeros(2 * N)
# init_state[0] = 1

# second_state = next_state(init_state, 1)
# # print(second_state)
# # print(np.linalg.norm(second_state, ord=2)) # Should be 1, is not 1 :(