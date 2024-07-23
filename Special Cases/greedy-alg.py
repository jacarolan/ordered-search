# A playground for different angles used by the 1-round greedy algorithm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("N", help="The instance size.", type=int)
args = parser.parse_args()
N = args.N

# Phases are considered relative to the greedy algorithm, e.g. all 0 represents greedy
def phase(n):
    return 0 #1j * np.pi * n / N

# Entropy of a list of probabilities
def entropy(p_list):
    entropy = 0
    for p in p_list:
        entropy -= p * np.log2(p)
    return entropy

# Probability of outcome x using the algorithm with phases given by phase(m)
def prob(x):
    return (np.abs(np.sum([np.exp(2j * np.pi * (2 * n + 1) * x / (2 * N) + phase(n)) / np.sin(np.pi * (2 * n + 1) / (2 * N)) for n in range(N)])) / (N ** 1.5)) ** 2

prob_list = [prob(x) for x in range(N)]
ent = entropy(prob_list)
base_ent = np.log2(N)
print("Entropy: ", ent)
print("Information: ", base_ent - ent)
