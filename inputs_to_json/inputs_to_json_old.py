# region Libraries
import os
import numpy as np
import scipy as cp
import pandas as pd
import pickle
from datetime import datetime
import math
import sys
import docplex.mp
from docplex.mp.model import Model

# endregion

# region json
# %% Json Module
import json
from json import JSONEncoder


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


# endregion

# region Inputs
# %% Detect the current working directory and print it ###

path = os.getcwd()
print("The current working directory is %s" % path)

# %% Cplex settings
url = None
key = None
gap_stop = 1e-4  # 0.01 for 1% and 0.1 for 10% Our default = 1e-4
run_time = 3 * 60 * 60

# %% Parameters
i = 325  # no of demand points
j = 5  # no of risk clusters
t = 60
# pop = 6861925
# pop_from_n_ij = 6909840
days = 60

# %% rand_60
# rand_60 = np.random.dirichlet(np.ones(60),size=1)[0]
# with open('rand_60', 'wb') as f:
# pickle.dump([rand_60], f)

with open("rand_60.json", "r") as read_file:
    rand_60 = np.asarray(json.load(read_file)["rand_60"])

sortet_rand_60 = np.sort(rand_60)
rand_60 = sortet_rand_60[::-1]

# %% n_ij
with open("n_ij.json", "r") as read_file:
    n_ij = np.asarray(json.load(read_file)["n_ij"])

n_ij_total = n_ij * 60
n_ij = np.zeros((i, j, t))

for ct in range(t):
    for ci in range(i):
        for cj in range(j):
            n_ij[ci, cj, ct] = np.ceil(n_ij_total[ci, cj] * rand_60[ct])

# %% Parameters
pop = np.sum(n_ij) * days  # based on the sum of n_ij
M = 100000
# epsilon = 6e-8 #is calculated based on values of k in the caller file.

k = 12
alpha = int(
    (i / 3) * 5 * 60
)  # 1 cars per 5 hospitals working 8 hours (5 hrs actual) pay day (in minutes as t_ii is in min)
select = 10  # Select 5 closest demand points
s = 9597
# s=int(pop/(days*k))
# c_i = c_i_base * ci_multiplier

# %% Read Weights of Priority Groups w_ij
with open("w_j.json", "r") as read_file:
    w_j = np.asarray(json.load(read_file)["w_j"])

sorted_w_j = -1 * np.sort(-1 * w_j) / 100
gamma = min(
    tuple(np.abs([sorted_w_j[c2 - 1] - c1 for c2, c1 in enumerate(sorted_w_j)][1:]))
)

# %% C_i from pickels
# #### Generating random c_i proportionate to the population of the assigned area
# c_i = [0] * i
# for c in range(i):
#     c_rand = np.random.uniform(
#         np.sum(n_ij, axis=1)[c] * 0.8, np.sum(n_ij, axis=1)[c] * 1.2, size=1
#     ).astype(int)[0]
#     if c_rand < pop / (days * i):
#         c_i[c] = int(pop / (days * i))
#     else:
#         c_i[c] = c_rand

with open("c_i.json", "r") as read_file:
    c_i_base = np.asarray(json.load(read_file)["c_i"])
c_i = np.copy(c_i_base)


# %% Load t_ii
with open("t_ii.json", "r") as read_file:
    t_ii = np.asarray(json.load(read_file)["t_ii"])

# To ensure for the distance of js hospitals to i is unique.
for c1 in range(i):
    for c2 in range(i):
        for c3 in range(i):
            if c2 != c3:
                if t_ii[c1, c2] == t_ii[c1, c3]:
                    t_ii[c1, c3] = t_ii[c1, c3] + 0.00001
                    t_ii[c3, c1] = t_ii[c1, c3]


# %% Theta
theta = np.max(t_ii)

# %% Creating nu_ii for C10
nu_ii = np.zeros(shape=(i, i))  # Create matrix nu_ij

n_i = np.sum(n_ij[:, :, 0], axis=1)  # Total demand of each hospital sumation over j

for c in range(i):
    if n_i[c] != 0:
        # print(c)
        hos_with_demand_index = np.where(n_i > 0)[0]
        hos_with_demand_index = np.delete(
            hos_with_demand_index, np.where(hos_with_demand_index == c)
        )
        t_i_all_demand = t_ii[c, hos_with_demand_index]

        short_dist_index = np.argsort(t_i_all_demand)[:select]
        short_dist = t_i_all_demand[
            short_dist_index
        ]  # values of distance from the 5 hospitals from the main hospital c

        hos_js = [
            np.where(t_ii[c] == c1)[0][0] for c1 in t_i_all_demand[short_dist_index]
        ]
        for c2 in hos_js:
            nu_ii[c, c2] = 1
            nu_ii[c2, c] = 1

# %% epsilon
epsilon = 0.8 * np.min(w_j) / ((k - 1) * theta)

# endregion


inputs_json = {
    "url": url,
    "key": key,
    "gap_stop": gap_stop,
    "run_time": run_time,
    "pop": 6861925,
    "i": i,
    "j": j,
    "t": t,
    "days": days,
    "rand_60": rand_60,
    "select": select,
    "M": M,
    "s": s,
    "k": k,
    "w_j": w_j,
    "c_i": c_i,
    "alpha": alpha,
    "t_ii": t_ii,
    "theta": theta,
    "epsilon": epsilon,
    "gamma": gamma,
    "n_ij": n_ij,
    "nu_ii": nu_ii,
    "n_ij_total": n_ij_total,
}

with open("inputs.json", "w") as write_file:
    json.dump(inputs_json, write_file, cls=NumpyArrayEncoder)
