import numpy as np
from test_configuration_bigger_matrix import *

sts = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]  # 16
ratios = [[1000,1],[100,1],[10,1],[1,1],[1,10],[1,100],[1,1000],[1,1000],[1,10000],[1,100000],[1,1000000],[1,10000000]]    #7
sizes = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]  #10

st = 16
n_max = 50
weights = [1,1]

outs = []
# for n_max in sizes:
# for st in sts:
for weights in ratios:
    out = run_model(weights, st, n_max)
    if out is not None:
        outs.append(out)
np.save(f"data/run_analysis/result_weights.npy", np.array(outs))
