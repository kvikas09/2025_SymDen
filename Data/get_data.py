import numpy as np
from xy_ring import *
import pickle, pprint
import matplotlib.pyplot as plt
import math
import time

J = 1.0
K = 0.0
# max_t = 0.05
# min_t = 2.05
lattice_shape = (8,8) #It can be  changed to (16,16) or (32,32)
steps = 5
iters_per_step = 64
random_state =1000
t_vals = np.linspace(min_t, max_t, 32)
print(t_vals)

# betas = 1 / T_vals
lattices = []
accept = 0
reject = 0
# #Monte Carlo Simulation
# start = time.time()
# for beta in t_vals:
#         lat=[]
#         print(beta)
#         random_state=random_state+1
#         xy=XYModelMetropolisSimulation(lattice_shape=lattice_shape,
#                                        beta=1/beta,J=J,K=K,random_state=random_state)
#         for q in range(12000):
#             xy.simulate(steps,iters_per_step)
#             lat.append(xy.L+0)
#             accept += xy.accept
#             reject += xy.reject
#             # draw_grid(lattice_shape[0],xy.L,1/beta)
#         lattices.append(lat[2000:])
#         print('Done')

# stop = time.time()        
# #Saving Data


output = open('8x8_lattices_32_temps_test.pkl', 'wb')
pickle.dump(lattices, output)
output.close()
print("accept:",accept)
print("reject:",reject)
print("accept ratio :",accept/(accept+reject))
print("time :",stop-start)