# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 19:20:46 2023

@author: rfsyl
"""
import create_occ_grid
import pathing
import networkx as nx
import math
import numpy as np
from matplotlib import pyplot as plt
import pickle
import time

clock = time.time()
occ_grid, obj_list, center, scale, map_size = create_occ_grid.load_occ_grid("test_occ_3")

#This code creates the RRG graphs
'''
fig, ax = plt.subplots()
im = ax.imshow(occ_grid)
plt.show()
'''
w_radius = 1            #Robot wheel radius (m)
width = 2               #Robot width (m)
N_samples = 200         #Number of nodes
d_max = 20              #Radius within which gragh will look for nodes with valid connections
cost_max = 1000         #Maximum cost for a valid edge
max_wheel = math.pi     #Maximum allowed wheel angular speed (rad/sec)
opt_speed = 4           #Speed robot would prefer to move at (m/s)
particle_num =1         #leave this at 1, it is a remnant from when the code evaluated edge cost using a particle filter
sigma_2 = np.zeros(2)   #leave this as is (see above)
step_num = 20           #Steps size used when checking for collisions with obstacles
col_penalty = 10000     #Cost penalty for collisions
col_percent_max = 0.3   #Remnant from particle filter cost calculation, leave as is
path_func = pathing.est_curve_alt   #function used for inter-node path connection method from pathing.py

map_size = np.asarray([[0,99],[0,99]])

occ_grid = create_occ_grid.mask_occ_grid(occ_grid,width,False)
graph = pathing.make_PRM(map_size, occ_grid, N_samples,path_func,d_max,cost_max,w_radius,width,max_wheel,opt_speed,particle_num,sigma_2,step_num,col_penalty,col_percent_max)
finish = clock-time.time()
#Tell what file to save graph to
pickle.dump(graph, open('test_graph_PRM_est_curve_alt_many_100_mask.pickle', 'wb'))
