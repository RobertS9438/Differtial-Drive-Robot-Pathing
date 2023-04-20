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
w_radius = 1
width = 2
N_samples = 200
d_max = 20
cost_max = 1000
max_wheel = math.pi
opt_speed = 4
particle_num =1
sigma_2 = np.zeros(2)
step_num = 20
col_penalty = 10000
col_percent_max = 0.3
path_func = pathing.est_curve_alt

map_size = np.asarray([[0,99],[0,99]])

occ_grid = create_occ_grid.mask_occ_grid(occ_grid,width,False)
graph = pathing.make_PRM(map_size, occ_grid, N_samples,path_func,d_max,cost_max,w_radius,width,max_wheel,opt_speed,particle_num,sigma_2,step_num,col_penalty,col_percent_max)
finish = clock-time.time()
#Tell what file to save graph to
pickle.dump(graph, open('test_graph_PRM_est_curve_alt_many_100_mask.pickle', 'wb'))
