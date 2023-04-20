# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 18:36:39 2023

@author: rfsyl
"""
import create_occ_grid
import pathing
from matplotlib import pyplot as plt
from spatialmath import SE2
import numpy as np
import pickle
import networkx as nx
import math

#create_occ_grid.save_occ_grid("test_occ_2",100,100,[100,100],10,["circle"],[[0,100],[0,100]],[3,10])

#output = pathing.test_logm()

#This code plots the edges of the RRG graphs for display purposes
#This can take a while to run, especially if there are a lot of edges with multiple wheel commands
w_radius = 1
width = 2
N_samples = 300
d_max = 30
cost_max = 150
max_wheel = math.pi*5
opt_speed = 4
particle_num =1
sigma_2 = np.zeros(2)
step_num = 20
col_penalty = 100
col_percent_max = 0.3

occ_grid, obj_list, center, scale, map_size = create_occ_grid.load_occ_grid("test_occ_2")

map_size = [[0,99],[0,99]]

graph = pickle.load(open('test_graph_PRM_curve_planner.pickle', 'rb'))

sample_nodes = np.random.choice(graph.nodes,2,replace=False)
start = sample_nodes[0]
goal = sample_nodes[1]

#node_path = nx.astar_path(graph,start, goal, weight='weight')

#[path,weight,time,wh_speed] = pathing.RecoverPath(graph, node_path)

fig, ax = plt.subplots()
im = ax.imshow(occ_grid)
'''
for i in range(len(path)):
    t = np.linspace(0,time[i],step_num)
    x = np.zeros((len(t),1))
    y = np.zeros((len(t),1))
    current_pose = path[i]
    x[0] = current_pose.t[0]
    y[0] = current_pose.t[1]
    for k in range(step_num-1):
        current_pose = pathing.PF_prop(t[k],t[k+1],current_pose,wh_speed[i],w_radius,width,sigma_2)
        x[k+1] = current_pose[0,2]
        y[k+1] = current_pose[1,2]
        
    
    ax.plot(x,y,color='red')
'''
N = len(graph.nodes)
x_node = np.zeros((N,1))
y_node = np.zeros((N,1))
for node in range(len(graph.nodes)):

    
    pose = current_pose = graph.nodes[node]['pose']
    x_node[node] = pose.t[0]
    y_node[node] = pose.t[1]
edge_num = 0 
for edge in graph.edges:
    time = graph.edges[edge]['time']
    wh_speed = graph.edges[edge]['wh_speed']
    #wh_speed= [wh_speed]
    wh_speed = np.transpose(np.asarray(wh_speed).reshape((2,len(time))))
    current_pose = graph.nodes[edge[0]]['pose'].A
    wh_num = 0
    #time = [time] #Only for curve_planner, uncomment othrwise
    for ti in range(len(time)):
    #if True:
        t = np.linspace(0,time[ti],step_num)
        #t = np.linspace(0,time,step_num)
        #t = np.asarray(time)
        #x = np.zeros((len(t)+1,1))
        #y = np.zeros((len(t)+1,1))
        x = np.zeros((len(t),1))
        y = np.zeros((len(t),1))
        
        x[0] = current_pose[0,2]
        y[0] = current_pose[1,2]
        current_pose = np.asarray(current_pose)
        for g in range(step_num-1):
        #for g in range(2):
            #current_pose = pathing.PF_prop(0,t[g],current_pose,wh_speed[g],w_radius,width,sigma_2)
            #current_pose = pathing.PF_prop(t[g],t[g+1],current_pose,wh_speed[ti],w_radius,width,sigma_2)
            current_pose = pathing.PF_prop(t[g],t[g+1],current_pose,wh_speed[ti],w_radius,width,sigma_2)
            x[g+1] = current_pose[0,2]
            y[g+1] = current_pose[1,2]
            #x[k] = current_pose[0,2]
            #y[k] = current_pose[1,2]
             
        wh_num +=1 
        ax.plot(x,y,color='red')
        ax.scatter(x_node,y_node,color='blue',s=1)
    edge_num+=1
    if edge_num%20 == 0:
        print(f'{edge_num} edges plotted')
    
plt.show()