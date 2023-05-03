# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:52:48 2023

@author: rfsyl
"""
#This code runs the simulation tests

import math
import pathing
import create_occ_grid
import numpy as np
import random
from spatialmath import SE2
import simulation_setup
import networkx as nx
import pickle
import pybullet as p
import time
from matplotlib import pyplot as plt

w_radius = 1
width = 2
N_samples = 300
d_max = 20
cost_max = 400
max_wheel = math.pi
opt_speed = 4
particle_num =1
sigma_2 = 0
sigma_p = 0
step_num = 20
col_penalty = 100000
col_percent_max = 0.3
#Pathing function used for adding nodes to graph
#path_func = pathing.est_curve_alt
path_func_add = pathing.spin_and_move_planning
#path_func_add = pathing.est_curve_alt
#path_func_add = pathing.curve_planner
PosError_max = 10       #Max acceptable position error for course correcting algorithm
OriError_max = math.pi*0.3  #Max acceptable orientation error for course correction algorithm
#Determines if the system uses particle filter and gets noisy measurements
uncertainty = False

clk = time.time()

occ_grid, obj_list, center, scale, map_size = create_occ_grid.load_occ_grid("test_occ")
occ_grid = create_occ_grid.mask_occ_grid(occ_grid, width, False)
scale_factor = len(occ_grid)/map_size[0]
map_size = np.asarray([[0,99],[0,99]])
graph = pickle.load(open('test_graph_PRM_basic_curve_few_400_mask.pickle', 'rb'))
rdm= True

if rdm == True:
    
    start = pathing.sample_grid(occ_grid, map_size)
    goal = pathing.sample_grid(occ_grid, map_size)
else:
    start = SE2(20,20,0)
    goal = SE2(90,65,0)
    
xyt_start = start.xyt()
startPos = np.concatenate((xyt_start[0:2], [1.1]))
startAngle = [0,0,xyt_start[2]]

xyt_goal = goal.xyt()
goalPos = np.concatenate((xyt_goal[0:2], [0]))
goalAngle = [xyt_start[2],0,0]

graph_length = len(graph.nodes)
pathing.add_to_PRM(start, graph_length, occ_grid, graph, path_func_add, 15, cost_max, w_radius, width, max_wheel, opt_speed, 1, sigma_2, step_num, col_penalty, col_percent_max, map_size)
pathing.add_to_PRM(goal, graph_length+1, occ_grid, graph, path_func_add, 15, cost_max, w_radius, width, max_wheel, opt_speed, 1, sigma_2, step_num, col_penalty, col_percent_max, map_size)


node_path = pathing.A_star_search(graph, graph_length, graph_length+1, weight='weight')

#[path, weight, time, wh_speed,node_loc] = pathing.RecoverPath_iter(graph, node_path)
[path, weight, time, wh_speed,node_loc
 ] = pathing.RecoverPath_iter(graph, node_path)

for target in path:
    print(target.xyt())

physicsClient, robotID = simulation_setup.set_up(startPos, startAngle, scale_factor,False,"test_occ")



fig, ax = plt.subplots()
im = ax.imshow(occ_grid)
ax.scatter(startPos[0],startPos[1],color = 'cyan')
ax.scatter(goalPos[0],goalPos[1],color = 'magenta')
current_pose_true = np.broadcast_to(start.A,(particle_num,3,3))
current_pose_pred = current_pose_true
input()
step_count = 0
fix_count = 0

current_pose_pred,step_count,fix_count =simulation_setup.run_command_series(graph, occ_grid, current_pose_pred, ax, fig,node_path, time, weight, wh_speed, node_loc, 
                                    robotID,physicsClient, path_func_add, scale_factor, PosError_max, OriError_max, 
                                    cost_max, w_radius, width, max_wheel, opt_speed, particle_num, 
                                    sigma_2,sigma_p, step_num, col_penalty, col_percent_max, map_size, uncertainty,step_count,fix_count)

robPos, robOrn = p.getBasePositionAndOrientation(robotID)
robPos = np.asarray(robPos)
robAngle = p.getEulerFromQuaternion(robOrn)
pos_error_fin = np.sqrt(np.sum((robPos[0:2]-xyt_goal[0:2])**2))
angle_error_fin = abs(robAngle[2]-xyt_goal[2])
print(pos_error_fin)
print(angle_error_fin)
print(fix_count)
print(step_count)

'''
for i in range(len(time)):
    t = np.linspace(0, time[i], step_num)
    x = np.zeros((len(t), 1))
    y = np.zeros((len(t), 1))
    x_true = np.zeros((len(t), 1))
    y_true = np.zeros((len(t), 1))
    
    #current_pose = path[i]
    #x[0] = current_pose.t[0]
    #y[0] = current_pose.t[1]
    x_true[0] = current_pose_true[0,0, 2]
    y_true[0] = current_pose_true[0,1, 2]
    #current_pose = np.asarray(current_pose)
    
    current_pose_true = np.asarray(current_pose_true)
    for k_step in range(step_num-1):
        #current_pose = PF_prop(t[k_step],t[k_step+1],current_pose,wh_speed[i],w_radius,width,sigma_2)
        #x[k_step+1] = current_pose[0,2]
        #y[k_step+1] = current_pose[1,2]
        current_pose_true = pathing.PF_prop(
            t[k_step], t[k_step+1], current_pose_true, wh_speed[i], w_radius, width, sigma_2)
        x_true[k_step+1] = current_pose_true[0,0, 2]
        y_true[k_step+1] = current_pose_true[0,1, 2]

    # ax.plot(x,y,color='red')
    ax.plot(x_true, y_true, color='green')
    simulation_setup.drive_wheels(wh_speed[i][0], wh_speed[i][1], robotID)
    simPos=simulation_setup.step_sim(time[i], 80,robotID,physicsClient,ax,scale_factor)
    step_count+=1
    if node_loc[i] != -1:
        target = graph.nodes[node_loc[i]]['pose']
        PosError, OriError,measure_pos,measure_ori = simulation_setup.get_error(robotID, target.xyt(),scale_factor=10)
        print(f'Position Error: {PosError}')
        print(f'Orientation Error: {OriError}')
        if PosError>PosError_max or OriError > OriError_max:
            pred_pose = SE2(measure_pos[0],measure_pos[1],measure_ori)
            pathing.add_to_PRM(pred_pose, len(graph.nodes), occ_grid, graph, path_func_add, 10, cost_max, w_radius, width, max_wheel, opt_speed, particle_num, sigma_2, step_num, col_penalty, col_percent_max, map_size)
            node_path_fix = pathing.A_star_search(graph, graph_length, node_loc[i], weight='weight')
            [path_fix, weight_fix, time_fix, wh_speed_fix,node_loc_fix] = pathing.RecoverPath_iter(graph, node_path_fix)

'''
    
    
input()

    
p.disconnect()