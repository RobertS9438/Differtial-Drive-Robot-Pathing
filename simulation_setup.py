# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:56:26 2023

@author: rfsyl
"""
#This is all of the utility code for running the simulation

import pybullet as p
import time
import pybullet_data
import os
import urdf_parser_py
import pickle
import pandas as pd
import create_occ_grid
import numpy as np
import pathing
from spatialmath import SE2
import copy
import math

def load_obstacles(client,occ_grid_filename,scale_factor):
    occ_grid, obj_list, center, scale, map_size = create_occ_grid.load_occ_grid(occ_grid_filename)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0,0)
    
    obj_id = []
    shape_id = []
    
    visualShapeID = -1
    mass = 0
    
    for obj_num in range(len(obj_list)):
        
        if obj_list[obj_num] == 'circle':
            #obstacle parameters multiplied by 10 to compensate for scale difference
            shape = p.createCollisionShape(p.GEOM_CYLINDER, radius = scale[obj_num][0]*scale_factor,height=4)
            visshape = p.createVisualShape(p.GEOM_CYLINDER, radius = scale[obj_num][0]*scale_factor, length = 4, rgbaColor = [0,0.3,1,1])
            shape_id.append(shape)
            base_pos = [center[obj_num][1]*scale_factor,center[obj_num][0]*scale_factor,0]
            #base_pos = [center[obj_num][0],center[obj_num][1],0]
            #base_pos = np.concatenate((center[obj_num],np.asarray([0])))
            base_orientation = [0,0,0,1]
            
            ID = p.createMultiBody(mass, shape,visshape,base_pos,base_orientation)
            obj_id.append(ID)
            
    #p.setAdditionalSearchPath(os.path.join(os.getcwd(),"urdf"))
    #boxId = p.loadURDF("rover.urdf",startPos, startOrientation)
            
    
    return obj_id

def drive_wheels(wh_l,wh_r,robotID,sigma_2):
    max_force = 50000
    #This need to be done for curve planner method
    if type(wh_l) == list:
        wh_l = wh_l[0]
        wh_r = wh_r[0]
    #[wh_l,wh_r] = np.random.multivariate_normal([wh_l,wh_r], sigma_2*np.identity(2))
    
    p.setJointMotorControl2(robotID, 1,
                            controlMode = p.VELOCITY_CONTROL,
                            targetVelocity = wh_l,
                            force = max_force)
    
    p.setJointMotorControl2(robotID, 2,
                            controlMode = p.VELOCITY_CONTROL,
                            targetVelocity = wh_r,
                            force = max_force)
    
    return


def set_up(startPos,orientation,scale_factor,mask,occ_name):
    if p.isConnected(0)==True:
        p.disconnect()
    
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    #planeId = p.loadURDF("plane.urdf")
    #startPos = [0,0,1]
    startOrientation = p.getQuaternionFromEuler(orientation)
    if mask:
        load_obstacles(physicsClient, occ_name,0)
    else:
        load_obstacles(physicsClient, occ_name,scale_factor)
    
    p.setAdditionalSearchPath(os.path.join(os.getcwd(),"urdf"))
    roverID = p.loadURDF("rover_rad_1.urdf",startPos, startOrientation)
    p.changeDynamics(roverID, 1, lateralFriction=15.0)
    p.changeDynamics(roverID, 2, lateralFriction=15.0)
    robPos, robOrn = p.getBasePositionAndOrientation(roverID)
    print(robPos)
    print(p.getEulerFromQuaternion(robOrn))
    
    
    return physicsClient,roverID


def step_sim(T,time_accel,robotID,physicsClient,ax,scale_factor):
    if type(T)== np.ndarray or type(T)==list:
        T = T[0]
        
    step_num = round(T*240)
    robAngle_array = np.zeros(step_num)
    robX = np.zeros(step_num)
    robY = np.zeros(step_num)
    #x_est = np.zeros(step_num)
    #y_est = np.zeros(step_num)
    
    #x_est[0] = np.mean(current_pose_pred[:,0, 2])
    #y_est[0] = np.mean(current_pose_pred[:,1, 2])
    #current_pose = np.asarray(current_pose)
    
    #current_pose_pred= np.asarray(current_pose_pred)
    
    for t in range(step_num):
        p.stepSimulation(physicsClient)
        robPos, robOrn = p.getBasePositionAndOrientation(robotID)
        robAngle_array[t] = p.getEulerFromQuaternion(robOrn)[2]
        robX[t] = robPos[0]#*scale_factor
        robY[t] = robPos[1]#*scale_factor
        #time.sleep(1./240./time_accel)
        '''
        current_pose_pred = pathing.PF_prop(
            0, 1/240, current_pose_pred, wh_speed, w_radius, width, sigma_2)
        x_est[t] = np.mean(current_pose_pred[:,0, 2])
        y_est[t] = np.mean(current_pose_pred[:,1, 2])
        '''
        
    robPos, robOrn = p.getBasePositionAndOrientation(robotID)
    robAngle = p.getEulerFromQuaternion(robOrn)[2]
    
    ax.plot(robX,robY,color= 'red')
    #ax.plot(x_est,y_est,color= 'green')

    print(f'Robot Position: {robPos[0:2]}')
    print(f'Orientation: {robAngle}')    
        
    return np.asarray(robPos[0:2]),robAngle,step_num,#current_pose_pred

def get_error(robotID, target,scale_factor):
    
    robPos, robOrn = p.getBasePositionAndOrientation(robotID)
    robPos = np.asarray(robPos)
    robAngle = p.getEulerFromQuaternion(robOrn)
    
    pos_error = np.sqrt(np.sum((robPos[0:2] - target[0:2])**2))
    ori_error = abs(robAngle[2] - target[2])
    if ori_error > math.pi:
        ori_error-=2*math.pi
    
    measure_pos = robPos[0:2]
    measure_ori = robAngle[2]
    
    return pos_error, ori_error, measure_pos,measure_ori

def run_commands(robotID,time_accel,wh_l,wh_r,time,ax,step_count,scale_factor):
    
    for i in range(len(time)):
        
        drive_wheels(wh_l[i],wh_r[i],robotID)
        [Pos,Ang,step_add] =step_sim(time[i],time_accel,robotID,physicsClient,ax,scale_factor)

        
    return ()


def run_command_series(graph,occ_grid,current_pose_pred,ax,fig,node_path,time,weight,wh_speed,
                       node_loc, robotID, physicsClient, path_func_add,scale_factor,PosError_max,OriError_max,
                       cost_max, w_radius, width, max_wheel, opt_speed, 
                       particle_num, sigma_2,sigma_p, step_num, col_penalty,
                       col_percent_max, map_size,uncertainty,step_count,fix_count):
    
    for i in range(len(time)):
        
        
        t = np.linspace(0, time[i], step_num)
        x = np.zeros((len(t), 1))
        y = np.zeros((len(t), 1))
        x_est = np.zeros((len(t), 1))
        y_est = np.zeros((len(t), 1))
        
        #current_pose = path[i]
        #x[0] = current_pose.t[0]
        #y[0] = current_pose.t[1]
        x_est[0] = np.mean(current_pose_pred[:,0, 2])
        y_est[0] = np.mean(current_pose_pred[:,1, 2])
        #current_pose = np.asarray(current_pose)
        
        current_pose_pred_plot= np.asarray(current_pose_pred)
        for k_step in range(step_num-1):
            #current_pose = PF_prop(t[k_step],t[k_step+1],current_pose,wh_speed[i],w_radius,width,sigma_2)
            #x[k_step+1] = current_pose[0,2]
            #y[k_step+1] = current_pose[1,2]
            current_pose_pred_plot = pathing.PF_prop(
                t[k_step], t[k_step+1], current_pose_pred_plot, wh_speed[i], w_radius, width, sigma_2)
            x_est[k_step+1] = np.mean(current_pose_pred_plot[:,0, 2])
            y_est[k_step+1] = np.mean(current_pose_pred_plot[:,1, 2])

        # ax.plot(x,y,color='red')
        ax.plot(x_est, y_est, color='green')
        
        current_pose_pred = pathing.PF_prop(
            0, time[i], current_pose_pred, wh_speed[i], w_radius, width, sigma_2)
        drive_wheels(wh_speed[i][0], wh_speed[i][1], robotID,sigma_2)
        simPos,simAngle,step_size=step_sim(time[i], 1,robotID,physicsClient,ax,scale_factor)
        Zt = np.random.multivariate_normal(
            simPos, sigma_p*np.identity(2))
        At = np.random.normal(
            simAngle, sigma_p)
        if uncertainty == True:
            current_pose_pred = pathing.PF_update(current_pose_pred, Zt, sigma_p)
            pred_x = np.mean(current_pose_pred[:,0,2])
            pred_y = np.mean(current_pose_pred[:,1,2])
            pred_ori = np.mean(np.arctan2(current_pose_pred[:,1,0],current_pose_pred[:,2,0]))
            pred_pose = SE2(pred_x,pred_y,pred_ori)
        else:
            pred_pose = SE2(Zt[0],Zt[1],At)    
        
        step_count +=step_size
        
        if node_loc[i] != -1:
            target = graph.nodes[node_loc[i]]['pose']
            PosError, OriError,measure_pos,measure_ori = get_error(robotID, target.xyt(),scale_factor)
            print(f'Position Error: {PosError}')
            print(f'Orientation Error: {OriError}')
            if uncertainty:
                bel_posErr, bel_OriErr = bel_error(current_pose_pred,target)
                print(f'Believed Position Error: {bel_posErr}')
                print(f'Believed Orientation Error: {bel_OriErr}')
            else:
                bel_posErr = PosError
                bel_OriErr = OriError
            
            
                
            if bel_posErr>PosError_max or abs(bel_OriErr) > OriError_max:
                node_index = len(graph.nodes)
                target_fix = node_loc[i]
                pathing.add_to_PRM(pred_pose, node_index, occ_grid, graph, path_func_add, 20, cost_max, w_radius, width, max_wheel, opt_speed, 1, sigma_2, step_num, col_penalty, col_percent_max, map_size)
                node_path_fix = pathing.A_star_search(graph, node_index, node_path[-1], weight='weight')
                [path_fix, weight_fix, time_fix, wh_speed_fix,node_loc_fix] = pathing.RecoverPath_iter(graph, node_path_fix)
                
                    
                current_pose_pred,step_count,fix_count = run_command_series(graph,occ_grid,current_pose_pred,ax,fig,node_path_fix, time_fix,weight_fix,wh_speed_fix,
                                      node_loc, robotID,physicsClient, path_func_add,scale_factor,PosError_max,OriError_max,
                                      cost_max, w_radius, width, max_wheel, opt_speed, 
                                      particle_num, sigma_2, sigma_p, step_num, col_penalty,
                                      col_percent_max, map_size,uncertainty,step_count,fix_count)
                fix_count +=1
    
                break
                break
            
    return current_pose_pred,step_count,fix_count



def bel_error(Xt,target):
    
     
    x= np.mean(Xt[:,0,2]) 
    y= np.mean(Xt[:,1,2])
    theta = np.mean(np.arctan2(Xt[:,1,0],Xt[:,0,0]))
    
    xyt = target.xyt()
    PosError = np.sqrt((xyt[0]-x)**2+(xyt[1]-y)**2)
    AngError = abs(theta-xyt[2])
    if AngError> math.pi:
        AngError -= 2*math.pi
    
    return PosError, AngError
            
            
            
            
