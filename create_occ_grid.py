# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:51:31 2023

@author: rfsyl
"""
#This code is for creating and handingte occupancy grid

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import cv2

def gen_obstacle(obs_type,pos_limits, scale_limits):
    
    obj = np.random.choice(obs_type)
    
    if obj == "rectangle":
        
        center = gen_pos(pos_limits)
        sides = gen_size(scale_limits,2)
        return(obj,center,sides)
    elif obj == "circle":
        center = gen_pos(pos_limits)
        r = gen_size(scale_limits,1)
        return(obj,center,r)
    
    elif obj == "triangle":
        center = gen_pos(pos_limits)
        sides = gen_size(scale_limits,2)
        return(obj,center,sides)
    else:
        return("Invalid shape")
    


#Generate random position poslimits[0] and poslimits[1] should be the limits of x and y coord respectively
def gen_pos(pos_limits):
    
    x = np.random.uniform(pos_limits[0][0],pos_limits[0][1])
    y = np.random.uniform(pos_limits[1][0],pos_limits[1][1])
    
    return(x,y)

def gen_size(size_limits,num):
    
    size = np.random.uniform(size_limits[0],size_limits[1],num)
    
    return size

def gen_occ_grid(N,M,map_size,obj_num,obj_types,pos_limits,scale_limits):
    
    occ_grid = np.zeros((N,M))
    obj_list = []
    center_list = []
    rot_list = [] #List of obstacle rotations - Not yet implemented
    scale_list = []
    error = False
    
    for i in range(obj_num):
        
        [obj,center,size] = gen_obstacle(obj_types, pos_limits, scale_limits)
        if obj != "Invalid shape":
            obj_list.append(obj)
            center_list.append((center))
            scale_list.append((size))
            
            if obj == "circle":
                u_line = (np.linspace(0,map_size[0],N) - center[0])**2
                v_line = (np.linspace(0,map_size[1],M) - center[1])**2
                u_mat = np.transpose(np.broadcast_to(u_line,(M,N)))
                v_mat = np.broadcast_to(v_line,(N,M))
                
                dis_mat = np.sqrt(u_mat+v_mat)
                
                for u in range(N):
                    for v in range(M):
                        if dis_mat[u,v]<=size:
                            occ_grid[u,v] = 1
                        
                    
                
            else:
                return()
                
        else:
            error =True
            
    
    return(occ_grid,np.reshape(np.asarray(obj_list),(obj_num,1)),np.asarray(center_list),np.asarray(scale_list),error)

def save_occ_grid(name,N,M,map_size,obj_num,obj_types,pos_limits,scale_limits):
    
    [occ_grid,obj_list,center_list,scale_list,error_check] =gen_occ_grid(N,M,map_size,obj_num,obj_types,pos_limits,scale_limits)
    list_size = len(obj_list)
    map_size = np.broadcast_to(map_size,(list_size,2))
    
    if error_check == False:
        #save_grid = pd.DataFrame(list(zip([occ_grid,obj_list,center_list,scale_list,map_size])),columns=["occ_grid","obj","center","obj_scale","map_size"])
        save_grid = pd.DataFrame(occ_grid)
        save_grid_info = pd.DataFrame(list(zip(obj_list,center_list,scale_list,map_size)),columns=["obj","center","obj_scale","map_size"])
        fig, ax = plt.subplots()
        im = ax.imshow(save_grid)
        plt.show()
        
        save_grid.to_csv(name)
        save_grid_info.to_csv(name+"_info")
        #save_grid_info.to
        return("Occ_grid Saved")
        
    else:
        return("Error")


def load_occ_grid(name):

    occ_grid = pd.read_csv(name)
    
    occ_grid = occ_grid.to_numpy()
    occ_grid = occ_grid[:,1:]
    occ_grid_info = pd.read_csv(name+'_info')
    
    obj_list  = []
    center = []
    scale = []
    map_size = np.fromstring(occ_grid_info['map_size'][0][1:-1], sep=' ')
    for i in range(len(occ_grid_info["obj"])):
        obj_list.append(occ_grid_info["obj"][i][2:-2])
        center.append(np.fromstring(occ_grid_info["center"][i][1:-1], sep= ' '))
        scale.append(np.fromstring(occ_grid_info["obj_scale"][i][1:-1], sep= ' '))
    
    scale = np.asarray(scale)
    center = np.asarray(center)
    return(occ_grid,obj_list,center,scale,map_size)
                
                        
def mask_occ_grid(occ_grid,width,fil_cost):
    
    #Can adjust Buffer Zone
    #radius = width#/2*2  Double Buffer zone
    radius = math.ceil(width/2)
    radius_2 = radius**2
    [N,M] = occ_grid.shape
    
    occ_grid_alt = cv2.GaussianBlur(occ_grid,(15,15), 3.0)
    '''
    if fil_cost:
        occ_grid_alt = occ_grid
    else:
        occ_grid_alt = occ_grid
    '''
    for i in range(N):
        for j in range(M):
            if occ_grid[i,j] == 1:
                for a in range(max(0,i-radius),min(N,i+radius)):
                    for b in range(max(0,j-radius),min(M,j+radius)):
                        if occ_grid_alt[a,b] != 1:
                            dx = j-b
                            dy = i-a
                            d= dx**2+dy**dy
                            if d <=radius_2:
                                occ_grid_alt[a,b] = 1

    
   
    
    return occ_grid_alt
    