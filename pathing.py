# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:21:47 2023

@author: rfsyl
"""
#All of the pathing algorithms are stored here
#This includes the code for creating the RRG's, the particle filter update and
#propogation step code
#It also has the search algorithm and path reconstruction algorithms
import numpy as np
import math
import scipy as sp
import heapq as hq
import heapdict as hd
import math
from PIL import Image
from matplotlib import pyplot as plt
import random
import networkx as nx
import spatialmath.base as sm
import spatialmath.pose2d as s
from spatialmath import SE2
import copy


def PF_prop(t1, t2, Xt, wh_speed, r, w, sigma_2):

    X_next = np.empty(Xt.shape)
    # Create set of actual wheel velocities representative of probability distribution
    E = np.random.multivariate_normal(
        np.array([0.0, 0.0]), sigma_2*np.identity(2), len(Xt))
    wh_speed = np.asarray(wh_speed)
    if wh_speed.shape == (2,1):
        wh_dot = np.broadcast_to(np.transpose(wh_speed), (len(Xt), 2))+E
    else:
        wh_dot = np.broadcast_to(wh_speed, (len(Xt), 2))+E


    # Create lie algebra matrix for each wheel velocity pair
    omega_dot = np.zeros((len(Xt), 3, 3))
    omega_dot[:, 1, 0] = r/w*(wh_dot[:, 1]-wh_dot[:, 0])
    omega_dot[:, 0, 1] = r / w * (wh_dot[:, 0] - wh_dot[:, 1])
    omega_dot[:, 0, 2] = r / 2 * (wh_dot[:, 0] + wh_dot[:, 1])

    # Calculate pose at end of path taken from t1 to t2
    for i in range(len(Xt)):
        X_next[i] = np.matmul(Xt[i], sp.linalg.expm(omega_dot[i]*(t2-t1)))

    return X_next

#Particle Filter Update Step
def PF_update(X_prior,Zt,sigma_p):

    weight = np.zeros(len(X_prior))
    inv_cov_mat = np.identity(2)*sigma_p**(-2)
    pos_prior = X_prior[:,0:2,2]

    for j in range(len(X_prior)):
        #Assign weight to each particle in accordance with its probability given Zt
        weight[j] = np.linalg.norm(1/(2*math.pi*sigma_p)*sp.linalg.expm(-1/2*np.matmul(np.transpose(Zt-pos_prior[j]),np.matmul(inv_cov_mat,Zt-pos_prior[j]))))
        
    weight = weight/np.sum(weight)
    if any(np.isnan(weight)):
        input()

    #Take weighted samples from prior particle set with replacement to create posterior particle set
    X_posterior = X_prior[np.random.choice(range(len(X_prior)),len(X_prior),replace= True, p =weight)]

    return X_posterior


def sample_grid(M, grid_scale):

    sample = ()
    #y = np.linspace(0,grid_scale[0],len(M))
    #x = np.linspace(0,grid_scale[1],len(M[0]))
    done = False

    while done == False:
        i = random.sample(range(len(M)), k=1)
        j = random.sample(range(len(M[0])), k=1)
        angle = np.random.uniform(0, math.pi*2)
        if M[i[0], j[0]] != 1:
            # Need to reverse indices for plotting functions
            #sample_pose = pos_angle_to_pose(j[0],i[0],angle)
            sample_pose = SE2(j[0], i[0], angle)
            done = True

    return sample_pose


def pos_angle_to_pose(pos_x, pos_y, angle):

    pose = np.zeros((3, 3))

    pose[0:2, 0:2] = [[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]]  # Check signs here

    pose[0, 2] = pos_x
    pose[1, 2] = pos_y

    return pose

# Make path graph with cost values and wheel speeds


def make_PRM(grid_scale, occ_grid, N_samples, path_func, d_max, cost_max, w_radius, width, max_wheel, opt_speed, particle_num, sigma_2, step_num, col_penalty, col_percent_max):

    G = nx.DiGraph()
    v_list = []
    # To avoid repeating this calculation
    d_max_2 = d_max**2

    for n in range(N_samples):
        new = False
        while new != True:
            v_new = sample_grid(occ_grid, grid_scale)
            test_new = True
            for other_node in v_list:
                if np.allclose(other_node, v_new):
                    test_new = False

            if test_new == True:
                new = True
                v_list.append(v_new)

        new_node(occ_grid, G, v_new, n,path_func, d_max_2, cost_max, w_radius, width, max_wheel,
                 opt_speed, particle_num, sigma_2, step_num, col_penalty, col_percent_max, grid_scale)
        
        if n %10 == 0:
            print(f'{n} nodes added')
            print(f'{len(G.edges)} edges added')

    return G

#Adds random nodes to existing PRM
def expand_PRM(M_grid, occ_grid,  graph, N_samples, path_func, d_max, cost_max, w_radius, width, max_wheel, opt_speed, particle_num, sigma_2, step_num, col_penalty, col_percent_max, map_size):

    # To avoid repeating this calculation
    d_max_2 = d_max**2

    for n in range(N_samples):
        new = False
        while new != True:
            v_new = sample_grid(M_grid)
            if v_new not in graph.nodes[:]['pose']:
                new = True

        new_node(occ_grid, graph, v_new, n, path_func, d_max_2, cost_max, w_radius, width, max_wheel,
                 opt_speed, particle_num, sigma_2, step_num, col_penalty, col_percent_max, map_size)

    return ()

#Adds specific pose to PRM
def add_to_PRM(new_node_pose, new_node_num, occ_grid, graph, path_func, d_max, cost_max, w_radius, width, max_wheel, opt_speed, particle_num, sigma_2, step_num, col_penalty, col_percent_max, map_size):
    # To avoid repeating this calculation
    d_max_2 = d_max**2

    new_node(occ_grid, graph, new_node_pose, new_node_num,path_func, d_max_2, cost_max, w_radius, width,
             max_wheel, opt_speed, particle_num, sigma_2, step_num, col_penalty, col_percent_max, map_size)

    return (graph)

#Recovers commands from node_path ourout of A_star_search
def RecoverPath(path_graph, node_path):

    path = []
    time = []
    wh_speed = []
    node_loc = []
    weight = 0
    for i in range(len(node_path)):
        path.append(path_graph.nodes[node_path[i]]['pose'])

        if i != 0:
            weight += path_graph.get_edge_data(
                node_path[i-1], node_path[i])['weight']
            time.append(path_graph.get_edge_data(
                node_path[i-1], node_path[i])['time'])
            wh_speed.append(path_graph.get_edge_data(
                node_path[i-1], node_path[i])['wh_speed'])
            node_loc.append(node_path[i])

    return (path, weight, time, wh_speed,node_loc)

#Same as above but for the edges with a series of commands each
def RecoverPath_iter(path_graph, node_path):

    path = []
    time = []
    wh_speed = []
    node_loc = []
    weight = 0
    for i in range(len(node_path)):
        path.append(path_graph.nodes[node_path[i]]['pose'])
        
            

        if i != 0:
            weight_temp = path_graph.get_edge_data(
                node_path[i-1], node_path[i])['weight']
            time_temp = path_graph.get_edge_data(
                node_path[i-1], node_path[i])['time']
            wh_speed_temp = path_graph.get_edge_data(
                node_path[i-1], node_path[i])['wh_speed']
            if type(time_temp)  == np.float64:
                time.append(time_temp)
                wh_speed.append(wh_speed_temp)
                weight+=weight_temp
                node_loc.append(node_path[i])
            else:
                
                wh_speed_temp = np.transpose(np.asarray(
                    wh_speed_temp).reshape((2, len(time_temp))))

                for j in range(len(time_temp)):
                    if i != 0:
                        weight += np.sum(weight_temp)
                        time.append(time_temp[j])
                        wh_speed.append(wh_speed_temp[j])
                        if j != len(time_temp)-1:
                            node_loc.append(-1)
                        else:
                            node_loc.append(node_path[i])

    return (path, weight, time, wh_speed,node_loc)

# Function to add new node and any valid edges to graph


def new_node(occ_grid, graph, v_new, node_num, path_func, d_max_2, cost_max, w_radius, width, max_wheel, opt_speed, k, sigma_2, step_num, col_penalty, col_percent_max, map_size):
    graph.add_node(node_num, pose=v_new)
    for v in graph:
        if np.allclose(graph.nodes[v]['pose'], v_new) == False:
            if dist_2_calc(v_new, graph.nodes[v]['pose']) <= d_max_2:
                #[cost,phi_dot_l,phi_dot_r,T,col_percent,end_nodes] = curve_planner(occ_grid,v_new,graph.nodes[v]['pose'],w_radius,width,max_wheel,opt_speed,k,sigma_2,step_num,col_penalty,map_size)
                #[cost,phi_dot_l,phi_dot_r,T,col_percent,end_nodes] = curve_planner_interp(occ_grid,v_new,graph.nodes[v]['pose'],w_radius,width,max_wheel,opt_speed,k,sigma_2,step_num,col_penalty,map_size)
                #[cost,phi_dot_l,phi_dot_r,T,col_percent,end_nodes] = spin_and_move_planning(occ_grid,v_new,graph.nodes[v]['pose'],w_radius,width,max_wheel,opt_speed,k,sigma_2,step_num,col_penalty,map_size)
                #[cost, phi_dot_l, phi_dot_r, T, col_percent, end_nodes] = est_curve_alt(
                #    occ_grid, v_new, graph.nodes[v]['pose'], w_radius, width, max_wheel, opt_speed, k, sigma_2, step_num, col_penalty, map_size)
                #[cost,phi_dot_l,phi_dot_r,T,col_percent,end_nodes] = iterate_curve(occ_grid,v_new,graph.nodes[v]['pose'],w_radius,width,max_wheel,opt_speed,k,sigma_2,step_num,col_penalty,map_size)
                [cost,phi_dot_l,phi_dot_r,T,col_percent,end_nodes] = path_func(occ_grid,v_new,graph.nodes[v]['pose'],w_radius,width,max_wheel,opt_speed,k,sigma_2,step_num,col_penalty,map_size)
                if (cost <= cost_max) & (col_percent <= col_percent_max):
                    graph.add_edge(v, node_num, weight=cost, wh_speed=[
                                   phi_dot_l, phi_dot_r], time=T)

                #[cost,phi_dot_l,phi_dot_r,T,col_percent,end_nodes] = curve_planner(occ_grid,graph.nodes[v]['pose'],v_new,w_radius,width,max_wheel,opt_speed,k,sigma_2,step_num,col_penalty,map_size)
                #[cost,phi_dot_l,phi_dot_r,T,col_percent,end_nodes] = curve_planner_interp(occ_grid,graph.nodes[v]['pose'],v_new,w_radius,width,max_wheel,opt_speed,k,sigma_2,step_num,col_penalty,map_size)
                #[cost,phi_dot_l,phi_dot_r,T,col_percent,end_nodes] = spin_and_move_planning(occ_grid,graph.nodes[v]['pose'],v_new,w_radius,width,max_wheel,opt_speed,k,sigma_2,step_num,col_penalty,map_size)
                #[cost, phi_dot_l, phi_dot_r, T, col_percent, end_nodes] = est_curve_alt(
                #    occ_grid, graph.nodes[v]['pose'], v_new, w_radius, width, max_wheel, opt_speed, k, sigma_2, step_num, col_penalty, map_size)
                #[cost,phi_dot_l,phi_dot_r,T,col_percent,end_nodes] = iterate_curve(occ_grid,graph.nodes[v]['pose'],v_new,w_radius,width,max_wheel,opt_speed,k,sigma_2,step_num,col_penalty,map_size)
                [cost,phi_dot_l,phi_dot_r,T,col_percent,end_nodes] = path_func(occ_grid,graph.nodes[v]['pose'],v_new,w_radius,width,max_wheel,opt_speed,k,sigma_2,step_num,col_penalty,map_size)
                if (cost <= cost_max) & (col_percent <= col_percent_max):
                    graph.add_edge(node_num, v, weight=cost, wh_speed=[
                                   phi_dot_l, phi_dot_r], time=T)

    return()


def curve_planner(occ_grid, new_node, start_node, w_radius, width, max_wheel, opt_speed, k, sigma_2, step_num, col_penalty, map_size):

    [phi_dot_l, phi_dot_r, T, omega_dot_prime] = get_wheel_speeds(
        new_node, start_node, w_radius, width, max_wheel, opt_speed)

    omega_dot = create_lie_algebra(
        phi_dot_l, phi_dot_r, w_radius, width, k, sigma_2)

    [cost, test_target, col_prime,cost_path] = local_cost_calc(
        occ_grid, omega_dot_prime, T, start_node, new_node, step_num, col_penalty, map_size)
    end_nodes = []
    if col_prime == False:

        cost_total = 0
        col_count = 0

        for i in range(k):
            [cost, end_point, col_temp,cost_path] = local_cost_calc(
                occ_grid, omega_dot[i], T, start_node, new_node, step_num, col_penalty, map_size)
            cost_total = cost_total + cost_path
            # end_nodes.append(end_point)
            end_nodes = end_point
            if col_temp == True:
                col_count = col_count+1

        cost_total = cost_total/k
        col_percent = col_count/k
    else:
        cost_total = cost_path
        col_percent = 1
        end_nodes = start_node
    
    if cost>2:
        col_percent=1
        
    return(cost_total,[phi_dot_l], [phi_dot_r], [T], col_percent, end_nodes)


def curve_planner_interp(occ_grid, new_node, start_node, w_radius, width, max_wheel, opt_speed, k, sigma_2, step_num, col_penalty, map_size):

    interp_path = start_node.interp(new_node, step_num)

    current_node = start_node
    old_end_node = current_node

    wh_r = np.zeros((step_num-1, 1))
    wh_l = np.zeros((step_num-1, 1))
    time = np.zeros((step_num-1, 1))
    cost_total = 0
    for i in range(len(interp_path)-1):

        [cost, wh_l[i], wh_r[i], time[i], col_percent, end_nodes] = curve_planner(
            occ_grid, interp_path[i+1], current_node, w_radius, width, max_wheel, opt_speed, k, sigma_2, step_num, col_penalty, map_size)
        cost_total += cost
        if col_percent >= 0.3:
            cutoff = np.max([0, i-1])
            wh_r = wh_r[0:cutoff]
            wh_l = wh_l[0:cutoff]
            time = time[0:cutoff]
            break
        else:
            old_end_node = end_nodes
            current_node = end_nodes

    return(cost_total, wh_l, wh_r, time, col_percent, old_end_node)

#The spin and rotate code
def spin_and_move_planning(occ_grid, new_node, start_node, w_radius, width, max_wheel, opt_speed, k, sigma_2, step_num, col_penalty, map_size):

    wheel_commands, inter_poses = wheel_speed_spin_move(
        new_node, start_node, w_radius, width, max_wheel, opt_speed)
    current_pose = start_node
    cost_total = 0
    col_final = False

    for i in range(3):
        omega_dot = create_lie_algebra(
            wheel_commands[i, 0], wheel_commands[i, 1], w_radius, width, k, sigma_2)
        [cost, endpoint, col_temp,cost_path] = local_cost_calc(
            occ_grid, omega_dot[0], wheel_commands[i, 2], current_pose, inter_poses[i], step_num, col_penalty, map_size)
        cost_total += cost_path
        if col_temp == True:
            col_final = 1
            cost_total += dist_2_calc(current_pose, new_node)*col_penalty
            wheel_commands = wheel_commands[0:i, :]
            break
        else:
            current_pose = endpoint

    cost_total = cost_total/(3*step_num)

    return cost_total, wheel_commands[:, 0], wheel_commands[:, 1], wheel_commands[:, 2], col_final, current_pose


def wheel_speed_spin_move(new_node, start_node, w_radius, width, max_wheel, opt_speed):

    wh_l_r_t = np.zeros((3, 3))
    inv_start = start_node.inv()
    transform = inv_start*new_node
    xyt = transform.xyt()
    # calculate parameters of  first spin towards target
    angle_ini = np.arctan2(xyt[1], xyt[0])
    wh_l_r_t[0, 2] = abs(width/2*angle_ini/(max_wheel/2*w_radius))
    if angle_ini >= 0:
        k_mult = 1
    else:
        k_mult = -1

    wh_l_r_t[0, 1] = k_mult*max_wheel/2
    wh_l_r_t[0, 0] = -k_mult*max_wheel/2

    # Calculate Parameters of motion towards target
    dist = np.sqrt(np.sum(xyt[0:2]**2))
    wh_l_r_t[1, 0:3] = [max_wheel/2, max_wheel/2, dist/(max_wheel/2)]

    # Align with target frame
    angle_fin = xyt[2] - angle_ini
    wh_l_r_t[2, 2] = abs(width/2*angle_fin/(max_wheel/2*w_radius))
    if angle_fin >= 0:
        k_mult = 1
    else:
        k_mult = -1

    wh_l_r_t[2, 1] = k_mult*max_wheel/2
    wh_l_r_t[2, 0] = -k_mult*max_wheel/2

    # get intermediate poses
    poses = []

    end = new_node.xyt()
    ini = start_node.xyt()
    poses.append(SE2(ini[0], ini[1], ini[2]+angle_ini))
    poses.append(SE2(end[0], end[1], end[2]-angle_fin))
    poses.append(new_node)

    #error = np.sqrt(np.sum(np.sum((transform - poses[2])**2)))

    # poses.append(end_node)

    '''
    if error > 2:
        print("Error with Spin-Move-Commands: %f",error)
        
    
    poses.append(start_node)
    poses.append(new_node)
    poses.append(new_node)
    '''

    return wh_l_r_t, np.asarray(poses)

#The estimated curve algorithm
def est_curve_alt(occ_grid, new_node, start_node, w_radius, width, max_wheel, opt_speed, k, sigma_2, step_num, col_penalty, map_size):

    inv_start = start_node.inv()
    test = inv_start*new_node
    test_xyt = test.xyt()
    
    if (test_xyt[1]*test_xyt[2] < 0)| (test_xyt[0]<0):
        
        offset = width*1
        
        mid_point_1 = np.asarray([offset, 0])
        mid_point_2 = np.asarray([test_xyt[0]-offset*np.cos(test_xyt[2]),test_xyt[1]-offset*np.sin(test_xyt[2])])
        
        true_midpoint = (mid_point_1+mid_point_2)/2
        dif = mid_point_2-mid_point_1
        mid_angle = np.arctan2(dif[1],dif[0])
        
        inter_node = start_node*SE2(true_midpoint[0],true_midpoint[1],mid_angle)
        phi_dot_l = np.zeros((4))
        phi_dot_r = np.zeros((4))
        T = np.zeros((4))
        omega_dot_prime = np.zeros((4, 3, 3))
        target = []
        [phi_dot_l[0:2], phi_dot_r[0:2], T[0:2], omega_dot_prime[0:2], target] = wheel_speed_alt_basic(
            inter_node, start_node, w_radius, width, max_wheel, opt_speed)
        [phi_dot_l[2:4], phi_dot_r[2:4], T[2:4], omega_dot_prime[2:4], target_temp] = wheel_speed_alt_basic(
            new_node, inter_node, w_radius, width, max_wheel, opt_speed)
        target.append(target_temp[0])
        target.append(target_temp[1])
    else:
        [phi_dot_l, phi_dot_r, T, omega_dot_prime, target] = wheel_speed_alt_basic(
            new_node, start_node, w_radius, width, max_wheel, opt_speed)
    
    '''
    if (test_xyt[1]*test_xyt[2] < 0):
        start_xyt = start_node.xyt()
        new_xyt = new_node.xyt()
        mid_point = (new_xyt + start_xyt)/2
        
        if test_xyt[2] <= 0:
            mult = 1
        else:
            mult = -1

        mid_point[2] = mid_point[2] + mult*math.pi/2
        inter_node = SE2(mid_point)
        phi_dot_l = np.zeros((4))
        phi_dot_r = np.zeros((4))
        T = np.zeros((4))
        omega_dot_prime = np.zeros((4, 3, 3))
        target = []
        [phi_dot_l[0:2], phi_dot_r[0:2], T[0:2], omega_dot_prime[0:2], target] = wheel_speed_alt_basic(
            inter_node, start_node, w_radius, width, max_wheel, opt_speed)
        [phi_dot_l[2:4], phi_dot_r[2:4], T[2:4], omega_dot_prime[2:4], target_temp] = wheel_speed_alt_basic(
            new_node, inter_node, w_radius, width, max_wheel, opt_speed)
        target.append(target_temp[0])
        target.append(target_temp[1])
    else:
        [phi_dot_l, phi_dot_r, T, omega_dot_prime, target] = wheel_speed_alt_basic(
            new_node, start_node, w_radius, width, max_wheel, opt_speed)
        '''
        
        
    

    cost_total = 0
    col_count = 0
    begin_node = start_node

    for h in range(len(phi_dot_l)):
        omega_dot = create_lie_algebra(
            phi_dot_l[h], phi_dot_r[h], w_radius, width, k, sigma_2)

        [cost_base, test_target, col_prime,cost_path_base] = local_cost_calc(
            occ_grid, omega_dot_prime[h], T[h], begin_node, target[h], step_num, col_penalty, map_size)
        end_nodes = []
        cost_sum = 0
        cost_total += cost_path_base
        if col_prime == False:

            for i in range(k):
                [cost, end_point, col_temp,cost_path] = local_cost_calc(
                    occ_grid, omega_dot[i], T[h], begin_node, target[h], step_num, col_penalty, map_size)
                cost_sum = cost_sum + cost
                # end_nodes.append(end_point)
                end_nodes = end_point
                if col_temp == True:
                    col_count = col_count+1

            cost_sum = cost_sum/k
            col_percent = col_count/k
        else:
            cost_sum = cost_base
            col_percent = 1
            end_nodes = begin_node
            break

        begin_node = end_nodes
        
    if cost_sum>=2:
        col_percent =1

    #cost_total += cost_sum

    return(cost_total, phi_dot_l, phi_dot_r, T, col_percent, end_nodes)

#Used to calculate Lie Algebra and extrac wheel speeds an time
def get_wheel_speeds(end_node, start_node, w_radius, width, wheel_speed_max, opt_speed):
    '''
    inv_start = np.zeros((3,3))
    inv_start[0:2,0:2] = np.transpose(start_node[0:2,0:2])
    inv_start[2,2] = 1
    inv_start[0:2,2] = -np.transpose(start_node[0:2,0:2])@start_node[0:2,2]
    '''
    inv_start = start_node.inv()
    #[W, V] = sp.linalg.eig(inv_start@end_node)

    #V_inv = np.linalg.inv(V)

    #log_W = np.log(W)
    M = inv_start*end_node
    #omega_dot = M.log()
    omega_dot = sm.transforms2d.trlog2(M.A)
    phi_dot_l = (omega_dot[0, 1]*width/2+omega_dot[0, 2])/w_radius
    phi_dot_r = omega_dot[0, 2]*2/w_radius - phi_dot_l

    t = max([abs(phi_dot_l)/wheel_speed_max, abs(phi_dot_r) /
            wheel_speed_max, abs(omega_dot[0, 2])/opt_speed])
    if t == 0:
        phi_dot_l = 0
        phi_dot_r = 0
    else:
        phi_dot_l = phi_dot_l/t
        phi_dot_r = phi_dot_r/t
        
    test = dist_calc(SE2().Exp(omega_dot),M)
    if test > 0.5:
        speed_change = test*math.pi/t
        dl = speed_change/w_radius*(phi_dot_l/(phi_dot_l+phi_dot_r))
        dr = speed_change/w_radius*(phi_dot_r/(phi_dot_l+phi_dot_r))
        if omega_dot[0,2]<0:
            phi_dot_l += dl
            phi_dot_r += dr
        else:
            phi_dot_l -= dl
            phi_dot_r -= dr
        if ((abs(phi_dot_l)>wheel_speed_max) or (abs(phi_dot_r)>wheel_speed_max)):
            if abs(phi_dot_l)>=abs(phi_dot_r):
                slow = wheel_speed_max/abs(phi_dot_l)
            else:
                slow = wheel_speed_max/abs(phi_dot_r)
            phi_dot_l *= slow
            phi_dot_r *= slow
            t /= slow
            
        omega_dot = create_lie_algebra(phi_dot_l, phi_dot_r, w_radius, width, 1, np.zeros((2,2)))*t
        omega_dot = omega_dot[0]
            
    
    return(phi_dot_l, phi_dot_r, t, omega_dot)


def iterate_curve(occ_grid, new_node, start_node, w_radius, width, max_wheel, opt_speed, k, sigma_2, step_num, col_penalty, map_size):
    
    error = 10000
    old_error = 10000
    max_error = 0.6
    current_node = start_node
    left_commands = []
    right_commands =[]
    Time = []
    path = []
    it_num =0
    it_min = 1
    min_error = 100
    good_est = False
    #while error>max_error:
    while good_est == False:
        [wh_l,wh_r,T,omega_prime] = get_wheel_speeds(new_node,current_node,w_radius,width,max_wheel,opt_speed)
        [wh_l,wh_r,good_est] = adjust_omega(wh_l, wh_r, omega_prime, w_radius, width)
        omega_dot = create_lie_algebra(wh_l,wh_r,w_radius,width,k,sigma_2)
        test_target = current_node*SE2().Exp(omega_dot[0]*T)
        error = error_pose(new_node,test_target)
        left_commands.append(wh_l)
        right_commands.append(wh_r)
        if error < max_error:
            good_est =True
        '''
        if min_error>=error:
            min_error = error
            it_min = copy.copy(it_num)
        if (error < max_error):
            Time.append(T)
            current_node = test_target
            path.append(test_target)
            break
        '''
        if (good_est):
            Time.append(T)
            current_node = test_target
            path.append(test_target)
            break
        elif it_num >= step_num:
            #path = np.asarray(path[0:it_min])
            #left_commands = np.asarray(left_commands[0:it_min])
            #right_commands = np.asarray(right_commands[0:it_min])
            # Time = np.asarray(Time[0:it_min])
            Time.append(T)
            path.append(new_node)
            break
        else:
            Time.append(T/step_num)
            current_node = current_node*SE2().Exp(omega_dot[0]*T/(step_num))
            path.append(current_node)
            old_error = error
            it_num +=1
            
    
    
    cost_total = 0
    col_final = 0
    current_node = start_node
    for i in range(len(Time)):
        omega_dot = create_lie_algebra(left_commands[i],right_commands[i],w_radius,width,k,sigma_2)
        [cost,end_pose,col,cost_path] = local_cost_calc(occ_grid, omega_dot[0], Time[i], current_node, new_node, step_num, col_penalty, map_size)
        cost_total += cost_path
        if col ==True:
            end_pose = current_node
            col_final = 1
            break
        else:
            current_node = end_pose
    
    if cost>2:
        col_final=1
    
    
    cost_total += cost
    
    return cost_total, np.asarray(left_commands),np.asarray(right_commands),np.asarray(Time),col_final,end_pose
        
        
        

#Utility code for the iterated curve algorithm
def adjust_omega(wh_l,wh_r, omega_dot, w_radius, width):

    tol = 0.07
    delta = 2
    ratio = abs(omega_dot[1,2]/omega_dot[0,2])
    if ratio<=tol:
        good_est = True
        return wh_l,wh_r, good_est
    else:    
        angle_error = np.arctan2(omega_dot[1,2],omega_dot[0,2])
        good_est = False
        adjust = width*angle_error/w_radius*delta
        wh_l_new = wh_l-adjust
        wh_r_new= wh_r +adjust
        
    return(wh_l_new,wh_r_new,good_est)
    

    # Not finished


def create_lie_algebra(phi_dot_l, phi_dot_r, w_radius, width, k, sigma_2):

    omega_dot = np.zeros((k, 3, 3))

    E = np.random.multivariate_normal(
        np.array([0.0, 0.0]), sigma_2*np.identity(2), k)
    wh_dot = np.broadcast_to([phi_dot_l, phi_dot_r], (k, 2))+E

    # Create lie algebra matrix for each wheel velocity pair
    omega_dot[:, 1, 0] = w_radius/width*(wh_dot[:, 1]-wh_dot[:, 0])
    omega_dot[:, 0, 1] = w_radius / width * (wh_dot[:, 0] - wh_dot[:, 1])
    omega_dot[:, 0, 2] = w_radius / 2 * (wh_dot[:, 0] + wh_dot[:, 1])

    return(omega_dot)


def wheel_speed_alt_basic(end_node, start_node, w_radius, width, wheel_speed_max, opt_speed):

    inv_start = start_node.inv()
    transform = inv_start*end_node
    xyt = transform.xyt()
    dir_con = 1
    '''
    if xyt[0] < 0:
        dir_con = -1
        xyt[0] = -xyt[0]
        xyt[2] = -xyt[2]
    else:
        dir_con = 1 
    '''

    if (np.tan(xyt[2]) == 0):
        wh_l = np.zeros((2))
        wh_r = np.zeros((2))
        t = np.zeros((2))
        omega_dot = np.zeros((2, 3, 3))
        [wh_l[0], wh_r[0], t[0], omega_dot[0]] = get_wheel_speeds(
            end_node, start_node, w_radius, width, wheel_speed_max, opt_speed)
        target_list = [end_node, end_node]

    else:
        mx = xyt[0]-xyt[1]/np.tan(xyt[2])
        my = np.sqrt(np.sum(np.asarray([xyt[0]-mx, xyt[1]])**2))

        if abs(mx) <= abs(my):
            [dx, dy] = simple_curve_calc(xyt[2], mx)
            inter_pose = SE2(dx, dy, xyt[2])
        else:
            [dx, dy] = simple_curve_calc(xyt[2], my)
            inter_pose = SE2(xyt[0]-dx, xyt[1]-dy, 0)

        wh_l = np.zeros((2))
        wh_r = np.zeros((2))
        t = np.zeros((2))
        omega_dot = np.zeros((2, 3, 3))
        [wh_l[0], wh_r[0], t[0], omega_dot[0]] = get_wheel_speeds(
            inter_pose, SE2(), w_radius, width, wheel_speed_max, opt_speed)
        [wh_l[1], wh_r[1], t[1], omega_dot[1]] = get_wheel_speeds(
            transform, inter_pose, w_radius, width, wheel_speed_max, opt_speed)
        target_list = [start_node*inter_pose, end_node]
        

        if any(np.isnan(wh_l)) | any(np.isnan(wh_r)):
            print(mx)
            print(my)
            print(inter_pose)
            print(xyt)

    return dir_con*wh_l, dir_con*wh_r, t, dir_con*omega_dot, target_list

#Utility code for getting information need to construct the simple curve
def simple_curve_calc(theta, d):

    R = d/np.tan(theta/2)
    dx = R*np.sin(theta)
    dy = R*(1-np.cos(theta))

    return [dx, dy]

#The path cost function
def local_cost_calc(occ_grid, omega_dot, T, start_node, target_node, step_num, col_penalty, map_size):

    x_min = map_size[0, 0]
    x_max = map_size[0, 1]
    y_min = map_size[1, 0]
    y_max = map_size[1, 1]
    t = np.linspace(0, T, step_num)
    x = np.linspace(x_min, x_max, len(occ_grid))
    y = np.linspace(y_min, y_max, len(occ_grid[0]))
    P = []
    for i in range(len(t)):
        P.append(start_node*SE2().Exp(omega_dot*t[i]))

    col = False
    end_point = P[-1]
    
    cost_path = abs(T)

    for p in range(len(P)):
        # Reverse indices for matrix indexing vs plotting
        out = P[p].t
        i = out[1]
        j = out[0]
        i = np.argmin(abs(x-i))
        j = np.argmin(abs(y-j))
        #i = P[p][0][1]
        #j = P[p][0][0]
        if (occ_grid[i, j] == 1) | (out[0] < x_min) | (out[0] > x_max) | (out[1] < y_min) | (out[1] > y_max):
            end_point = P[p-1]
            col = True
            break
        else:
            cost_path += abs(T)/step_num*occ_grid[i,j]*10
        
    if omega_dot[0,2]<0:
        cost_path *=1.1        

    cost = error_pose(target_node, end_point)
    

    if col == True:
        cost = cost+col_penalty

    

    return(cost, end_point, col,cost_path)

#The different error and distance functions
def dist_calc(A, B):
    #difx = A[0,2] - B[0,2]
    #dify = A[1,2] - B[1,2]
    #dif = np.asarray([difx,dify])
    dif = A.t - B.t
    dist = np.sqrt(np.sum(dif**2))
    return(dist)


def dist_2_calc(A, B):
    dif = np.asarray(A)-np.asarray(B)
    dist_2 = np.sum(dif[0:2, 2]**2)
    return(dist_2)


def dist_node_calc(graph,node1, node2):
    A = graph.nodes[node1]['pose']
    B = graph.nodes[node2]['pose']
    dis = dist_calc(A, B)

    return(dis)


def error_pose(A, B):
    dif = A-B
    error = np.sum(np.sum(dif**2))
    return(error)

# function to find optimal path through graph
def A_star_search(graph,s,g,weight):
    h=dist_node_calc
    #Initialize Dictionaries
    CostTo = {}
    EstTotalCost = {}
    pred = {}
    for v in list(graph.nodes):
        CostTo[v] = float('inf')
        EstTotalCost[v] = float('inf')

    CostTo[s] = 0
    EstTotalCost[s] = h(graph,s,g)
    #Used Heapdict instead of heapq because it is difficult to update priority in heapq
    Q= hd.heapdict()

    Q[s] = h(graph,s,g)

    #While Q is not empty, keep looping
    while Q != []:
        #Get val from que
        [v, est] = Q.popitem()
        #If you reach goal return path from start to goal
        if v == g:
            return Recover_NodePath(s,g,pred)

        #For all neighbors of vertex v, if going to neighbor i through v is shorter than previous path to i, update
        #All dictionaries and que
        for i in neighbors(graph,v):
            pvi = CostTo[v] + graph.edges[(v,i)][weight]
            if pvi < CostTo[i]:
                pred[i] = v
                CostTo[i] = pvi
                EstTotalCost[i] = pvi + h(graph,i,g)
                #Regardless of the condiitonal in the psuedocode, this will correctly add/update vertex i in the queue
                Q[i] = EstTotalCost[i]



    return set()




def neighbors(graph,current_node):
    neighbors = set()

    #I just identified adjacent squares in the 'grid' as neighbors
    for n in list(graph.out_edges(current_node)):
        neighbors.add(n[1])
        
    return neighbors

def Recover_NodePath(s,g,pred):
    j = g
    path = [s]
    while s != pred[j]:
        path.insert(1,pred[j])
        j = pred[j]
        
    path.append(g)

    return path

#Not implemented - look at A* search
def opt_path_search_broad(occ_grid, graph, start_node, goal_node, pathing_alg, d_max, cost_max, w_radius, width, max_wheel, opt_speed, particle_num, sigma_2, step_num, col_penalty, col_percent_max):

    queue = hq.heapify([])
    best_next = {}  # Dictionary of best next node in path
    total_cost = {}  # Dictionary of lowest total cost to goal

    graph_size = len(graph)
    priority = 0
    start_num = graph_size
    goal_num = graph_size+1

    add_to_PRM(goal, goal_num, occ_grid, graph, d_max, cost_max, w_radius, width,
               max_wheel, opt_speed, particle_num, sigma_2, step_num, col_penalty, col_percent_max)
    add_to_PRM(start, start_num, occ_grid, graph, d_max, cost_max, w_radius, width,
               max_wheel, opt_speed, particle_num, sigma_2, step_num, col_penalty, col_percent_max)

    hq.heappush(queue, goal)
    searched = [goal_num]
    total_cost[goal_num] = 0

    while len(queue) > 0:
        current_node = hq.heappop(queue)

        for u, v in graph.edges(current_node):
            if v not in searched:
                searched.append(v)
                hq.heappush(queue, v)
                best_next[v] = total_cost[current_node] + \
                    graph.edges[u, v]['weight']

    return()

#Functions for testing different function and code
def test_interp():

    x = SE2()

    y = SE2(5,3,math.pi/2)
    #y = s.SE2.Tx(3)*y
    #y = s.SE2.Ty(3)*y

    #x = SE2.Rand()
    #y = SE2.Rand()

    pose_array = x.interp(y, 10)

    # pose_array.plot()
    for i in range(len(pose_array)):
        pose_array[i].plot(frame='A')

    # plt.plot(pose_array[:][0][2],pose_array[:][1][2])
    a = pose_array[2]
    b = pose_array[3]

    return(x, y, a.inv()*b.log())


def test_logm():

    ytest = s.SE2.Rot(math.pi/2)
    ytest_np = np.zeros((3, 3))
    ytest_np[:][:] = ytest[:][:]
    ytest_np[0][2] = 1.8
    ytest_np[1][2] = 2

    t = np.linspace(0, 1, 20)

    #lie_alg = ytest.log()
    lie_alg = sm.transforms2d.trlog2(ytest_np)
    lie_alg_alt = copy.deepcopy(lie_alg)
    lie_alg_alt[1, 2] = 0
    lie_alg_alt[1, 0] += 0
    lie_alg_alt[0, 1] -= 0
    pose_array = []
    pose_array_alt = []

    for i in range(len(t)):
        pose_array.append(s.SE2.Exp(lie_alg*t[i]))
        pose_array_alt.append(s.SE2.Exp(lie_alg_alt*t[i]))
        pose_array[i].plot(frame='A', figure=1)
        pose_array_alt[i].plot(frame='B', figure=2)

    return(lie_alg, ytest_np)


def test_pathing(graph, occ_grid, step_num, w_radius, width, sigma_2):
    sample_nodes = np.random.choice(graph.nodes, 2, replace=False)
    start = sample_nodes[0]
    goal = sample_nodes[1]

    node_path = A_star_search(graph, start, goal, weight='weight')

    [path, weight, time, wh_speed,node_loc] = RecoverPath_iter(graph, node_path)
    #wh_speed = np.transpose(np.asarray(wh_speed).reshape((2,len(time))))

    ti = path[0].t
    tf = path[-1].t

    fig, ax = plt.subplots()
    im = ax.imshow(occ_grid)
    ax.scatter(ti[0], ti[1], color='blue')
    ax.scatter(tf[0], tf[1], color='magenta')
    current_pose_true = np.asarray(path[0])

    for i in range(len(time)):
        t = np.linspace(0, time[i], step_num)
        x = np.zeros((len(t), 1))
        y = np.zeros((len(t), 1))
        x_true = np.zeros((len(t), 1))
        y_true = np.zeros((len(t), 1))
        #current_pose = path[i]
        #x[0] = current_pose.t[0]
        #y[0] = current_pose.t[1]
        x_true[0] = current_pose_true[0, 2]
        y_true[0] = current_pose_true[1, 2]
        #current_pose = np.asarray(current_pose)
        current_pose_true = np.asarray(current_pose_true)
        for k_step in range(step_num-1):
            #current_pose = PF_prop(t[k_step],t[k_step+1],current_pose,wh_speed[i],w_radius,width,sigma_2)
            #x[k_step+1] = current_pose[0,2]
            #y[k_step+1] = current_pose[1,2]
            current_pose_true = PF_prop(
                t[k_step], t[k_step+1], current_pose_true, wh_speed[i], w_radius, width, sigma_2)
            x_true[k_step+1] = current_pose_true[0, 2]
            y_true[k_step+1] = current_pose_true[1, 2]

        # ax.plot(x,y,color='red')
        ax.plot(x_true, y_true, color='green')

    plt.show()


def test_spin():
    new_node = SE2.Rand()
    start_node = SE2.Rand()
    # new_node.plot(color='magenta')
    # start_node.plot(color='green')
    w_radius = 1
    width = 2
    max_wheel = math.pi*16
    opt_speed = 4
    commands, poses = wheel_speed_spin_move(
        new_node, start_node, w_radius, width, max_wheel, opt_speed)

    i = 0
    label = ['1', '2', '3', 'G']
    ini_node = SE2([0, 0, 0])
    ini_node.plot(color='green')
    for p in poses:
        p.plot(color='red', frame=label[i])
        i += 1

    plt.show()


def basic_path(graph, occ_grid, step_num, w_radius, width, wheel_speed_max, opt_speed, sigma_2, k):

    x = SE2()
    y = SE2(5, -3, math.pi/2)
    step_num = 10
    col_penalty = 100
    map_size = np.array([[0,100],[0,100]])

    #[wh_l, wh_r, T, omega, target] = wheel_speed_alt_basic(
    #    y, x, w_radius, width, wheel_speed_max, opt_speed)
    #[cost,wh_l, wh_r, T, col,target] = iterate_curve(occ_grid,
    #     y, x, w_radius, width, wheel_speed_max, opt_speed,k,sigma_2,step_num,col_penalty,map_size)
    [cost,wh_l, wh_r, T, col,target] = est_curve_alt(occ_grid,
         y, x, w_radius, width, wheel_speed_max, opt_speed,k,sigma_2,step_num,col_penalty,map_size)
    current_pose = x
    current_pose.plot(frame='X',color='magenta')
    y.plot(frame='T',color = 'red')
    for i in range(len(wh_l)):
        wheel_speed = [wh_l[i], wh_r[i]]
        omega_dot = create_lie_algebra(
            wh_l[i], wh_r[i], w_radius, width, k, sigma_2)

        #current_pose = PF_prop(0, T[i], current_pose, [wheel_speed], w_radius, width, sigma_2)

        print(omega_dot)
        current_pose = current_pose*SE2.Exp(omega_dot[0]*T[i])
        current_pose.plot(frame='R')

    print(error_pose(current_pose, y))