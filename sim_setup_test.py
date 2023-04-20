# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:28:53 2023

@author: rfsyl
"""

import pybullet as p
import time
import pybullet_data
import os
import urdf_parser_py

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
p.setAdditionalSearchPath(os.path.join(os.getcwd(),"urdf"))
boxId = p.loadURDF("rover.urdf",startPos, startOrientation)
#boxId = p.loadURDF("r2d2",startPos, startOrientation)
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
