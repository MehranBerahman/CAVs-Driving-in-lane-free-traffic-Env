# -*- coding: utf-8 -*-
"""
Created on Tue May 30 13:47:59 2023

@author: mehran
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 08:45:58 2022

@author: mehran
"""

import os
import sys
from laneFreeUtils3_ import *
import traci
import sumolib
import traci.constants as tc
#from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from collections import namedtuple
import pandas as pd
import torch
import random
import gymnasium as gym
#import gym
from gymnasium.spaces import Box, Discrete, Tuple, Dict
import torch.nn.functional as F
import math
import copy
sumoCfgFile = r"D:\Traffic Research Topics\LaneFreeEnv\circle/circle_3.sumocfg"
#sumoCfgFile = r"D:\Traffic Research Topics\LaneFreeEnv\circle/circle_5.sumocfg"




class laneFreeMagent_CirEnv(gym.Env):
    """

   A template to implement custom OpenAI Gym environments

    """

    metadata = {'render.modes': ['human']}

    def __init__(self, sumoCfgFile=sumoCfgFile, agentNum=6, vehWidth=1.8, vehLength=3.2,
                 out_csv_name=None, use_gui=False, maxTimeSim=1000, edgeID="e1", Init_minMaxSpeed=[5, 20], Desired_minMaxSpeed=[10, 35],
                 longLatCollisionDis=[.4, .2], initPos0=100, max_step=1000, speedRewardCoef=1, Nudge_ind=1,
                 longLatSpeedRewardCoef=[.4, .4], print_initial_statuse=False, maxLongLatSpeedAcc=[5, 1.5], n_intrusion=.15, wrongLatActPun=-1, discelTimeCoeff=1, reqDisLen=4,
                 MaxSpeedRiseCoef=1.1, single_agent=False, seed=0, NudgeCoef=.3, intrusionStressCoef=1, MinspeedReward=-.2, VehSignalling=True, aCoef=1.38, tds=.5, posiAccFac=1,
                 jeLatCoef=.7, JeLonCoef=.6, circleLength=400, depSpeed=25, ignorCollision=False, forced_act=False):
        
        self.forced_act = forced_act 
        self.ignorCollision = ignorCollision
        self.__version__ = "0.0.1"
        self.print_initial_statuse = print_initial_statuse
        self.depSpeed = depSpeed
        # Execution of SUMO environment:
        self.sumoCfgFile = sumoCfgFile
        self.use_gui = use_gui
        if self.use_gui:
            sumo_cmd = [sumolib.checkBinary(
                'sumo-gui'), '-c', self.sumoCfgFile]
            sumo_cmd.append('--start')
        else:
            sumo_cmd = [sumolib.checkBinary('sumo'), '-c', self.sumoCfgFile]

        traci.start(sumo_cmd)
        traci.simulationStep()  # it must be the first simulation step
        
        #  *******************************************  #
        self.circleLength = circleLength
        #self.corList = [(119.23,64.91), (118.18,82.11), (111.06,100.53), (95.76,110.73), (78.48,116.36), (65.55,123.68)]
        if circleLength == 400:
            # for cicle with the length of 400 meters
            #self.corList = [(13.73,106.46), (23.25,110.44), (53.66,122.51),(109.49,103.90), (35.04,121.94), (71.70,118.82)]
            #self.corList = [(-2.47,61.64), (9.46,104.81), (66.94,118.69), (115.31,92.01), (116.96,25.51), (34.39,-0.18)]
            self.corList = [(-2.47,61.64), (9.46,104.81), (66.94,118.69), (115.31,92.01), (116.96,25.51), (34.39,-0.18)
                            , (0.27,24.95), (63.62,-0.58), (118.46,60.12), (99.74,112.69), (55.86,126.87), (18.42,104.75)
                               , (66.99,-6.92), (-6.24,77.98), (126.16,65.15), (120.35,72.98), (114.49,81.68), (100.68,105.13)
                               , (77.80,121.58) , (43.95,120.63), (29.96,119.31), (3.10,90.94), (-9.38,56.15), (-2.00,41.21)
                               , (6.32,28.16), (8.78,18.89), (20.12,4.14), (84.61,-1.91), (93.69,8.11), (119.59,41.21)
                               #, (14.26,17.38), (37.71,-4.37), (43.57,-0.21), (77.04,1.12), (105.22,12.09)
                               #, (105.41,19.84), (125.46,48.77), (122.81,80.54), (86.31,114.96)
                               #, (4.24,86.22), (-5.78,70.71)
                                ]
            self.xy = (122.87,58.62)
            self.length = 199.11 + 202.49 +.25 + .24
        
        if circleLength == 1000:
            # for cicle with the length of 1000 meters
            self.corList = [(319.55,161.85), (302.61,244.51), (170.42,320.48), (4.16,222.25), (50.17,37.73), (255.64,25.44)]
            self.xy = (323.41,158.79)
            self.length = 516.25 + 514.96 + .51 + .34
        
        self.l_routh_1 = 3*self.length / 4
        self.l_routh_1_1 = 7*self.length / 8
        self.l_routh_3_2 = 3*self.length / 8
        self.l_routh_4 = 1*self.length / 4
        self.l_routh_2 = self.length / 2
        #self.vehsInfo = {}
        self.n_intrusion = n_intrusion # acceptable intrusion to consider jerk in reward function
        self.N_ind = Nudge_ind
        self.laneID = edgeID + "_" + str(traci.edge.getLaneNumber(edgeID) - 1)
        self.laneWidth = traci.lane.getWidth(self.laneID)
        self.half_Freedom = .5 * (self.laneWidth - vehWidth)
        self.target_speed =  {}
        self.midAccel = .7*maxLongLatSpeedAcc[0]
        self.wrongLatActPun = wrongLatActPun
        self.discelTimeCoeff = discelTimeCoeff
        self.reqDisLen = reqDisLen
        
        #   *****************************************  #
        self.jeLatCoef, self.JeLonCoef = jeLatCoef, JeLonCoef
        self.VehSignalling = VehSignalling
        self.posiAccFac = posiAccFac
        # Defining environment's parameters and their values:
        self.Info = namedtuple("env_Info", field_names=[
                               "CurrentStates", "NextStates", "actions", "rewards", "dones", "env_information"])
        # VAR_SPEED = 64 ,VAR_LANEPOSITION_LAT = 184, VAR_LANEPOSITION = 86, VAR_MAXSPEED = 65
        traci.edge.subscribeContext(edgeID, tc.CMD_GET_VEHICLE_VARIABLE,1000, [86,184,64,50,114,66, 65])
        
        #self.minMaxLongDist = minMaxLongDist
        traci.simulationStep()  # it must be the first simulation step
        self.timeStep = traci.simulation.getDeltaT()
        #self.timeStep = 1
        self.MinspeedReward = MinspeedReward
        self.aCoef = aCoef
        self.agentNum = agentNum
        self.minMaxSpeed = Init_minMaxSpeed
        self.DesiredSpeed = Desired_minMaxSpeed
        self.vehLength = vehLength
        self.vehWidth = vehWidth
        self.tds = tds
        self.old_Seed = seed
        self.max_stepNum = max_step
        self.maxACC_Speed = maxLongLatSpeedAcc
        self.maxLongLatSpeedAcc = self.timeStep * np.array(maxLongLatSpeedAcc)
        self.speedRewardCoef = speedRewardCoef
        self.longLatSpeedRewardCoef = longLatSpeedRewardCoef
        self.latNudgeCoef, self.longNudgeCoef = .2, .2
        self.NudgeCoef = NudgeCoef
        self.intrusionStressCoef = intrusionStressCoef
        self.MaxSpeedRiseCoef = MaxSpeedRiseCoef
        self.desiredSpeed = []

        self.maxLatChange = self.laneWidth / 2 - vehWidth/2
        ##self.maxLongChangeR, self.maxLongChangeF = maxLongChange
        #self.intermedSpeed = int(np.mean(minMaxSpeed))
        #self.max_speedDiff = minMaxSpeed[1] - minMaxSpeed[0]
        self.initVehiCoord = traci.lane.getShape(self.laneID)[0]
        # initXEgoVeh = initVehiCoord[0] + 100  # Initial Longitudinal Ego Vehicle Position
        # initYEgoVeh = initVehiCoord[1] + initLatPosEgo  # Initial Lateral Ego Vehicle Position
        self.iniPos0 = initPos0  # Initial longitudinal Position of veh_0
        self.edgeID = edgeID
        self.rearVehNum = (agentNum-1) // 2
        self.frontVehNum = self.rearVehNum if agentNum % 2 != 0 else self.rearVehNum+1
        self.MaxSpeedDev = MaxSpeedRiseCoef - 1
        # observation Space: [actual_speed, deviation_from_desired_speed, lateral_position, (agentNum-1)*RepulsiveStress,
        #(agentNum-1)*NudgeStress,(agentNum-1)*lateral_Diffrence, (agentNum-1)*longitudinal_diffrence,
        # (agentNum-1)*speed_Diffrence], according to above description, each agent observation has 28 element
        maxStr = self.NudgeCoef  # Maximum intrusion stress
        self.observation_space = Dict({f"agent_{i}": Box(low=np.array([0, -1, -self.maxACC_Speed[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2*self.maxLatChange, -2*self.maxLatChange, -2*self.maxLatChange,
                                                                       -2*self.maxLatChange, -2*self.maxLatChange, -180, -270, -320, -320, -330, -45, -45, -45, -45, -45]),
                                                         high=np.array([51, 45, self.maxLatChange, maxStr, maxStr, maxStr, maxStr, maxStr, maxStr, maxStr, maxStr, maxStr, maxStr,
                                                                        2*self.maxLatChange, 2*self.maxLatChange, 2*self.maxLatChange,
                                                                        2*self.maxLatChange, 2*self.maxLatChange, 320, 310, 310, 270, 190, 50, 50, 50, 50, 50]))
                                       for i in range(self.agentNum)})

        """
        self.observation_space = [Box(low=np.array([20,-self.maxLatChange,-self.maxLatChange,-20,-10,-self.maxLatChange,-self.maxLatChange,-self.maxLatChange, -self.maxLatChange,
                                                    -self.maxLatChange, -200, -200, -200, -200, -200]),
                                      high=np.array([38,self.maxLatChange,self.maxLatChange,20,10,self.maxLatChange,self.maxLatChange,self.maxLatChange,self.maxLatChange
                                                     ,self.maxLatChange,200,200,200,200,200]),
                                      dtype=np.float32) for i in range(self.agentNum)]
        """

        # Action : lateral change and longitudinal acceleration
        SpAct, latAct = maxLongLatSpeedAcc
        self.action_space = Dict({f"agent_{i}": Box(low=np.array([-SpAct, -latAct]), high=np.array([SpAct, latAct]))
                                  for i in range(self.agentNum)})
        """
        self.action_space =[Box(low=np.array([-latAct,-SpAct]),high=np.array([latAct,SpAct]),dtype=np.float32) 
                           for _ in range(self.agentNum)]
        """

        """

        self.observation_space = (Box(low=np.array([0,0,0,0]),
                                     high=np.array([20,15,30,40]), dtype=np.float32),
                                  Box(low=np.array([0,0,0,0]),
                                                               high=np.array([20,15,30,40]), dtype=np.float32),
                                  Box(low=np.array([0,0,0,0]),
                                                               high=np.array([20,15,30,40]), dtype=np.float32))
                                  
                            
                                       
        #action_space: sublane change, speed change
        self.action_space =(Box(low=np.array([-latAct,-SpAct]),high=np.array([latAct,SpAct]),dtype=np.float32),
                            Box(low=np.array([-latAct,-SpAct]),high=np.array([latAct,SpAct]),dtype=np.float32),
                            Box(low=np.array([-latAct,-SpAct]),high=np.array([latAct,SpAct]),dtype=np.float32))
        """

    # Distance between two consequence vehicles having overlap
    
    def addVehFunc(self):
        speedList = np.random.choice(np.arange(self.DesiredSpeed[0],self.DesiredSpeed[1]), self.agentNum, replace=True)
        #initialSPeedList = np.random.choice(np.arange(self.minMaxSpeed[0],self.minMaxSpeed[1]), 6, replace=False)
        self.minMaxSpeed
        self.vehDict={}
        self.vehsInfo = {}
        depRef = random.randint(-10,0)
        for i in range(self.agentNum):
            veh_name = "veh" + "_" + str(i) + "_" + str(speedList[i])
            maxSpeed = self.MaxSpeedRiseCoef * speedList[i]
            depSpeed = np.clip(speedList[i] + random.randint(depRef-3, depRef+5), 2, maxSpeed)
            
            #traci.vehicle.add(veh_name, routeID="route0", typeID="type2",departLane="best", departSpeed=depSpeed)
            traci.vehicle.add(veh_name, routeID="route0", typeID="type2",departLane="best", departSpeed=self.depSpeed)

            traci.vehicle.moveToXY(veh_name, self.edgeID, 1, self.corList[i][0], self.corList[i][1])
            traci.vehicle.setMaxSpeed(veh_name, maxSpeed)
            traci.vehicle.setLaneChangeMode(veh_name, 0b000000000000)
            traci.vehicle.setSpeedMode(veh_name, 0)
            traci.vehicle.setSignals(veh_name, 0)
            self.vehsInfo[veh_name] = {"leadVehInfo":{}, "followVehInfo":{},
                                  "dis2ref":0,"primPred":"None", "desiredSpeed":speedList[i],
                                  "speedRewards":[0,0]}
            #self.vehsInfo[veh_name] = {"al":0, "aaaa":[]}
            
        #traci.simulationStep()   
   
    def scanFunc(self):
        #global vehsInfo, length, l_routh_2,l_routh_1_1, l_routh_3_2, vList, xy, tdict
        #VAR_SPEED = 64, VAR_LANEPOSITION = 86 ,VAR_LANEPOSITION_LAT = 184, VAR_SPEED_LAT = 50, POSITION_2D  = 1
        # VAR_ROAD_ID = 80, GET_LANE_VARIABLE = 81, VAR_ACCELERATION = 114 
        #["LANEPOSITION", "LANEPOSITION_LAT", "speed", "latSpeed", "accel", "position_2D"]
        self.tdict = traci.edge.getContextSubscriptionResults(self.edgeID)
        #self.vList = list(self.tdict.keys())
        #n_self.vList = copy.deepcopy(self.vList)
        for veh in self.vList:
            dist = self.length - traci.vehicle.getDrivingDistance2D(veh, self.xy[0], self.xy[1])
            self.vehsInfo[veh]["leadVehInfo"] = {}
            self.vehsInfo[veh]["followVehInfo"] = {}
            self.vehsInfo[veh]["dis2ref"] = dist
            self.vehsInfo[veh]["devFromDesiredSpeed"] = (self.vehsInfo[veh]["desiredSpeed"] - self.tdict[veh][64])/self.vehsInfo[veh]["desiredSpeed"]
            self.vehsInfo[veh]["l_rLatFreedom"] = [self.half_Freedom - self.tdict[veh][184],
                                                   self.half_Freedom + self.tdict[veh][184]]
            self.vehsInfo[veh]["prim_lead&fol_veh"] = ["", ""]

            """
            {"leadVehInfo":{}, "followVehInfo":{},
                                  "dis2ref":traci.vehicle.getDrivingDistance2D(veh, self.xy[0], self.xy[1]),
                             "vehObserv":[False, {"Nudge": [[], [], []], "Repulsion":[
                                      [], []]}, {"aLon_n": [[0], [None]]}, 0, 0],
                             "primPred":"None"}
            """
                          
        vNum = len(self.vList)
        for i in range(vNum):
            #n_vList = copy.deepcopy(vList)
            veh = self.vList[i]
            #n_vList.remove(veh)
            dist_v = self.vehsInfo[veh]["dis2ref"] 

            #x, y = self.tdict[veh][66]
            for j in range(i+1, vNum):
                n_veh = self.vList[j]
                neigh_v = "leadVeh"
                dist_nV = self.vehsInfo[n_veh]["dis2ref"]
                interVehDist = dist_nV - dist_v
                if interVehDist < 0: interVehDist = dist_nV + self.length - dist_v 
                if interVehDist > self.l_routh_2 :neigh_v = "followVeh"
                
                if interVehDist > self.l_routh_2: interVehDist = self.length - interVehDist
                
                """
                if dist_v > self.l_routh_1_1 and dist_nV < self.l_routh_3_2: neigh_v = "followVeh"
                elif dist_v < self.l_routh_3_2 and dist_nV > self.l_routh_1_1: neigh_v = "leadVeh"
                elif dist_v > self.l_routh_1 and dist_nV < self.l_routh_4: neigh_v = "followVeh"
                elif dist_v < self.l_routh_4 and dist_nV > self.l_routh_1: neigh_v = "leadVeh"
                else: 
                    neigh_v = "leadVeh" if dist_v - dist_nV >= 0 else "followVeh"
                interVehDist = abs(dist_v - dist_nV)
                if interVehDist > self.l_routh_2: interVehDist = self.length - interVehDist
                """
                latDist = self.tdict[veh][184] - self.tdict[n_veh][184]
                speedDif = self.tdict[veh][64] - self.tdict[n_veh][64]
                latSpeedDiff = self.tdict[veh][50] - self.tdict[n_veh][50]
                accelDif = self.tdict[veh][114] - self.tdict[n_veh][114]
                #      *************************    #
                
                if (neigh_v == "leadVeh" and speedDif > 0) or (neigh_v == "followVeh" and speedDif < 0):
                    SpeedDIFF = abs(speedDif)
                    required_time = SpeedDIFF / self.midAccel
                    required_dis = -.5 * self.midAccel * required_time**2 + SpeedDIFF * required_time
                    safeDist = 2 + required_dis
                else: safeDist = 2
                    
                
                if abs(interVehDist) <= self.vehLength + safeDist:
                    freedome = max(0, abs(latDist) - self.vehWidth - 1.5)
                    if latDist > 0: # n_veh is in the right side of veh
                        
                        if freedome < self.vehsInfo[veh]["l_rLatFreedom"][1]: 
                            self.vehsInfo[veh]["l_rLatFreedom"][1] = freedome
                        
                        #veh is in the left side of n_veh    
                        if freedome < self.vehsInfo[n_veh]["l_rLatFreedom"][0]:
                           self.vehsInfo[n_veh]["l_rLatFreedom"][0] = freedome
                    
                    else:# n_veh is in the left side of veh
                        if freedome < self.vehsInfo[veh]["l_rLatFreedom"][0]: 
                            self.vehsInfo[veh]["l_rLatFreedom"][0] = freedome
                        
                        # veh is in the left side of n_veh 
                        if freedome < self.vehsInfo[n_veh]["l_rLatFreedom"][1]:
                           self.vehsInfo[n_veh]["l_rLatFreedom"][1] = freedome
                
                
                if neigh_v == "followVeh":
                    # ["longDist", "latDist", "speedDif", "latSpeedDiff","accelDif", "nudgeList"]
                    self.vehsInfo[veh]["followVehInfo"][n_veh] = []
                    self.vehsInfo[n_veh]["leadVehInfo"][veh] = []

                    self.vehsInfo[veh]["followVehInfo"][n_veh].append(interVehDist)
                    self.vehsInfo[n_veh]["leadVehInfo"][veh].append(interVehDist)

                    #latDist = self.tdict[veh][184] - self.tdict[n_veh][184]
                    self.vehsInfo[veh]["followVehInfo"][n_veh].append(latDist)
                    self.vehsInfo[n_veh]["leadVehInfo"][veh].append(-latDist)
                    
                            
                    #speedDif = self.tdict[veh][64] - self.tdict[n_veh][64]
                    self.vehsInfo[veh]["followVehInfo"][n_veh].append(speedDif)
                    self.vehsInfo[n_veh]["leadVehInfo"][veh].append(-speedDif)
                    
                    #latSpeedDiff = self.tdict[veh][50] - self.tdict[n_veh][50]
                    self.vehsInfo[veh]["followVehInfo"][n_veh].append(latSpeedDiff)
                    self.vehsInfo[n_veh]["leadVehInfo"][veh].append(-latSpeedDiff)
                    
                    #accelDif = self.tdict[veh][114] - self.tdict[n_veh][114]
                    self.vehsInfo[veh]["followVehInfo"][n_veh].append(accelDif)
                    self.vehsInfo[n_veh]["leadVehInfo"][veh].append(-accelDif)
                    
                    self.vehsInfo[veh]["followVehInfo"][n_veh].append([0,0])# [nudge, accSpeedNudge]
                    self.vehsInfo[n_veh]["leadVehInfo"][veh].append(0)# repusion

                else:
                    # ["longDist", "latDist", "speedDif", "latSpeedDiff","accelDif", "repulsion"]
                    self.vehsInfo[veh]["leadVehInfo"][n_veh] = []
                    self.vehsInfo[n_veh]["followVehInfo"][veh] = []

                    self.vehsInfo[veh]["leadVehInfo"][n_veh].append(interVehDist)
                    self.vehsInfo[n_veh]["followVehInfo"][veh].append(interVehDist)
                    
                    #latDist = self.tdict[veh][184] - self.tdict[n_veh][184]
                    self.vehsInfo[veh]["leadVehInfo"][n_veh].append(latDist)
                    self.vehsInfo[n_veh]["followVehInfo"][veh].append(-latDist)
                    
                    #speedDif = self.tdict[veh][64] - self.tdict[n_veh][64]
                    self.vehsInfo[veh]["leadVehInfo"][n_veh].append(speedDif)
                    self.vehsInfo[n_veh]["followVehInfo"][veh].append(-speedDif)
                    
                    #latSpeedDiff = self.tdict[veh][50] - self.tdict[n_veh][50]
                    self.vehsInfo[veh]["leadVehInfo"][n_veh].append(latSpeedDiff)
                    self.vehsInfo[n_veh]["followVehInfo"][veh].append(-latSpeedDiff)
                    
                    #accelDif = self.tdict[veh][114] - self.tdict[n_veh][114]
                    self.vehsInfo[veh]["leadVehInfo"][n_veh].append(accelDif)
                    self.vehsInfo[n_veh]["followVehInfo"][veh].append(-accelDif)
                    
                    self.vehsInfo[veh]["leadVehInfo"][n_veh].append(0) #repusion
                    self.vehsInfo[n_veh]["followVehInfo"][veh].append([0,0])#[nudge, accSpeedNudge]
    
    def findAgents(self):
        self.agentList = self.agentNum * [""]
        for veh in self.vList:
            index = len(self.vehsInfo[veh]["followVehInfo"].keys())
            self.agentList[index] = veh
            
    def findPrimPred(self):
        for ind, veh in enumerate(self.agentList):
            for n_ind in range(ind+1, self.agentNum):
                n_veh = self.agentList[n_ind]
                if abs(self.tdict[veh][184] - self.tdict[n_veh][184]) <= (self.vehWidth + .2):self.vehsInfo[veh]["primPred"] = n_veh
                    
    

                    
        
    def R_disCalculation(self, L_velocity, F_velocity):
        """
        minVelocity = min(L_velocity, F_velocity)    
        d0 = 5 if minVelocity >= 10 else -0.125 * minVelocity + 6.25+3
        b = d0 + self.tds * minVelocity - .5*self.vehLength #as b is distance from central point of following vehicle
        delta_v = abs(L_velocity - F_velocity)
        required_time = delta_v / self.maxACC_Speed[0]
        required_dis = -.5 * self.maxACC_Speed[0] * required_time**2 + delta_v * required_time
        required_dis += b
        """
        delta_v = abs(L_velocity - F_velocity)
        required_time = delta_v / (.7*self.maxACC_Speed[0])
        required_dis = -.5 * \
            (.7*self.maxACC_Speed[0]) * \
            required_time**2 + delta_v * required_time
        required_dis += (self.vehLength + 2)
        return math.ceil(required_dis)
    ######################       Reset Function       #########################

    def reset(self, seed=0):
        if self.old_Seed != seed:
            random.seed(seed)
            self.old_Seed = seed
        #### variables that must be reset after calling rest function ######
        self.elapsedTime = False
        self.done, self.doneCause = self.agentNum*[0], ""
        self.stepNum = 0
        self.target_speed = {}
        ##################################################################
        VehIdList = traci.vehicle.getIDList()  # list all lane's vehicles
        # remove all current vehicles from the lane
        [traci.vehicle.remove(vehID) for vehID in VehIdList]
        # vehDict = {"veh_num":[[vehSpeed,vDesirSpeed, vDesiredSpeedCoef], vehLatPos, vehLongPos, MinLongDist, latOverlapVeh[0]], ....}
        #self.vehDict = {}
        
        self.addVehFunc()
        ###################################################################

        traci.simulationStep()
        self.tdict = traci.edge.getContextSubscriptionResults(self.edgeID)
        self.vList = list(self.tdict.keys())
        self.agentList = self.vList
        self.scanFunc()
        self.jerkDict_old = {}
        self.jerkScan = {}
        self.speedDevScan = {}
        self.desSpeedCoef = {}
        self.reqDist = {}
        for vehID in self.vList:
            self.jerkDict_old[vehID] = [0, 0]
            self.speedDevScan[vehID] = []
            self.jerkScan[vehID] = {"jerkLong":[], "jerkLat":[]}
            self.desSpeedCoef[vehID] = 1
            self.reqDist[vehID] = deque(maxlen=self.reqDisLen)
        self.old_tdict = copy.deepcopy(self.tdict)
        
        self._UpdateObs()
        ###################################################################

        
        self.pre_t_Observation = copy.deepcopy(self.t_Observation)
        currentStates = [self.t_Observation[vehID] for vehID in self.agentList]
        dictInfo = {"CurrentAgentList": self.agentList}
        envInformation = self.Info(currentStates, [], [], [], [], dictInfo)
        #envInformation = self.Info(self.t_Observation, "action", "reward", "done","it's the initial state")
        return envInformation
        # return LaneData
    ##################################  step function  #########################################

    def step(self, action):
        # action = np.array(agentNum, actionSize): --> action = agentNum*[lateralChange, speedAcceleration]
        if not any(self.done):
            self.stepNum += 1
            self.pre_t_Observation = copy.deepcopy(self.t_Observation)
            self.agentList_old = copy.deepcopy(self.agentList)
            #self.laneDict_ob_old = copy.deepcopy(self.laneDict_ob)
            #self.old_vehActDict = copy.deepcopy(self.vehActDict)
            #self.jerkDict_old = copy.deepcopy(self.jerkDict)
            self.old_tdict = copy.deepcopy(self.tdict)
            ###############################################################
            self.action = action
            # lateral action execution
            self.lateralAction = self.maxLongLatSpeedAcc[1] * action[:, 1]
            [traci.vehicle.changeSublane(self.agentList_old[index], latAct)
             for index, latAct in enumerate(self.lateralAction)]

            # speed action execution
            #######################################
            self.speedAction = []
            for ind_, act in enumerate(action[:, 0]):
                if act >= 0 :
                    if self.t_Observation[self.agentList_old[ind_]][-2] > .4 and self.forced_act == True: act = -.1
                    self.speedAction.append(self.maxLongLatSpeedAcc[0] * self.posiAccFac * act)
                else: self.speedAction.append(self.maxLongLatSpeedAcc[0] * act)
                
            #self.speedAction = self.maxLongLatSpeedAcc[0] * action[:, 0]
            #######################################
            
            [self.speedControllerFun(self.agentList_old[index], speedAcc)
             for index, speedAcc in enumerate(self.speedAction)]
            env_action = np.array(
                [self.speedAction, self.lateralAction]).transpose()

            ##############################################################
            traci.simulationStep()
            self._UpdateObs()
            # self.Done()

            #Data = (self.t_Observation, self.reward(), self.done, dictInfo)
            next_states = [self.t_Observation[vehID]
                           for vehID in self.agentList_old]
            self.reward()
            reward = [self.rewardDict[vehID] for vehID in self.agentList_old]

            currentStates = [self.t_Observation[vehID]
                             for vehID in self.agentList]
            dictInfo = {"Current_agentList": self.agentList,
                        "AgentList_old": self.agentList_old}
            envInformation = self.Info(
                currentStates, next_states, env_action, reward, self.done, dictInfo)

            #envInformation = self.Info(self.t_Observation, action, self.reward(), self.done, dictInfo)

            return envInformation
        else:
            self.CumReward = 0
            print("Total Step:{}, Cumulative Reward:{}, Done cause:{}".format(
                self.stepNum, self.CumReward, self.doneCause))
            print("Please reset env and then countinue")
            envInformation = self.Info(
                self.t_Observation, "action", "rewards", "done", "Please reset env and then countinue")
    ##################################   reward function  ######################################

    def reward(self):
        self.rewardValues = []
        major_punishment = -10

        self.IntrusionStressDict = {}
        self.longLatSpeedVarDict = {}
        self.rewardDict = {}
        
        """
        if self.collision == True:
            #for veh in self.vehInCollision:
            #self.inv_veh_inCol
            for veh in self.collider:
                self.rewardDict[veh] = major_punishment
                self.IntrusionStressDict[veh] = -1
        """
        
        for index, vehID in enumerate(self.agentList):
            #if vehID not in self.vehInCollision:
            #if vehID not in self.collider:
                #-------------------- Muximum Percentage of Intrusion selection ---------------------#
                # nudge is a combination of speed and intrusion Nudge
                nudgeList = np.abs([self.vehsInfo[vehID]["followVehInfo"][n_veh][-1][self.N_ind] for n_veh in 
                                    self.vehsInfo[vehID]["followVehInfo"].keys()])
                

                #Nudge_stress = max(nudgeList) if len(nudgeList) > 0 else 0
                if len(nudgeList) > 0:
                    Nudge_stress = sum(nudgeList)
                    maxNudge = max(nudgeList)
                else: Nudge_stress, maxNudge = 0, 0

                repList = np.abs([self.vehsInfo[vehID]["leadVehInfo"][n_veh][-1] for n_veh in 
                                    self.vehsInfo[vehID]["leadVehInfo"].keys()])
                #Rep_stress = max(repList) if len(repList) > 0 else 0
                if len(repList) > 0:
                    Rep_stress = sum(repList)
                    maxRep = max(repList)
                else:  Rep_stress, maxRep = 0, 0

                # --------------------------------------------------------------------------------#

                Stress = (maxRep, maxNudge)
                # Punishment reward to reduce long&lat chattering
                lonLatSpeedVarReward = self.speedVarRewardFunc(
                    index, vehID, Stress)
                self.longLatSpeedVarDict[vehID] = lonLatSpeedVarReward
                ####   ****   ####   ####   ****   ####    ####   ****   ####
                #Rep_stress *= -1
                #Nudge_stress *= -1
                #RepFalseActPun, nudgFalseActPun = 0, 0
                # ***************************************************** #
                SPEED_reward = 3 * self.vehsInfo[vehID]["speedRewards"][0]
                ### ******** #########
                intrusion_stress = -maxRep # just consider max_nudge to eleviate the effect of nudge in repulsion
                
                nudgeEffect = max(0, (1  - maxRep/self.n_intrusion))
                intrusion_stress += nudgeEffect * (SPEED_reward - maxNudge)# intrusion_stress and speedReward are always Negative
               
                
                # action Monitoring:
                #  ******  Longitudinal Action   ******
                maxRepOld, maxNudgeOld = self.pre_t_Observation[vehID][-2:]
                nudgeEffectMon = max(0, (1  - maxRepOld/self.n_intrusion))
                accelSigne = np.sign(maxRepOld - nudgeEffectMon * maxNudgeOld)
                
                wrongActLong, wrongActLat, unsuitActLong, unsuitActLat_1, unsuitActLat_2 = False, False, False,\
                    False, False
                
                if accelSigne != 0:
                    if accelSigne * self.action[index,0] >= 0: wrongActLong = True
                        #wrongAct = True, intrusion_stress -= 5 # wrong action
                    elif accelSigne > 0 and maxRepOld > 0.15 and \
                        self.action[index,0] > max(-1, -maxRepOld/.8): unsuitActLong = True
                        #if self.action[index,0] > max(-1, -maxRepOld/.8): intrusion_stress -= 3 # taking unsuitable action
                #  ******  Lateral Action   ******
                totalForce = maxRepOld - maxNudgeOld
                repSignOld = np.sign(totalForce)
                pre_l_rLatFreedom = self.pre_t_Observation[vehID][-4:-2]
                # lateral action must be suitable pertaining to lateral freedom, nudge and rep.
                if totalForce != 0:# totalForce != 0
                    #pre_l_rLatFreedom = self.pre_t_Observation[vehID][-4:-2]
                    if self.action[index,1] * repSignOld < 0: wrongActLat = True
                    #elif self.action[index,1] > max(-1, totalForce/.3) 
                    elif repSignOld > 0: # totalForce > 0
                        if pre_l_rLatFreedom[0] > 0:
                           if not (self.lateralAction[index] > 0 and self.lateralAction[index] <= \
                               pre_l_rLatFreedom[0]): unsuitActLat_1 = True
                           if (pre_l_rLatFreedom[0] >= self.maxLongLatSpeedAcc[1] and self.action[index,0] <= 0 and self.action[index,1] <= abs(self.action[index,0])):
                               unsuitActLat_2 = True # lateral act should be coresponding to long. act
                        elif self.lateralAction[index] != 0: unsuitActLat_1 = True   
                    else:
                        if pre_l_rLatFreedom[1] > 0:
                           if not (self.lateralAction[index] < 0 and abs(self.lateralAction[index]) <= \
                               pre_l_rLatFreedom[1]): unsuitActLat_1 = True
                           elif pre_l_rLatFreedom[1] >= self.maxLongLatSpeedAcc[1] and \
                               self.action[index,1] > max(-1, totalForce/.6): unsuitActLat_1 = True  
                           
                           if (pre_l_rLatFreedom[1] >= self.maxLongLatSpeedAcc[1] and self.action[index,0] >= 0 and self.action[index,1] >= -self.action[index,0] ): 
                               unsuitActLat_2 = True # lateral act should be coresponding to long. act
                                 
                        elif self.lateralAction[index] != 0: unsuitActLat_1 = True 
                        
                if wrongActLong == True: intrusion_stress -= 5
                elif unsuitActLong == True: intrusion_stress -= 2.5
                if wrongActLat == True: intrusion_stress -= 4
                elif unsuitActLat_1 == True or unsuitActLat_2 == True:
                    if unsuitActLat_1 == True: intrusion_stress -= 1.5
                    if unsuitActLat_2 == True: intrusion_stress -= 1.5
                if self.action[index,1] > 0 and pre_l_rLatFreedom[0] < self.maxLongLatSpeedAcc[1] * self.action[index,1]:
                    intrusion_stress -= 5
                if self.action[index,1] < 0 and pre_l_rLatFreedom[1] < abs(self.maxLongLatSpeedAcc[1] * self.action[index,1]):
                    intrusion_stress -= 5 

                self.IntrusionStressDict[vehID] = intrusion_stress

                self.rewardDict[vehID] = intrusion_stress + \
                    lonLatSpeedVarReward

        #return self.rewardDict

    def _UpdateObs(self):
        self.scanFunc()
        #self.findAgents() #define each veh for specific agent self.agentList
        self.findPrimPred() # check if there is front overlap veh
        self.Jerk_latSp_punishment = {} #  = [Var_punishment, Jerk_punishment]
        
        # computing of each throughput vehicles' ellipsoid border and its corresponding a_lat and a_long; afterward adding
        # these two properties in self.vehsInfo[vehID]
        self.aLat_Long_cal(bCoef=1)
        self.Done()
        
        # add speedNudge to nudge dict which is called Acc_speed_nudge
        self.accSpeedNudgeFunc()
        self.agentsObs()

    def Done(self):

        #maxActuLanePos = max(self.actuVehLanePos)
        # Is there any collision in traffic scenario
        self.collision, self.vehInCollision = self.detect_collision()

        if self.collision == True and self.ignorCollision == False:
            self.done, self.doneCause = self.agentNum * \
                [True], 'collision: ' + str(self.vehInCollision)
        if self.stepNum > self.max_stepNum:
            self.done, self.doneCause = self.agentNum * \
                [True], "elapsed Time"  # Time Elapsed

    def render(self):
        pass

    #########################################################################################
    def orderedVeh(self, ascending, vehDict):
        #tdict = traci.edge.getContextSubscriptionResults(edgeID)
        #laneDict = {}
        laneData = pd.DataFrame.from_dict(vehDict, orient='index',
                                          columns=['speed', 'latPos', 'longPos', 'MinLongDist', 'latOverlap'])
        LaneData = laneData.sort_values(by='longPos', ascending=ascending)
        LaneData = LaneData.reset_index()
        LaneData = LaneData.rename(columns={'index': 'veh_name'})
        LaneData = LaneData.to_numpy()
        return LaneData

    def latOverlap(self, vehLatPos, LaneData):
        overlapveh = [False]
        for vData in LaneData:
            if abs(vehLatPos - vData[2]) <= self.vehWidth + .8:
                overlapveh = vData
                break
        return overlapveh
    

    def aLat_Long_cal(self, bCoef=2.5):
        
        # Ellipsoid minpr axis calculation (Lateral axis) -> a
        a = self.aCoef * self.vehWidth
        for index, veh in enumerate(self.agentList):
            # Ellipsoid major axis calculation (longitudinal axis) -> b
            #self.obs_vehData[index, -1][3] = []
            d0 = 5 if self.tdict[veh][64] >= 10 else -0.125 * self.tdict[veh][64] + 6.25
            # d0=0
            # as b is distance from central point of following vehicle
            b = d0 + self.tds * self.tdict[veh][64] - .5*self.vehLength 
            #index = np.where(self.obs_vehData == veh)[0][0]

            # Longitudinal and Lateral position of ego-vehicle
            vehLonLatPos = [0, self.tdict[veh][184]]# longPos is zero as ego.veh is reference long point with respect to its leading neighbor veh
            # *vehLonLatPos = [self.laneDict_ob[veh][2], self.laneDict_ob[veh][1]]
            # for ind in range(index+1, maxIndex):
            for n_veh in list(self.vehsInfo[veh]["leadVehInfo"].keys()):
                # ["longDist", "latDist", "speedDif", "latSpeedDiff","accelDif"]    
                vehDistance = self.vehsInfo[veh]["leadVehInfo"][n_veh][0]
                lonDist_bump2bump = vehDistance - self.vehLength
                delta_v = self.vehsInfo[veh]["leadVehInfo"][n_veh][2]
                inevitableColi = False
                
                if delta_v >= 0:
                    #deltaVPerc = delta_v/(self.tdict[n_veh][64]+ .01)
                    #if deltaVPerc > .1 and delta_v > 5:
                    delta_a = self.tdict[veh][114] - self.tdict[n_veh][114]
                    velocity_coef = [-.5*self.discelTimeCoeff, delta_a, delta_v]
                    T_velocity = max(np.roots(velocity_coef))
                    motion_coef = [-self.discelTimeCoeff/6, delta_a/2, delta_v, 0]

                    motion_equ = np.poly1d(motion_coef)
                    required_dis = motion_equ(T_velocity)

                    #else:
                        #required_time = delta_v / self.midAccel
                        #required_dis = -.5 * self.midAccel * required_time**2 + delta_v * required_time
                    
                    self.reqDist[veh].append(required_dis)
                    R_required_dis = max(required_dis, np.mean(self.reqDist[veh]))+2
                    #required_time = delta_v / self.midAccel
                    #required_dis = -.5 * self.midAccel * required_time**2 + delta_v * required_time
                    d0_ = 5 if self.tdict[n_veh][64] >= 10 else - \
                        0.125 * self.tdict[n_veh][64] + 6.25
                    b1 = d0_ + self.tds * \
                        self.tdict[n_veh][64] - .5*self.vehLength
                    b_ = R_required_dis + b1
                    if R_required_dis >= lonDist_bump2bump: inevitableColi = True
                else:
                    b_ = b
                
                LatDIFF = self.vehsInfo[veh]["leadVehInfo"][n_veh][1]
                
                # Adding relative vehicle lateral displacement(relative_displacement) to a calculation -> a_
                if lonDist_bump2bump < b_:
                    latDiff_old = abs(
                        self.old_tdict[veh][50] - self.old_tdict[n_veh][50])
                    latDiff = abs(LatDIFF)
                    relative_displacement = latDiff - latDiff_old
                    # bumper to bumper long. distance -> leading_veh - ego_veh
                    #lonDist_bump2bump = self.laneDict_ob[veh_n][2] - self.laneDict_ob[veh][2] - self.vehLength
                    longDist = max(0, lonDist_bump2bump)
                    a_ = a - np.sqrt(b_**2 - longDist**2) * \
                        relative_displacement / b_
                    a_ = max(a, a_)
                else:
                    a_ = a
                intrusion_per = 0
                if LatDIFF < 0:
                    if LatDIFF > -(a_+ .1*self.vehWidth):
                        Long_Intr = b_ - lonDist_bump2bump 
                        if Long_Intr > 0:
                            intrusion_per = min(Long_Intr / b_, .9)
                        
                else:
                    # long. & lat. pos leading veh
                    n_longLatPo = [vehDistance,  self.tdict[n_veh][184]]
                    aLat, aLong, q, _, _, distance, distance0, _, _ = accCal(
                        vehLonLatPos, n_longLatPo, a_, b_, self.vehLength, self.vehWidth, self.vehLength, self.vehWidth)
                    
                    if q < 1 :
                        intrusion_per = self.intrusionStressCoef * distance / distance0
                        # if in
                if delta_v >= 0:
                    Nudge = self.NudgeCoef * intrusion_per if inevitableColi == False  else .7*intrusion_per
                    self.vehsInfo[n_veh]["followVehInfo"][veh][-1][0] = Nudge
                    self.vehsInfo[veh]["leadVehInfo"][n_veh][-1]= intrusion_per
                else:
                    #self.vehsInfo[veh]["leadVehInfo"][n_veh][-1] = .05* intrusion_per
                    self.vehsInfo[veh]["leadVehInfo"][n_veh][-1] = .00005* self.NudgeCoef * intrusion_per
                    

    def agentsObs(self):
        self.t_Observation = {} 
        for agent in self.agentList:
            self.t_Observation[agent] = []
            self.t_Observation[agent].append(self.tdict[agent][64]) # VAR_SPEED
            self.t_Observation[agent].append(1.0*self.vehsInfo[agent]["devFromDesiredSpeed"])# deviation of current speed from desired speed (desired speed is different from max_speed)
            self.t_Observation[agent].append(self.tdict[agent][114])# acceleration
            self.t_Observation[agent].append(self.tdict[agent][50]) # Lateral Speed
            
            # Left and right freedom
            self.t_Observation[agent].append(self.vehsInfo[agent]["l_rLatFreedom"][0])# lef Freedom
            self.t_Observation[agent].append(self.vehsInfo[agent]["l_rLatFreedom"][1])# right freedom
            
            
            # Repulsion
            primLead, maxRep = "None", 0
            for n_veh in self.vehsInfo[agent]["leadVehInfo"].keys():
                if self.vehsInfo[agent]["leadVehInfo"][n_veh][-1] > maxRep: primLead, maxRep = n_veh, self.vehsInfo[agent]["leadVehInfo"][n_veh][-1]
            
            self.t_Observation[agent].append(maxRep)
            self.vehsInfo[agent]["prim_lead&fol_veh"][0] = primLead
            
                    
            # nudge
            primFol, maxNudge = "None", 0
            for n_veh in self.vehsInfo[agent]["followVehInfo"].keys():
                if self.vehsInfo[agent]["followVehInfo"][n_veh][-1][self.N_ind] > maxNudge:
                    primFol, maxNudge = n_veh, self.vehsInfo[agent]["followVehInfo"][n_veh][-1][self.N_ind]
            
            self.t_Observation[agent].append(maxNudge)
            self.vehsInfo[agent]["prim_lead&fol_veh"][1] = primFol

    def detect_collision(self):
        collision = False
        self.inv_veh_inCol = []
        #self.colIncident = 0
        self.latLongDiffDict = {}
        coli = traci.simulation.getCollisions()
        self.collider = []
        inv_veh_inColSumo = []
        if len(coli) >= 1:
            collision = True
            for i in range(len(coli)):
                self.collider.append(coli[i].collider)
            self.collider_victim = [coli[0].collider, coli[0].victim]
            [inv_veh_inColSumo.append(vehID) for vehID in self.collider_victim]

        #inv_veh_inCol = set(inv_veh_inCol)
        inv_veh_in_col = []
        [inv_veh_in_col.append(veh)
         for veh in inv_veh_inColSumo if veh not in inv_veh_in_col]

        return [collision, self.inv_veh_inCol]

    def speedControllerFun(self, vehID, speedAcc):
        target_speed = self.old_tdict[vehID][64] + speedAcc
        self.target_speed[vehID] = target_speed
        #target_speed = traci.vehicle.getSpeed(vehID) + speedAcc
        revised_target_speed = np.clip(target_speed, 0,self.vehsInfo[vehID]["desiredSpeed"] * 1.1)  # desired speed can only be increased up to 10%
        # define actual speed for the next step time
        traci.vehicle.setSpeed(vehID, revised_target_speed)
        

    def latChangeControlFun(self, index, latAct):
        traci.vehicle.changeSublane(self.agentList[index], latAct)

    ######################   Reward Functions   ##############################
    def speedRewardFunc(self, vehID, Nudge_stress):
        """
        desSpeedCoef = 1.1 if Nudge_stress else 1
        #if Nudge_stress: self.desSpeedCoef[vehID] = min(1.1, self.desSpeedCoef[vehID] + .02)
        #else: self.desSpeedCoef[vehID] = max(1 , self.desSpeedCoef[vehID] - .02)
        #desSpeed = self.desSpeedCoef[vehID] * self.vehDict[vehID][0][1]
        desSpeed =  desSpeedCoef * self.vehsInfo[vehID]["desiredSpeed"]
        speedDev = (desSpeed - self.tdict[vehID][64]) / desSpeed
        speedNudge = 0
        if speedDev >= 0:
            if speedDev >= 0.01 and speedDev <= .025:
                speedReward = -.05
            elif speedDev > 0.025 and speedDev <= .05:
                speedReward = -.1
            elif speedDev > 0.05 and speedDev <= .1:
                speedReward = -.14
            elif speedDev > 0.1 and speedDev <= .2:
                speedReward = -.18
            elif speedDev > 0.2 and speedDev <= .4:
                speedReward = -.21
            elif speedDev > 0.4 and speedDev <= .7:
                speedReward = -.25
            elif speedDev > 0.7:
                speedReward = -.3
            else:
                speedReward = -5 * speedDev
            speedNudge = abs(speedReward)
        else:
            speedReward = 3 * speedDev
        """ 
        speedDev = (self.vehsInfo[vehID]["desiredSpeed"] - self.tdict[vehID][64]) / self.vehsInfo[vehID]["desiredSpeed"]
        speedReward = -abs(speedDev)
        speedNudge = speedDev if speedDev >= 0 else 0
        if Nudge_stress > 0 : speedNudge = max(speedDev, .05)
        
        self.vehsInfo[vehID]["speedRewards"] = [speedReward, speedNudge]

    def accSpeedNudgeFunc(self):
        self.speedRewardDict = {}
        followingVehs = {}
        for vehID in self.agentList:
            followingVehs[vehID] = list(self.vehsInfo[vehID]["followVehInfo"].keys())
            if vehID not in self.vehInCollision:
                nudgeList = []
                [nudgeList.append(self.vehsInfo[vehID]["followVehInfo"][n_veh][-1][0]) for 
                 n_veh in followingVehs[vehID]]
                
                
                nudgeList = np.abs(nudgeList)
                Nudge_stress = max(nudgeList) if len(nudgeList) > 0 else 0
                #desSpeed = self.vehDict[vehID][0][1]
                # Punishment reward: This function computes speed_reward based on deviation from desired speed
                self.speedRewardFunc(vehID, Nudge_stress)
            else:
                # The worst case condition is considered when collision occureced
                self.vehsInfo[vehID]["speedRewards"] = [-1, 0.3]

        for vehID in self.agentList:
            for n_veh in followingVehs[vehID]:
                speedNudge = self.vehsInfo[n_veh]["speedRewards"][1]
                intrusionNudge = self.vehsInfo[vehID]["followVehInfo"][n_veh][-1][0]
                if self.vehsInfo[vehID]["followVehInfo"][n_veh][2] <= 0 and intrusionNudge != 0:
                    self.vehsInfo[vehID]["followVehInfo"][n_veh][-1][1] = intrusionNudge + \
                        speedNudge / (1 + 2 *intrusionNudge)  # Accummulated Nude
              

    def speedVarRewardFunc(self, index, vehID, Stress):
        #Stress = (maxRep, maxNudge)
        maxStress = max(Stress)
        # Longitudinal speed variation
        
        #speedVarLong = abs(self.pre_t_Observation[vehID][0] - self.t_Observation[vehID][0]) / (self.maxLongLatSpeedAcc[0])
        #speedVarLat = abs(self.pre_t_Observation[vehID][2] - self.t_Observation[vehID][2]) / (self.maxLongLatSpeedAcc[1])
        
        
        speedDif = abs(self.vehsInfo[vehID]["devFromDesiredSpeed"])
        # ******* if there is no stress or speed Diff dont take any actions *******
        # self.longLatSpeedRewardCoef[0]
        speedVarLongCoef = .5 if not maxStress and speedDif <= .002 else 0
        # self.longLatSpeedRewardCoef[1]
        speedVarLatCoef = .5 if not maxStress else 0
        speedVarLong = -abs(self.action[index, 0]) * speedVarLongCoef
        speedVarLat = -abs(self.action[index, 1]) * speedVarLatCoef
        Var_punishment = speedVarLong + speedVarLat
        #   *************************************************************   #
        
        # jerk Calculation
        # self.posiAccFac is defined to change the influence of jerkLong and its max.value=1
        jerkLong_ = abs(self.tdict[vehID][114] - self.old_tdict[vehID][114]) / (
                (1+self.posiAccFac)*self.maxACC_Speed[0])  #self.posiAccFac Delta_T * abs(jerk) / Delta_AccMax
        jerkRewLong = -self.JeLonCoef * jerkLong_ # -.40 * jerkLong_
        
        # if inrusion is morethan n_intrusion do not consider jerk in your computation
        #jerkRewLong = jerkRewLong * max(0, 1 - Stress[0]/self.n_intrusion) 
        
        # Delata_lateral_speed(between two consecutive steps) / Delta_MaxLatteral_Speed
        self.jeLatCoef, self.JeLonCoef
        jerkLat_ = abs(
            self.tdict[vehID][50] - self.old_tdict[vehID][50]) / (2*self.maxACC_Speed[1])
        jerkRewLat = -self.jeLatCoef * jerkLat_# -.3 * jerkLat_
       
        # if inrusion is morethan n_intrusion do not consider jerk in your computation
        #jerkRewLat = jerkRewLat * max(0, 1 - Stress[0]/self.n_intrusion) 
        
        Jerk_punishment = jerkRewLat + jerkRewLong
        # if inrusion is morethan n_intrusion do not consider jerk in your computation
        Jerk_punishment = Jerk_punishment * max(0, 1 - Stress[0]/.3)
        
        self.jerkScan[vehID]["jerkLong"].append((self.tdict[vehID][114] - self.old_tdict[vehID][114]) / self.timeStep) #jerkLong_)
        self.jerkScan[vehID]["jerkLat"].append((self.tdict[vehID][50] - self.old_tdict[vehID][50]) / self.timeStep) #jerkLat_)
        self.speedDevScan[vehID].append(self.vehsInfo[vehID]["devFromDesiredSpeed"])
       

        self.Jerk_latSp_punishment[vehID] = [Var_punishment, Jerk_punishment]

        Total_Var_punishment = Var_punishment + Jerk_punishment

        #Var_punishment *= 1 - min(1, 2 * Stress)
        return Total_Var_punishment

    def Euc_rewardFunc(self, Eucl_vec, pre_Eucl_vec):
        if Eucl_vec >= pre_Eucl_vec:
            if Eucl_vec <= 1:
                return -.2*Eucl_vec
            else:
                return -1
        #improvement_rate = (pre_Eucl_vec - Eucl_vec)/pre_Eucl_vec
        elif Eucl_vec > 2 and (pre_Eucl_vec - Eucl_vec)/(pre_Eucl_vec + .001) <= .2:
            return (pre_Eucl_vec - Eucl_vec)/(pre_Eucl_vec + .001) - .8
        else:
            return (pre_Eucl_vec - Eucl_vec)/(pre_Eucl_vec + .001) - .5

def accCal(veID, vID, a=2.5, b=8.5, ve_len=3.2, ve_wid=1.6, v_len=3.2,v_wid=1.6):  # min latteral gap:1.489, min long gap: 5.4300
    # coordinates of vehicle
    # xce, yce = traci.vehicle.getLateralLanePosition(veID),traci.vehicle.getLanePosition(veID)
    xce, yce = veID[1], veID[0]
    # ve_len = traci.vehicle.getLength(veID)
    # ve_wid = traci.vehicle.getWidth(veID)
    # xcv, ycv = traci.vehicle.getLateralLanePosition(vID),traci.vehicle.getLanePosition(vID)
    xcv, ycv = vID[1], vID[0]
    # v_len = traci.vehicle.getLength(vID)
    # v_wid = traci.vehicle.getWidth(vID)
    vRearL = [xcv + v_wid / 2, ycv - v_len / 2]
    vRearR = [xcv - v_wid / 2, ycv - v_len / 2]
    vFrontL = [xcv + v_wid / 2, ycv + v_len / 2]
    vFrontR = [xcv - v_wid / 2, ycv + v_len / 2]

    if xce >= xcv and yce <= ycv:  # vehicle is in the right front position of ego vehicle
        vPosition = vRearL
    elif xce >= xcv and yce > ycv:  # vehicle is in the right rear position of ego vehicle
        vPosition = vFrontL
    elif xce < xcv and yce <= ycv:  # vehicle is in the left front position of ego vehicle
        vPosition = vRearR
    else:  # vehicle is in the left rear position of ego vehicle
        vPosition = vFrontR

    # ((xEll-xce)/a)**2 + ((yEll-yce)/b)**2
    dx, dy = abs(xce - vPosition[0]), abs(yce - vPosition[1])
    xEll, yEll = ellipse(vPosition, xce, yce, a, b)
    xDiff, yDiff = xEll - vPosition[0], yEll - vPosition[1] # diff between ideal ellipse border position and real position of lead. veh.
    distance = np.sqrt((xDiff) ** 2 + (yDiff) ** 2) # difference between ideal pos. and actual pos.
    xDiff_ideal, yDiff_ideal = xEll - xce, yEll - yce # ideal lateral and longitudinal diffrence between vehicles by means of ellipsoid border
    distance0 = np.sqrt(xDiff_ideal ** 2 + yDiff_ideal ** 2)
    distance1 = np.sqrt((xce - vPosition[0]) ** 2 + (yce - vPosition[1]) ** 2)
    # distance0 = distance1 + distance
    
    if yce >= vPosition[1]:
        distance = a - dx
        distance0 = a

    teta = np.arctan(dy / (dx + .0001))
    aLat, aLong = distance * np.cos(teta), distance * np.sin(teta)
    q = ((vPosition[0] - xce) / a) ** 2 + ((vPosition[1] - yce) / b) ** 2

    if xce >= xcv and q <= 1:  # vehicle is in the right position and inside of eliipse
        aLat = aLat  # if xce >= vPosition[0] else -aLat # drive to left and reduce or increase speed
        # aLat, aLong = aLat, -aLong
        aLong = -aLong if yce <= ycv else aLong
        status = "front right inside" if yce <= ycv else "rear right inside"
    if xce >= xcv and q > 1:  # vehicle is in the right position and outside of eliipse
        aLat = -aLat  # if xce >= vPosition[0] else aLat#drive to right and increase speed
        # aLat, aLong = -aLat, aLong
        aLong = aLong if yce <= ycv else -aLong  # drive to right and increase speed
        status = "front right outside" if yce <= ycv else "rear right outside"
    if xce < xcv and q <= 1:  # vehicle is in the left position and inside of eliipse
        aLat = -aLat  # if xce < vPosition[0] else aLat #drive to right and reduce speed
        # aLat, aLong = -aLat, -aLong
        aLong = -aLong if yce <= ycv else aLong
        status = "front left inside" if yce <= ycv else "rear left inside"
    if xce < xcv and q > 1:  # vehicle is in the left position and outside of eliipse
        aLat = aLat  # if xce < vPosition[0] else -aLat #drive to left and increase speed
        # aLat, aLong = aLat, aLong
        aLong = aLong if yce <= ycv else -aLong
        status = "front left outside" if yce <= ycv else "rear left outside"
    front = True if yce <= ycv else False
    return aLat, aLong, q, front, status, distance, distance0, distance1, teta

##################################################################
def ellipse(vPos, xce, yce, a=2.5, b=8.5):  # min latteral gap:1.489, min long gap: 5.4300
    dx, dy = abs(xce - vPos[0]), abs(yce - vPos[1])
    tan = dy / (dx + .0001)  # tangant teta
    x_el = np.sqrt((a * b) ** 2 / (b ** 2 + (a * tan) ** 2))
    y_el = np.sqrt(abs(a ** 2 - (x_el) ** 2)) * b / a
    if xce >= vPos[0]:  # vehicle is in the right position of ego vehicle
        x_el = -x_el
    if yce > vPos[1]:  # vehicle is in the rear position of ego vehicle
        y_el = -y_el
    return x_el + xce, y_el + yce


"""
# traci.close()
# env = laneFreeMagent_Env()
states = env.reset()
state_dim = np.size(env.observation_space)
action_dim = env.action_space[1].shape[0] 
max_action = env.action_space[0].high

traci.vehicle.getLateralLanePosition("veh_0")

data = env.step([])
data[1:]

"""
