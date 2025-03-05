# -*- coding: utf-8 -*-
"""
Created on Sat May 22 15:57:27 2021

@author: mehran
"""

import os
import sys, collections            
#from laneFreeUtils3_ import *
import traci
import sumolib
import traci.constants as tc
#from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pandas as pd
from collections import namedtuple, deque
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
import torch, random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#from gym.spaces import Box, Discrete, Tuple
import torch.nn.functional as F
import math, pickle
import pandas as pd
from bs4 import BeautifulSoup
import time
from  MultiAgentsCirEnv import *
from MultiAgentsDDPG_UnitedRBuffer import *
from MA_Seperated_st import *
#from MA_Seperated_st import *

#---------------------- Definition of global variables -----------------------#
global vehDict, LaneData
vehDict, LaneData = {}, []# vehDict containes specific data of each vehicle such as desired speed and
vehWidth, vehLenght, aCoef = 1.8, 3.2, 1.7
state_dim, agentsNum = 28, 6
#edgeID,laneID = "E1", "E1_0"
nudge_SpeedCoef = .1
edgeLengthDict = {"E_entrance": 1000, "E_onMerge":300, "E_1":100, "E_offMerge":300, "E_end":2400,
                  "E_offRamp":201, "E_onRamp":201, ":J_offMerge_0":3.25, ":J_offMerge_1":3.25,
                  ":J_offMerge_0_0":8, ":J_onMerge_0_0":3.25, ":J_onMerge_0_1":3.25,
                  ":J_onMerge_1_0":8, "E1":5000}
routeList = ["straight", "OnMerge", "straight_offMerge", "OnMerge_offMerge"]
midAccel = .7*4
discelTimeCoeff=1
reqDist, latMov = {}, {}
DeltaT = .25
intrusionStressCoef = 1
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


##############################################################################
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


######################################################
def getVehicleID(veID, longD=22):
    laneID = traci.vehicle.getLaneID(veID)
    vList = traci.lane.getLastStepVehicleIDs(laneID)
    xce, yce = traci.vehicle.getLateralLanePosition(veID), traci.vehicle.getLanePosition(veID)
    veList = [i for i in vList if yce < traci.vehicle.getLanePosition(i) < (yce + longD) or
              traci.vehicle.getLanePosition(i) == yce and
              xce > traci.vehicle.getLateralLanePosition(i)  # vehicle is in the right position of ego vehicle
              and i != veID]
    return veList


#########################################################
def totalAccCal(veID, longD=22, a=2.5, b=8.5):  # min latteral gap:1.489, min long gap: 5.4300
    # min lateral and longitudinal gap depend on a and b coefficients
    ALat, ALong, vListStatus = 0, 0, []
    vList = getVehicleID(veID, longD)
    for v in vList:
        aLat, aLong, vStatus = accCal(veID, v, a, b)
        ALat += aLat
        ALong += aLong
        vListStatus.append([aLat, aLong, vStatus])
    return ALat, ALong, vListStatus


#######################################################


def LatVehCal(egoVeh, vehDict, a=2.5, tds=.5, longD=25):
    vList = list(vehDict.keys())
    egoSpeed = traci.vehicle.getSpeed(egoVeh)
    # laneID = traci.vehicle.getLaneID(veID)
    # vList = traci.lane.getLastStepVehicleIDs(laneID)
    xce, yce = traci.vehicle.getLateralLanePosition(egoVeh), traci.vehicle.getLanePosition(egoVeh)
    veList_ego = [i for i in vList if abs(yce - traci.vehicle.getLanePosition(i)) < (longD) and
                  abs(traci.vehicle.getLateralLanePosition(i) - xce) <= 3 and i != egoVeh]
    vehDict[egoVeh][-1] = 0
    for v in veList_ego:
        vehSpeed, ycv = traci.vehicle.getSpeed(v), traci.vehicle.getLanePosition(v)
        follow_leadSpeed = [egoSpeed, vehSpeed] if (yce - ycv) <= 0 else [vehSpeed, egoSpeed]
        d0 = 5 if follow_leadSpeed[1] >= 10 else -0.125 * follow_leadSpeed[1] + 6.25
        b = d0 + tds * follow_leadSpeed[0]
        aLat, aLong, q, front, _, _, _ = accCal(egoVeh, v, a, b)
        aLong = aLong if front == True and aLong < 0 else 0
        if q < 1:
            vehDict[v][-2] -= .25 * aLat
            vehDict[egoVeh][-2] += .75 * aLat
            vehDict[egoVeh][-1] = min(vehDict[egoVeh][-1], aLong)
            vehDict[egoVeh][1] = 0
        else:
            vehDict[egoVeh][1] += 1
            # a_latList[egoVeh] += min(.5,init_lat_pos[egoVeh]-xce) if T[egoVeh] > 5 else 0

    return vehDict


def leadVeh_selection(egoVeh, vehDict, longD=30, MeanvehWidth=1.8):
    # lead_veh_ego = firstVeh
    vList = list(vehDict.keys())
    min_long_diff, min_lat_diff = float('inf'), float('inf')
    xce, yce = traci.vehicle.getLateralLanePosition(egoVeh), traci.vehicle.getLanePosition(egoVeh)
    veList_ego = [i for i in vList if yce < traci.vehicle.getLanePosition(i) < (yce + longD) and
                  abs(traci.vehicle.getLateralLanePosition(i) - xce) <= 1 and i != egoVeh]
    for v in veList_ego:
        long_diff = traci.vehicle.getLanePosition(v) - yce
        lat_diff = abs(traci.vehicle.getLateralLanePosition(v) - xce)
        if long_diff < min_long_diff:
            min_long_diff, vehDict[egoVeh][2] = long_diff, v

    return vehDict


##########################################################

def LatVehCal2(egoVeh, vehDict, a=2.8, tds=.5, longD=25):
    laneID = traci.vehicle.getLaneID(egoVeh)
    laneWidth = traci.lane.getWidth(laneID)
    egoVeh_width = traci.vehicle.getWidth(egoVeh)
    critPoint = .5 * laneWidth - .5 * egoVeh_width
    decel = traci.vehicle.getDecel(egoVeh)
    vList = list(vehDict.keys())
    egoSpeed = traci.vehicle.getSpeed(egoVeh)
    # longD = egoSpeed**2/(2*5) + 10 # maximume decceleration of policy is 5

    # laneID = traci.vehicle.getLaneID(veID)
    # vList = traci.lane.getLastStepVehicleIDs(laneID)
    xce, yce = traci.vehicle.getLateralLanePosition(egoVeh), traci.vehicle.getLanePosition(egoVeh)
    veList_ego = [i for i in vList if abs(yce - traci.vehicle.getLanePosition(i)) < (longD) and
                  abs(traci.vehicle.getLateralLanePosition(i) - xce) <= 3 and i != egoVeh]
    vehDict[egoVeh][-1], vehDict[egoVeh][3] = 0, False
    for v in veList_ego:
        vehSpeed, ycv = traci.vehicle.getSpeed(v), traci.vehicle.getLanePosition(v)
        xcv = traci.vehicle.getLateralLanePosition(v)
        follow_leadSpeed = [egoSpeed, vehSpeed] if (yce - ycv) <= 0 else [vehSpeed, egoSpeed]
        d0 = .1 if follow_leadSpeed[1] >= 10 else -0.125 * follow_leadSpeed[1] + 0.25
        b = d0 + tds * follow_leadSpeed[0] + 0.1
        aLat, aLong, q, front, status, _, _ = accCal(egoVeh, v, a, 1.0 * b)
        _, aLong, _, _, _, _, _ = accCal(egoVeh, v, a, b)  # as b is distance from central point of following vehicle

        aLong = aLong if front == True and aLong < 0 else float("inf")
        crit_position = False
        if q < 1:
            crit_position = critPoint - xce < .02 and xce == xcv
            if aLat != 0:
                aLat = .1 * np.sign(aLat) if abs(aLat) < .1 else aLat
            else:
                aLat = -.1 if xce >= 0 else .1
            vehDict[v][-2] -= .05 * aLat if crit_position == False else -.05 * aLat
            vehDict[egoVeh][-2] += .95 * aLat if crit_position == False else -.95 * aLat
            if vehDict[egoVeh][-1] > aLong:
                vehDict[egoVeh][-1], vehDict[egoVeh][
                    3] = aLong, v  # leading vehivle is determined by regarding mimimum_aLong
            vehDict[egoVeh][1] = 0
        else:
            vehDict[egoVeh][1] += 1
            # a_latList[egoVeh] += min(.5,init_lat_pos[egoVeh]-xce) if T[egoVeh] > 5 else 0

    return vehDict


############################################################
def LFree_indLoop(distance, detected_vehNum=0, detected_veh=[]):
    detect = (np.where(LaneData[:, 1] >= distance))[0]
    for i in detect:
        if not LaneData[i][0] in detected_veh:
            detected_veh.append(LaneData[i][0])
            detected_vehNum += 1
    return detected_vehNum, detected_veh

def macro_Prop(denum, den_merge, zoonEnt=[350, 850], zoonOnMer=[934,1295], zoonCom1=[1304, 1804], zoonCom2=[1805, 2295], 
               zoonOffMer=[2304,2604], zoonEnd1=[2670,3170], zoonEnd2=[3171,3671]):
    #global VehCrosEnt, vehCrosOnMer, vehCrosCom1, vehCrosCom2, vehCrosOffMer, vehCrosEnd1, vehCrosEnd2
    global zoonEntData,zoonOnMerData,zoonCom1Data,zoonCom2Data,zoonOffMerData,zoonEnd1Data,zoonEnd2Data
    global VehCrosEntList, VehCrosOnMergList, VehCrosCom1List, VehCrosCom2List, VehCrosOffMergList, VehCrosEnd1List \
        , VehCrosEnd2List
    SimTime = traci.simulation.getTime()
    if Count == 0:# these dataFrame are created at the first step of simulation
        zoonEntData = pd.DataFrame(columns=['stepTime', 'outFlow', 'meanSpeed','density', 'denum', 'den_merge'])
        zoonOnMerData = pd.DataFrame(columns=['stepTime', 'outFlow', 'meanSpeed','density', 'denum', 'den_merge'])
        zoonCom1Data = pd.DataFrame(columns=['stepTime', 'outFlow', 'meanSpeed','density', 'denum', 'den_merge'])
        zoonCom2Data = pd.DataFrame(columns=['stepTime', 'outFlow', 'meanSpeed','density', 'denum', 'den_merge'])
        zoonOffMerData = pd.DataFrame(columns=['stepTime', 'outFlow', 'meanSpeed','density', 'denum', 'den_merge'])
        zoonEnd1Data = pd.DataFrame(columns=['stepTime', 'outFlow', 'meanSpeed','density', 'denum', 'den_merge'])
        zoonEnd2Data = pd.DataFrame(columns=['stepTime', 'outFlow', 'meanSpeed','density', 'denum', 'den_merge'])
        VehCrosEntList, VehCrosOnMergList, VehCrosCom1List, VehCrosCom2List, VehCrosOffMergList, VehCrosEnd1List \
            , VehCrosEnd2List = [], [], [], [], [], [], []
        #zoonEntData,zoonOnMerData,zoonCom1Data,zoonCom2Data,zoonOffMerData,zoonEnd1Data,zoonEnd2Data 
        #VehCrosEnt, vehCrosOnMer, vehCrosCom1, vehCrosCom2, vehCrosOffMer, vehCrosEnd1, vehCrosEnd2= 0,0,0,0,0,0,0
    DataLen = len(LaneData)
    
    # requierd data for zoonEntData 
    throughPutVeh = LaneData[[LaneData[i][5][0] >= zoonEnt[0] and LaneData[i][5][0] <= zoonEnt[1]  for i in range(DataLen)]]
    vehNum = len(throughPutVeh)# density
    meanSpeed_Ent = np.mean([throughPutVeh[i][3] for i in range(vehNum)]) if vehNum != 0 else None
    
    # flow calculation
    VehCros = LaneData[[LaneData[i][5][0] >= zoonEnt[1] and LaneData[i][6] == "E_entrance"  for i in range(DataLen)]]
    prevCrosVeh = len(VehCrosEntList) 
    [VehCrosEntList.append(VehCros[i][0]) for i in range(len(VehCros)) if VehCros[i][0] not in VehCrosEntList]
    vehCrosNum = len(VehCrosEntList) - prevCrosVeh
    
    newRow = {'stepTime':SimTime, 'outFlow':vehCrosNum, 'meanSpeed':meanSpeed_Ent,'density':vehNum, 'denum':denum, 'den_merge':den_merge}
    zoonEntData = zoonEntData.append(newRow,ignore_index=True)
    #--------------------------------------------------------------------------#
    # requierd data for zoonOnMerData
    throughPutVeh = LaneData[[LaneData[i][5][0] >= zoonOnMer[0] and LaneData[i][5][0] <= zoonOnMer[1]  for i in range(DataLen)]]
    vehNum = len(throughPutVeh)# density
    meanSpeed = np.mean([throughPutVeh[i][3] for i in range(vehNum)]) if vehNum != 0 else None
    
    # flow calculation
    VehCros = LaneData[[LaneData[i][5][0] >= zoonOnMer[1] for i in range(DataLen)]]
    prevCrosVeh = len(VehCrosOnMergList) 
    [VehCrosOnMergList.append(VehCros[i][0]) for i in range(len(VehCros)) if VehCros[i][0] not in VehCrosOnMergList]
    vehCrosNum = len(VehCrosOnMergList) - prevCrosVeh
    
    newRow = {'stepTime':SimTime, 'outFlow':vehCrosNum, 'meanSpeed':meanSpeed,'density':vehNum,'denum':denum, 'den_merge':den_merge}
    zoonOnMerData = zoonOnMerData.append(newRow,ignore_index=True)
    
    #--------------------------------------------------------------------------#
    # requierd data for zoonCom1Data
    throughPutVeh = LaneData[[LaneData[i][5][0] >= zoonCom1[0] and LaneData[i][5][0] <= zoonCom1[1]  for i in range(DataLen)]]
    vehNum = len(throughPutVeh)# density
    meanSpeed = np.mean([throughPutVeh[i][3] for i in range(vehNum)]) if vehNum != 0 else None
    
    # flow calculation
    VehCros = LaneData[[LaneData[i][5][0] >= zoonCom1[1] for i in range(DataLen)]]
    prevCrosVeh = len(VehCrosCom1List) 
    [VehCrosCom1List.append(VehCros[i][0]) for i in range(len(VehCros)) if VehCros[i][0] not in VehCrosCom1List]
    vehCrosNum = len(VehCrosCom1List) - prevCrosVeh
    
    newRow = {'stepTime':SimTime, 'outFlow':vehCrosNum, 'meanSpeed':meanSpeed,'density':vehNum,'denum':denum, 'den_merge':den_merge}
    zoonCom1Data = zoonCom1Data.append(newRow,ignore_index=True)
    
    #--------------------------------------------------------------------------#
    # requierd data for zoonCom2Data
    throughPutVeh = LaneData[[LaneData[i][5][0] >= zoonCom2[0] and LaneData[i][5][0] <= zoonCom2[1]  for i in range(DataLen)]]
    vehNum = len(throughPutVeh)# density
    meanSpeed = np.mean([throughPutVeh[i][3] for i in range(vehNum)]) if vehNum != 0 else None
    
    # flow calculation
    VehCros = LaneData[[LaneData[i][5][0] >= zoonCom2[1] for i in range(DataLen)]]
    prevCrosVeh = len(VehCrosCom2List) 
    [VehCrosCom2List.append(VehCros[i][0]) for i in range(len(VehCros)) if VehCros[i][0] not in VehCrosCom2List]
    vehCrosNum = len(VehCrosCom2List) - prevCrosVeh
    
    newRow = {'stepTime':SimTime, 'outFlow':vehCrosNum, 'meanSpeed':meanSpeed,'density':vehNum,'denum':denum, 'den_merge':den_merge}
    zoonCom2Data = zoonCom2Data.append(newRow,ignore_index=True)
    
    #--------------------------------------------------------------------------#
    # requierd data for zoonOffMerData
    throughPutVeh = LaneData[[LaneData[i][5][0] >= zoonOffMer[0] and LaneData[i][5][0] <= zoonOffMer[1]  for i in range(DataLen)]]
    vehNum = len(throughPutVeh)# density
    meanSpeed = np.mean([throughPutVeh[i][3] for i in range(vehNum)]) if vehNum != 0 else None
    
    # flow calculation
    VehCros = LaneData[[LaneData[i][5][0] >= zoonOffMer[1] for i in range(DataLen)]]
    prevCrosVeh = len(VehCrosOffMergList) 
    [VehCrosOffMergList.append(VehCros[i][0]) for i in range(len(VehCros)) if VehCros[i][0] not in VehCrosOffMergList]
    vehCrosNum = len(VehCrosOffMergList) - prevCrosVeh
    
    newRow = {'stepTime':SimTime, 'outFlow':vehCrosNum, 'meanSpeed':meanSpeed,'density':vehNum,'denum':denum, 'den_merge':den_merge}
    zoonOffMerData = zoonOffMerData.append(newRow,ignore_index=True)
    
    #--------------------------------------------------------------------------#
    # requierd data for zoonEnd1Data
    throughPutVeh = LaneData[[LaneData[i][5][0] >= zoonEnd1[0] and LaneData[i][5][0] <= zoonEnd1[1]  for i in range(DataLen)]]
    vehNum = len(throughPutVeh)# density
    meanSpeed = np.mean([throughPutVeh[i][3] for i in range(vehNum)]) if vehNum != 0 else None
    
    # flow calculation
    VehCros = LaneData[[LaneData[i][5][0] >= zoonEnd1[1] and LaneData[i][6] == "E_end" for i in range(DataLen)]]
    prevCrosVeh = len(VehCrosEnd1List) 
    [VehCrosEnd1List.append(VehCros[i][0]) for i in range(len(VehCros)) if VehCros[i][0] not in VehCrosEnd1List]
    vehCrosNum = len(VehCrosEnd1List) - prevCrosVeh
    
    newRow = {'stepTime':SimTime, 'outFlow':vehCrosNum, 'meanSpeed':meanSpeed,'density':vehNum,'denum':denum, 'den_merge':den_merge}
    zoonEnd1Data = zoonEnd1Data.append(newRow,ignore_index=True)
    
    #--------------------------------------------------------------------------#
    # requierd data for zoonEnd2Data
    throughPutVeh = LaneData[[LaneData[i][5][0] >= zoonEnd2[0] and LaneData[i][5][0] <= zoonEnd2[1]  for i in range(DataLen)]]
    vehNum = len(throughPutVeh)# density
    meanSpeed = np.mean([throughPutVeh[i][3] for i in range(vehNum)]) if vehNum != 0 else None
    
    # flow calculation
    VehCros = LaneData[[LaneData[i][5][0] >= zoonEnd2[1] and LaneData[i][6] == "E_end" for i in range(DataLen)]]
    prevCrosVeh = len(VehCrosEnd2List) 
    [VehCrosEnd2List.append(VehCros[i][0]) for i in range(len(VehCros)) if VehCros[i][0] not in VehCrosEnd2List]
    vehCrosNum = len(VehCrosEnd2List) - prevCrosVeh
    
    newRow = {'stepTime':SimTime, 'outFlow':vehCrosNum, 'meanSpeed':meanSpeed,'density':vehNum,'denum':denum, 'den_merge':den_merge}
    zoonEnd2Data = zoonEnd2Data.append(newRow,ignore_index=True)
    

    


def LaneScan(laneID,vehDict):
    laneDict = {}
    vList = list(traci.lane.getLastStepVehicleIDs(laneID))
    for v in vList:
        latPos = traci.vehicle.getLateralLanePosition(v)
        longPos = traci.vehicle.getLanePosition(v)
        speed = traci.vehicle.getSpeed(v)
        desiredSpeed = vehDict[v][2]
        speedDev = abs(speed - desiredSpeed)
        laneDict[v] = [longPos, latPos, speed, speedDev]
    laneData = pd.DataFrame.from_dict(laneDict, orient='index',
                                      columns=['longPos', 'latPos', 'speed', 'speedDev'])
    LaneData = laneData.sort_values(by='longPos', ascending=True)
    LaneData = LaneData.reset_index()
    LaneData = LaneData.rename(columns={'index': 'veh_name'})
    LaneData = LaneData.to_numpy()
    return LaneData

    
def LaneScan2():
    
    laneDict = {}
    #for eID in list(edgeLengthDict.keys()):
    tdict = traci.edge.getContextSubscriptionResults("E1")
    
    for v in list(tdict.keys()):#v is key's Dict and the vehicle's name
        l = list(tdict[v].values())#[longPosition, LatPosition, ActualSpeed]
        l.append(abs(l[2] - vehDict[v][0]))# float(v.split("_")[1])))#the abs(actualSpeed - DesiredSpeed) is added to l
        #l.append(abs(l[2] - vehDict[v][2]))# the abs(actualSpeed - DesiredSpeed) is added to l
        laneDict[v] = l
    laneData = pd.DataFrame.from_dict(laneDict, orient='index',
                                       columns=['longPos', 'latPos', 'speed', 'lat_speed', 'POSITION_2D',
                                                'roadID', 'laneID', "accel", 'speedDev' ])
    
    #hh = [laneData["POSITION_2D"].values[i][0] for i in range(laneData["POSITION_2D"].values.shape[0])]
    #laneData["longCor"] = hh 
    #laneData.sort_values(by='longCor', ascending=True, inplace=True) 
    laneData.sort_values(by='longPos', ascending=True, inplace=True)
    
    global LaneData
    LaneData = laneData.reset_index()
    LaneData = LaneData.rename(columns={'index': 'veh_name'})
    LaneData = LaneData.to_numpy()
    #return LaneData   
        
def coordinatesScan(laneID):
    laneDict = {}
    vList = list(traci.lane.getLastStepVehicleIDs(laneID))
    for veh in vList:
        vPos = traci.vehicle.getPosition(veh)
        speed = traci.vehicle.getSpeed(veh)
        color = traci.vehicle.getColor(veh)
        desier_sp = traci.vehicle.getMaxSpeed(veh)
        laneDict[veh] = [vPos[0], vPos[1], speed, color, desier_sp]
    LaneData = pd.DataFrame.from_dict(laneDict, orient='index',
                                      columns=['v_XPos', 'v_YPos', 'speed', 'color', 'desier_sp'])
    LaneData = LaneData.reset_index()
    LaneData = LaneData.rename(columns={'index': 'veh_name'})
    LaneData = LaneData.to_numpy()
    return LaneData


def vehSpawn(laneID, edgeID, LaneData, routeID="route0", typeID="type2"):
    for veh_name in LaneData[:, 0]:
        v_index = np.where(LaneData == veh_name)[0][0]
        traci.vehicle.add(veh_name, routeID=routeID, typeID=typeID, departLane="first",
                          departSpeed=LaneData[v_index][3])
    traci.vehicle.setMaxSpeed(veh_name, LaneData[v_index][5])
    traci.vehicle.moveToXY(veh_name, edgeID, 1, LaneData[v_index][1], LaneData[v_index][2])
    traci.vehicle.setLaneChangeMode(veh_name, 0b001000000000)
    traci.vehicle.setSpeedMode(veh_name, 11110)  # obey the max and min acceleration setpoint after speed initialization
    traci.vehicle.setColor(veh_name, LaneData[v_index][4])


def vehProperties(vehDict, detected_list, tripInfo, laneLength):
    vehPorp = {}
    with open(tripInfo, 'r') as f:
        data = f.read()
    Bs_data = BeautifulSoup(data, "html.parser")
    bs_list = list(Bs_data.find_all('tripinfo'))
    n = len(bs_list)
    bsVehList = []
    [bsVehList.append(u.get("id")) for u in bs_list]
    for vID in bsVehList:
        if vID in detected_list:
            veh_trip = Bs_data.find("tripinfo", {"id": vID})
            actualTravelTime = float(veh_trip.get('duration'))
            desTrraveltime = laneLength / vehDict[vID][2]
            deltaTraveltime = actualTravelTime - desTrraveltime
            # realTimeDelay, desierTimeDelay, averageSpeed, DesierSpeed
            vehPorp[vID] = [actualTravelTime, desTrraveltime, deltaTraveltime,
                            np.mean(vehDict[vID][1]), vehDict[vID][2]]
    return vehPorp

############# Added Function for multi-agent traffic environment #############
#a = aCoef * vehWidth

def veh_neighbours(egoVehID, numNeighbours=8, egoVeh_width=1.8, vehLength=3.2, tds=.5, NudgeCoef=.2,nudge_SpeedCoef=.1,a=3):
    global vehDict, reqDist
    #vehDict[egoVeh][4]={"rep":[[],[]],"nudge":[[],[],[]] } #rep_nudge_Dict:{repulsion:[[vehID],[rep.intusion]], nudge:[[vehID],[nudge.intrusion]]}
    #---------------------------------------------------------------------------------------#
    ego_index = np.where(LaneData == egoVehID)[0][0] # determining ego_veh index in LaneData list
    vehDict[egoVehID][7] = ego_index # Add ego_index to its vehDict data
    ego_vehSpeed = LaneData[ego_index][3]
    ego_vehSpeed_lat = LaneData[ego_index][4]
    ego_vehLonLatPos = LaneData[ego_index][1:3]# Longitudinal and Lateral position of ego_vehicle
    #ego_vehLonLatPos = LaneData[ego_index][5]# [long.Cor, latCor] 
    ego_oldPos_lat = LaneData[ego_index][2] - (ego_vehSpeed_lat * DeltaT) # actual lat. pos. - lat.Speed*deltaT = oldLat.Pos
    d0 = 5 if ego_vehSpeed >= 10 else -0.125 * ego_vehSpeed + 6.25
    #d0=0
    b = d0 + tds * ego_vehSpeed - .5 * vehLength # In order to consider bumper-to-bumper distance .5 * vehLength is invoved 
    numNeigh = min(ego_index + numNeighbours, LaneData.shape[0] - 1) # regarding the real number of ego_vehicle's front neighbours
    
    #determine if egoVeh is in off merge status if yes impose no nudge on its neighbour vehicles
    #intrCoefNudge = 1.4 if LaneData[ego_index][6] == 'E_onMerge' and LaneData[ego_index][5][1] < -9.28 else 1 
    #intrCoefNudge = 1.4 if LaneData[ego_index][6] == 'E_offMerge' and LaneData[ego_index][5][1] > -9.28 and vehDict[egoVehID][9] == 1 else 1 # if egoVeh is doing off merg action its nudge should be magnified
    intrCoefNudge = 0 if LaneData[ego_index][6] == 'E_1' and vehDict[egoVehID][9] == 1 and LaneData[ego_index][1] > 600 and LaneData[ego_index][2] > -4.2 else 1 # if egoVeh is doing off merg action its nudge should be magnified
    # Should be checked ???????????????????????????????????????????????????
    intrCoefRep = 0 if LaneData[ego_index][6] == 'E_1' and vehDict[egoVehID][9] == 1 and LaneData[ego_index][1] > 600 and LaneData[ego_index][2] > -4.2 else 1 # if egoVeh is doing off merg action its nudge should be magnified

    
    #intrCoefNudge = 1
    egoVehRoadID = LaneData[vehDict[egoVehID][7]][6]
    neighDict = neighborSel(egoVehID, numNeighbours, ego_index)
    ego_oldPos_lat_lane = LaneData[ego_index][2] - (ego_vehSpeed_lat * DeltaT)
    ego_vehLonLatPos_lane= LaneData[ego_index][1:3] 

    for n_vehID in neighDict.keys():
        n_index = neighDict[n_vehID][0]
        n_vehRoadID = LaneData[n_index][6]
        # ["longDist", "latDist", "speedDif", "latSpeedDiff","accelDif"]
        ###########################################################################################
        if egoVehRoadID == n_vehRoadID :#and egoVehRoadID in ["E_onRamp", "E_offRamp"]: # if n_veh is in the egoVeh lane use lateral lane pos. instead of lat. cor. pos.
             #lon.& lat. pos lane are utilized if both vehicle are in "E_onRamp", or "E_offRamp"
             ego_vehLonLatPos_lane= LaneData[ego_index][1:3] 
             n_vehLonLatPos_lane= LaneData[n_index][1:3]
             ego_oldPos_lat = ego_vehLonLatPos_lane[1] - (ego_vehSpeed_lat * DeltaT)
             n_oldPos_lat = n_vehLonLatPos_lane[1] - (LaneData[n_index][4] * DeltaT)
             

        else:
            ego_vehLonLatPos_lane = LaneData[ego_index][5]# [long.Cor, latCor] 
            n_vehLonLatPos_lane = LaneData[n_index][5]
            ego_oldPos_lat = ego_vehLonLatPos_lane[1] - (ego_vehSpeed_lat * DeltaT) # actual lat. pos. - lat.Speed*deltaT = oldLat.Pos 
            n_oldPos_lat = n_vehLonLatPos_lane[1] - (LaneData[n_index][4] * DeltaT)

             
        #n_vehLonLatPos_lane= LaneData[n_index][1:3]
        vehDistance = n_vehLonLatPos_lane[0] - ego_vehLonLatPos_lane[0]
        lonDist_bump2bump = vehDistance - vehLength
        delta_v = LaneData[ego_index][3] - LaneData[n_index][3]
        inevitableColi = False
        if delta_v > 0:
            delta_a = LaneData[ego_index][8] - LaneData[n_index][8] 
            velocity_coef = [-.5 * discelTimeCoeff, delta_a, delta_v]
            T_velocity = max(np.roots(velocity_coef))
            motion_coef = [-discelTimeCoeff/6, delta_a/2, delta_v, 0]
            motion_equ = np.poly1d(motion_coef)
            required_dis = motion_equ(T_velocity)
            reqDist[egoVehID].append(required_dis)
            R_required_dis = max(required_dis, np.mean(reqDist[egoVehID]))
            d0_ = 5 if LaneData[n_index][3] >= 10 else - \
                0.125 * LaneData[n_index][3] + 6.25
            b1 = d0_ + tds * \
                LaneData[n_index][3] - .5 * vehLength
            b_ = R_required_dis + b1
            if R_required_dis >= lonDist_bump2bump: inevitableColi = True
        else:
            b_ = b
        LatDIFF = ego_vehLonLatPos_lane[1] - n_vehLonLatPos_lane[1]
        if lonDist_bump2bump < b_:
            n_oldPos_lat = n_vehLonLatPos_lane[1] - (LaneData[n_index][4] * DeltaT)
            latDiff_old = abs(ego_oldPos_lat - n_oldPos_lat)
                #self.old_tdict[veh][50] - self.old_tdict[n_veh][50])
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
        intrusion_per, Nudge = 0, 0
        if LatDIFF < 0:
            if LatDIFF > -(a_+ .1*vehWidth):
                Long_Intr = b_ - lonDist_bump2bump 
                if Long_Intr > 0:
                    intrusion_per = min(Long_Intr / b_, .9)   
        else:
            # long. & lat. pos leading veh
            aLat, aLong, q, _, _, distance, distance0, _, _ = accCal(
                ego_vehLonLatPos_lane, n_vehLonLatPos_lane, a_, b_, vehLength, vehWidth, vehLength, vehWidth)
            
            if q < 1 :
                intrusion_per = intrusionStressCoef * distance / distance0
                # if in
        if delta_v >= 0:
            Nudge = NudgeCoef * intrusion_per if inevitableColi == False  else .7*intrusion_per
            #self.vehsInfo[n_veh]["followVehInfo"][veh][-1][0] = Nudge
            #self.vehsInfo[veh]["leadVehInfo"][n_veh][-1]= intrusion_per
        else:
            intrusion_per *= .02
        

            
        Euc_dist = np.sqrt((ego_vehLonLatPos_lane[0] - n_vehLonLatPos_lane[0])**2 + (ego_vehLonLatPos_lane[1] - n_vehLonLatPos_lane[1])**2)
        #if intrusion_per != 0:
        vehDict[egoVehID][4]["rep"][0].append(n_vehID)
        vehDict[egoVehID][4]["rep"][1].append(intrusion_per)
        vehDict[egoVehID][4]["rep"][2].append(Euc_dist) 
        #if Nudge != 0:
        vehDict[n_vehID][4]["nudge"][0].append(egoVehID)
        vehDict[n_vehID][4]["nudge"][1].append(Nudge)# vehDict[egoVehID][8][0] = nudgeCoef that is defined for each vehicle in order to consider emergency vehicles or high priority vehicles
        vehDict[n_vehID][4]["nudge"][2].append(0)# sp_nudge calculated and redefined later ...
        vehDict[n_vehID][4]["nudge"][3].append(Euc_dist)
             ###########################################################################

def neighborSel(egoVehID, neighNum, ego_index):
    neighborDict = {}
    frontVehNum = LaneData.shape[0] - vehDict[egoVehID][7] - 1 # the number of vehicles located in front of ego vehicle in terms of long. coordination
    #n_index = vehDict[egoVehID][7]
    n_index = copy.deepcopy(ego_index)
    ########
    egoVehLongLatCor = LaneData[vehDict[egoVehID][7]][5]
    egoVehRoadID = LaneData[vehDict[egoVehID][7]][6]
    #egoVehLaneID = LaneData[vehDict[egoVehID][7]][7]
    egoVehRoadLength = edgeLengthDict[egoVehRoadID]
    ########
    while frontVehNum > 0 and neighNum > 0:
        n_index += 1
        n_vehID = LaneData[n_index][0] # neighbor vehID
        n_vehLongLatCor = LaneData[n_index][5]
        #longLaneDiff = LaneData[n_index][1] - LaneData[vehDict[egoVehID][7]][1]
        longCorDiff = n_vehLongLatCor[0] - egoVehLongLatCor[0]
        if longCorDiff < 80:
            
            n_vehLatCor = LaneData[n_index][5][1] 
            n_vehlonPos = LaneData[n_index][1]
            n_vehRoadID = LaneData[n_index][6]
            #n_vehLaneID = LaneData[n_index][7]
            # whether correspondig vehicles have latDiff less than 4m w.r.t their lateral coordination
            latCorDiff = egoVehLongLatCor[1] - n_vehLongLatCor[1]
            if abs(latCorDiff) <= 3.5:
                respLongDis = n_vehlonPos if egoVehRoadID == n_vehRoadID else n_vehlonPos + egoVehRoadLength # if front vehicle has different roadID the lenth of egoVehRoadID add to the front_veh longPos
                # if longDiff between vehicles is less than 200m consider the front veh as neighbor vehicle
                if (respLongDis - LaneData[vehDict[egoVehID][7]][1]) < 200:
                    #n_vehLatPos = n_vehLongLatCor[1] if egoVehLaneID != n_vehLaneID else LaneData[n_index][2] # if n_veh is in the egoVeh lane use lateral lane pos. instead of lat. cor. pos.
                    neighborDict[n_vehID] = [n_index, respLongDis, n_vehLongLatCor[1], longCorDiff, latCorDiff]
                    neighNum -= 1
        frontVehNum -= 1
    return neighborDict

def ab_Computation(ego_vehLonLatPos, ego_vehSpeed, ego_oldPos_lat, ego_index, n_index, n_vehLonLatPos, tds, a, b, vehLength):
    n_vehSpeed = LaneData[n_index][3]
    n_vehSpeed_lat = LaneData[n_index][4]
    n_oldPos_lat = n_vehLonLatPos[1] - (n_vehSpeed_lat * DeltaT) # actual lat. pos. - lat.Speed*deltaT = oldLat.Pos

    interVeh_longDis = n_vehLonLatPos[0] - ego_vehLonLatPos[0] # central-to-central long. distance
    #if interVeh_longDis >= 0:# ??????
    # adding relative vehicle speed to b Calaculation -> b_
    delta_v = ego_vehSpeed - n_vehSpeed 
    if delta_v > 0:
        required_time = delta_v / (4*maxLongLatSpeedAcc[0])# required time to decrease speed diff to zero: 4*maxLongLatSpeedAcc[0]= max_acceleration
        required_dis = -.5 *4* maxLongLatSpeedAcc[0] * required_time**2 + delta_v * required_time
        #required_dis1 = DisEst(ego_vehSpeed, n_vehSpeed)
        d0_ = 5 if n_vehSpeed >= 10 else -0.125 * n_vehSpeed + 6.25 
        b1 = d0_ + tds * ego_vehSpeed - .5 * vehLength # In order to consider bumper-to-bumper distance .5 * vehLength is invoved 
        b_ = required_dis + b1
    else: b_ = b
    
    # Adding relative vehicle lateral displacement(relative_displacement) to a calculation -> a_
    lonDist_bump2bump = interVeh_longDis - vehLength
    latDiff = abs(ego_vehLonLatPos[1] - n_vehLonLatPos[1])
    # both vehocle must be in the same edge to consider a_
    if lonDist_bump2bump < b_ and LaneData[ego_index][6] == LaneData[n_index][6]:
        latDiff_old = abs( ego_oldPos_lat -  n_oldPos_lat)
        relative_displacement = latDiff - latDiff_old
        # bumper to bumper long. distance -> leading_veh - ego_veh
        #lonDist_bump2bump = self.laneDict_ob[veh_n][2] - self.laneDict_ob[veh][2] - self.vehLength
        longDist = max(0, lonDist_bump2bump)
        a_ = a - np.sqrt(b_**2 - longDist**2) * relative_displacement / b_
        a_ = max(a, a_ )
    else: a_ = a
    return a_, b_, lonDist_bump2bump, latDiff            
    

def DisEst(vEgo, vLead):
    speedDiff = vEgo - vLead
    Est_dis = 0
    n = math.ceil(speedDiff / maxLongLatSpeedAcc[0])
    for _ in range(1000):
        Est_dis += .001 * timeStep * (vEgo ** 2 - (vEgo - n * maxLongLatSpeedAcc[0]) ** 2) / (2 * maxLongLatSpeedAcc[0])
        vEgo -= maxLongLatSpeedAcc[0] / 1000
    Est_dis = Est_dis
    critcalDis = Est_dis - vLead * n * timeStep
    return critcalDis 

def spNudgeEffct_Func(egoVeh):
    global vehDict
    # check if any nudge imposed on egoVeh if yes then use nudg_desiredSpeed instead of desiredSpeed for egoVeh
    effecDesiSpeed = vehDict[egoVeh][5] if len (vehDict[egoVeh][4]["nudge"][0]) > 0 else vehDict[egoVeh][0]
    index = np.where(LaneData == egoVeh)[0][0] # determining egoVeh index in LaneData list
    vehSpeed = LaneData[index][3] # actual speed of egoVeh
    speedDev = abs(effecDesiSpeed - vehSpeed) / effecDesiSpeed
    spNudgeEffect = 0
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
        spNudgeEffect = abs(speedReward)

    vehDict[egoVeh][6] = spNudgeEffect
    
    

def sp_nugde_func(egoVeh):
    global vehDict, statesData
    if len (vehDict[egoVeh][4]["nudge"][0]) > 0:
        for N_vehIndex, nudgeVeh in enumerate(vehDict[egoVeh][4]["nudge"][0]):
            # sp_nudge calculation w.r.t nudgeVeh -> egoVeh
            nudge = vehDict[egoVeh][4]["nudge"][1][N_vehIndex]
            spNudgeEffect = vehDict[nudgeVeh][6] if nudgeVeh not in ["onMergeForce", "offMergeForce"] else 0 # spNudgeEffect of nudgeVeh
            sp_nugde = nudge + np.sign(nudge) * spNudgeEffect / (1 + 2 * np.abs(nudge)) #Accummulated Nude
            vehDict[egoVeh][4]["nudge"][2][N_vehIndex] = sp_nugde

vehForceDict = {"onMergeForce":.3,"offMergeForce":-.3}

half_Freedom = .5 * (10.2 - 1.8)
def LR_freedom(laneWidth, vehWidth, half_Freedom, latPos):
    l_rLatFreedom = [half_Freedom - latPos, half_Freedom + latPos]
    

def get_veh_state(egoVeh, n_Num):
    #egoVeh_state includes: deviation_from_desired_speed, lateral_position, (agentNum-1)*RepulsiveStress,
    #(agentNum-1)*NudgeStress,(agentNum-1)*lateral_Diffrence
    global vehDict
    egoVeh_state = []
    egoVehIndex = vehDict[egoVeh][7] # egoVehIndex_in_LaneData
    #egoVehLongLatPos = [LaneData[egoVehIndex][1], LaneData[egoVehIndex][5][1]] # Long. lane_pos. & Lat. coordinate_position of egoVeh
    egoVeh_desSpeed = vehDict[egoVeh][0]
    egoVeh_actSpeed = LaneData[egoVehIndex][3]
    speedDev = (egoVeh_desSpeed - egoVeh_actSpeed) / egoVeh_desSpeed
    egoVeh_state.append(egoVeh_actSpeed)
    egoVeh_state.append(speedDev)
    egoVeh_state.append(LaneData[egoVehIndex][8]) # long Acceleration
    egoVeh_state.append(LaneData[egoVehIndex][4]) # lat Speed
    
    #egoVeh_state.append(egoVehLongLatPos[1])
    # use actual lane position if lateral coordination is greater than -7 else use virtual lane position -2
    
    vehDfRep = pd.DataFrame({'vehID':vehDict[egoVeh][4]['rep'][0],'nudgeRep':vehDict[egoVeh][4]['rep'][1],
                                  'Eucl.Dis':vehDict[egoVeh][4]['rep'][2],
                                  "isRep":np.ones(len(vehDict[egoVeh][4]['rep'][0]))}) #,'long.Pos':vehDict[egoVeh][4]['rep'][3]
    vehDfNudge = pd.DataFrame({'vehID':vehDict[egoVeh][4]['nudge'][0],'nudgeRep':vehDict[egoVeh][4]['nudge'][2],
                                  'Eucl.Dis':vehDict[egoVeh][4]['nudge'][3],
                                  "isRep":np.zeros(len(vehDict[egoVeh][4]['nudge'][0]))}) # just sp_nudge(['nudge'][2]) is used as nudge force                     
                                   # 'long.Pos':vehDict[egoVeh][4]['nudge'][4],
    
    ##########################################################################################################
    latPos = LaneData[egoVehIndex][2]
    # Should be checked ??????????????????????????????????????????????
    l_rLatFreedom = [half_Freedom - latPos, half_Freedom + latPos]
    nudgeVehs = vehDfNudge.sort_values(by="nudgeRep",ascending=False).head(1)
    RepVehs = vehDfRep.sort_values(by="nudgeRep",ascending=False).head(1)
    nudgeRepVehs = pd.concat([RepVehs, nudgeVehs], axis=0, ignore_index=True)
    #nudgeRepVehs.reset_index(drop=True, inplace=True)
    #nudgeRepVehs = RepVehs.append(nudgeVehs, ignore_index=True)
    nudgeVeh, repVeh = "", ""
    maxRep, maxNug = 0, 0
    
    for n, n_veh in enumerate(nudgeRepVehs['vehID'].values):
        n_vehIndex = vehDict[n_veh][7]
        #######################################################################
        egoVehRoadID = LaneData[egoVehIndex][6]
        n_vehRoadID = LaneData[n_vehIndex][6]
        if egoVehRoadID == n_vehRoadID :#and egoVehRoadID in ["E_onRamp", "E_offRamp"]: # if n_veh is in the egoVeh lane use lateral lane pos. instead of lat. cor. pos.
             #lon.& lat. pos lane are utilized if both vehicle are in "E_onRamp", or "E_offRamp"
             ego_vehLonLatPos_lane= LaneData[ego_index][1:3] 
             n_vehLonLatPos_lane= LaneData[n_index][1:3]
             ego_oldPos_lat = ego_vehLonLatPos_lane[1] - (ego_vehSpeed_lat * DeltaT)
             n_oldPos_lat = n_vehLonLatPos_lane[1] - (LaneData[n_index][4] * DeltaT)
             

        else:
            ego_vehLonLatPos_lane = LaneData[ego_index][5]# [long.Cor, latCor] 
            n_vehLonLatPos_lane = LaneData[n_index][5]
            ego_oldPos_lat = ego_vehLonLatPos_lane[1] - (ego_vehSpeed_lat * DeltaT) # actual lat. pos. - lat.Speed*deltaT = oldLat.Pos 
            n_oldPos_lat = n_vehLonLatPos_lane[1] - (LaneData[n_index][4] * DeltaT)
        
        #######################################################################
        latDist = ego_vehLonLatPos_lane[1] - n_vehLonLatPos_lane[1]
        interVehDist = ego_vehLonLatPos_lane[0] - n_vehLonLatPos_lane[0]
        speedDif = LaneData[egoVehIndex][3] - LaneData[n_vehIndex][3]
        if nudgeRepVehs.iloc[n]["isRep"] == 1:
            neigh_v = "leadVeh"
            maxRep = nudgeRepVehs.iloc[n]["nudgeRep"]
            repVeh = n_veh
        else:
            neigh_v ="followVeh"
            maxNug = nudgeRepVehs.iloc[n]["nudgeRep"]
            nudgeVeh = n_veh
            
        if (neigh_v == "leadVeh" and speedDif > 0) or (neigh_v == "followVeh" and speedDif < 0):
            SpeedDIFF = abs(speedDif)
            required_time = SpeedDIFF / midAccel
            required_dis = -.5 * midAccel * required_time**2 + SpeedDIFF * required_time
            safeDist = 2 + required_dis
        else: safeDist = 2
        if abs(interVehDist) <= vehLength + safeDist:
            freedome = max(0, abs(latDist) - vehWidth - 1.5)
            if latDist > 0: # n_veh is in the right side of veh
                if freedome < l_rLatFreedom[1]: 
                    l_rLatFreedom[1] = freedome
                #veh is in the left side of n_veh    
                if freedome < l_rLatFreedom[0]:
                   l_rLatFreedom[0] = freedome
            else:# n_veh is in the left side of veh
                if freedome < l_rLatFreedom[0]: 
                    l_rLatFreedom[0] = freedome
                # veh is in the left side of n_veh 
                if freedome < l_rLatFreedom[1]:
                   l_rLatFreedom[1] = freedome
    
    egoVeh_state.append(l_rLatFreedom[0]) # left Freedom
    egoVeh_state.append(l_rLatFreedom[1]) # right Freedom
    egoVeh_state.append(maxRep)
    egoVeh_state.append(maxNug)
    ###################################################################################################
    vehDict[egoVeh][2] = 3#agentNum
    #global vehDict, statesData
    #global statesData
    #egoStateData = copy.deepcopy(data[2000])
    
    vehDict[egoVeh][1][0] = egoVeh_state # egoVeh_state
    vehDict[egoVeh][1][1] = scaledDataFun2(scaled_States2,egoVeh_state)  
    #egoStateData[agentNum][1:18] = egoVeh_state
    #egoStateData = scaledDataFun(scaled_States,egoStateData,state_dim)
    vehDict[egoVeh][1][2] = [nudgeVeh, repVeh] #[repVeh,nudgeVeh]
    #Mul_agent.agents[agentNum].act(np.array(vehDict[egoVeh][1]), episode_num)
    
#scaled_States.transform(np.reshape(states,(-1,8)))    



# Scaling egoVeh stateList

dataPath0_ = r"D:\Traffic Research Topics\LaneFreeEnv\circle\result\firstResult/"

with open(dataPath0_ +'/Data.pkl', 'rb') as file:
    data = pickle.load(file)
#np.save(dataPath0+dataPath, data)
#L_index, state_dim = 18, 13
reshaped_data2= np.reshape(data, (-1,8))

from sklearn import preprocessing
#state_Normalization = preprocessing.StandardScaler().fit(reshData)
#max_abs_scaler = preprocessing.MaxAbsScaler()
scaled_States2 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scaled_Data1 = scaled_States2.fit_transform(reshaped_data2)

def scaledDataFun2(scaled_States2,data, stateDim=8, agentNum=1):
    scaledData = np.reshape(data, (-1,agentNum*stateDim))
    scaledData = scaled_States2.transform(scaledData)
    scaledData = np.reshape(scaledData, (agentNum,stateDim))
    return scaledData


 
#reshaped_data= np.reshape(data, (-1,state_dim*agentsNum))
#Data = [np.reshape(data[i], (-1,state_dim*6)) for i in range(len(data))]
from sklearn import preprocessing
#state_Normalization = preprocessing.StandardScaler().fit(reshData)
#max_abs_scaler = preprocessing.MaxAbsScaler()
#scaled_States = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#scaled_Data1 = scaled_States.fit_transform(reshaped_data)

def scaledDataFun(scaled_States,data2scale, stateDim=8):
    scaledData = np.reshape(data2scale, (-1,6*stateDim))
    scaledData = scaled_States.transform(scaledData)
    scaledData = np.reshape(scaledData, (6,stateDim))
    return scaledData




#indexList = [0,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
agent_Num = 6
#AgentState = []
#[AgentState.append(0) for _ in range(28)]
#statesData = [AgentState for i in range(agent_Num)]
#state_dim = 8

def egoState_ScalingFunc(egoVeh):
    global vehDict, statesData
    statesData[vehDict[egoVeh][2]] = vehDict[egoVeh][1]
    statesData = scaledDataFun(scaled_States,statesData,state_dim)
    vehDict[egoVeh][1] = statesData[vehDict[egoVeh][2]][MaskList] # egoVeh_state


def load_Policy(spaceDim=8, random_seed=0, device='cpu'):
    # state Dim., action Dim., agent Numbers, ...
    Mul_agent = MADDPG_Sep(spaceDim, 2, 6, random_seed, device=device)
    Mul_agent.reset()
    loadMADDPG_path = r"D:\Traffic Research Topics\LaneFreeEnv\circle\model\Last Version/"
    #loadMADDPG_path = r"D:\Traffic Research Topics\LaneFreeEnv\circle\model/"
    Mul_agent.load(loadMADDPG_path, device=device)
    return Mul_agent

def add_veh_func(veh_name, initEgoVehSpeed, edgeID, laneID, color, desiredSpeed,
                 laneLength=50, neiVehNum=5, routeID="straight", typeID="type2", nudgeCoef=.5, priority=0):
    global vehDict, vehNumP, reqDist
    
    selectedLoc = 0
    # initEgoVehSpeed = random.uniform(minMaxSpeed[0], minMaxSpeed[1])
    if routeID in ["straight_offMerge" ,"straight", "straight0"]:
       meanSpeed, selectedLoc  = add_veh_schime(laneLength, neiVehNum)
       initEgoVehSpeed = min(meanSpeed, initEgoVehSpeed)
    routeID = "route0"   
    if selectedLoc != -12 and initEgoVehSpeed > 1: # if there is an appropriate lateral location to insert veh, add veh
        traci.vehicle.add(veh_name, routeID=routeID, typeID=typeID,
                          departLane="best", departSpeed=initEgoVehSpeed)
        # traci.simulationStep()
        
        isOffMerge = True if routeID.split("_")[-1] == "offMerge" else False
        nudg_desiredSpeed = (nudge_SpeedCoef + 1) * desiredSpeed # defines an extra desired speed owing to the being nudge effect (the existance of nudge force)
        # [desiredSpeed, veh.StateList:[[scalesState],[rawState],[repVeh, nudgeVeh]], agent_num, actionsList:[speedChange, lateralAction, revised_target_speed], rep_nudge_Dict:{repulsion:[[vehID],[rep],[Eucl.Dis],[long.pos]], nudge:[[vehID],[nudge],[sp_nudge],[Eucl.Dis],[lon.pos]]},
        # nudge_desiredSpeed, spNudgeEffect, egoVehIndex_in_LaneData]
        #vehDict[egoVehID][8] = [nudgeCoef, priority], priority: strat from zero: no priority to highest priority with higher number 
        #vehDict[egoVehID][9] = isOffMerge
        # vehDict[egoVehID][10] used fo low speed veh no to locate in high speed lane area
        vehDict[veh_name] = [desiredSpeed, [[],[],[]], 0, [], {"rep":[[],[],[],[]],"nudge":[[],[],[],[],[]]}, nudg_desiredSpeed, 0, 0,[nudgeCoef,priority],
                             isOffMerge, 0]# desiredSpeed, veh.StateList, agent_num, actionsList, rep_nudge_Dict
        #{"rep":{'vehID':[],'rep':[],'EucDis':[]},"nudge":{'vehID':[],'nudge':[],'sp_nudge':[],'EucDis':[]}}
        # traci.vehicle.setSignals(vehID, 0) # https://sumo.dlr.de/docs/TraCI/Vehicle_Signalling.html
    
        
        traci.vehicle.setLaneChangeMode(veh_name, 0b001000000000)
        # traci.vehicle.setSpeedMode(veh_name, 11000)
    
        # Max.Speed is defined 20% greater than the desired Speed for each vehicle
        traci.vehicle.setMaxSpeed(veh_name, 1.2*desiredSpeed)
        # if vehicle's rout start from the begining of laneFree edge
        if routeID in ["straight_offMerge" ,"straight", "straight0", "route0"]:
            #initLatPosEgo = random.uniform(-maxLatChange, maxLatChange)
            #initLatPos = np.arange(-maxLatChange, maxLatChange, 2.7)
            #initLatPosEgo = [-4.2, -1.4,  1.4,  4.2][vehNumP % 4]
            #vehNumP += 1
            # traci.vehicle.setSpeed(veh_name, initEgoVehSpeed)
            initVehiCoord = traci.lane.getShape(laneID)[0]
            # Initial Longitudinal Ego Vehicle Position
            initXEgoVeh = initVehiCoord[0] + 1
            # Initial Lateral Ego Vehicle Position
            initYEgoVeh = initVehiCoord[1] + selectedLoc
            traci.vehicle.moveToXY(veh_name, edgeID, 1, initXEgoVeh, initYEgoVeh)
            
        # traci.simulationStep()
        # obey the max and min acceleration setpoint after speed initialization
        traci.vehicle.setSpeedMode(veh_name, 11110)
        traci.vehicle.setColor(veh_name, color)
    else:
        print("the is no appropriate location to insert veh.")
        pass
    reqDist[veh_name] = deque(maxlen=4)
    latMov[veh_name] = deque(maxlen=2)
    [latMov[veh_name].append(0) for _ in range(2)]
    

    
def add_veh_schime(laneLength, neiVehNum, td=.5, d0=4, vehWidth=1.8):
    # mean speed estimation of the first predefined length of entarnce lane
    #throughPutVeh = LaneData[[LaneData[i][1] <= laneLength and LaneData[i][6] == "E_entrance"  for i in range(len(LaneData[:30]))]]
    global latLoc
    latLoc.rotate(1)
    throughPutVeh = [LaneData[i] for i in range(len(LaneData[:30])) if LaneData[i][1] <= laneLength and LaneData[i][6] == "E_entrance"]
    vehNum = len(throughPutVeh)# density
    # if there is no veh in defined zoon, 25 m/s is regarded as mean speed
    meanSpeed = np.mean([throughPutVeh[i][3] for i in range(vehNum)]) if vehNum != 0 else 25
    vehData = throughPutVeh[:neiVehNum]
    #neiVeh = np.flip(vehData, axis=0)# reverse vehData of nth first veh's considered as having longer distance from entrance point
    neiVeh = throughPutVeh[:neiVehNum]
    veh_num = len(neiVeh)
    selectedLoc = -12
    if veh_num == 0:
        selectedLoc = latLoc[0]
    
    else:
        for loca in latLoc:
            satisf = 0
            #if selectedLoc != None: break
            
            for ind, vehData in enumerate(neiVeh):
                if abs(loca - vehData[2]) <= (vehWidth + .2): # if there is lateral overlap bettween corresponding vehicle then:
                   if (meanSpeed - vehData[3]) * td + d0 < vehData[1]:
                       satisf += 1
                       
                else:
                    satisf += 1
            if veh_num == satisf:
                selectedLoc = loca
                break
    return meanSpeed, selectedLoc
                


def resetVehDict(egoVeh):
    global vehDict
    vehDict[egoVeh][3] = [] #egoVeh Action reset
    vehDict[egoVeh][1] = [[],[],[]] # egoVeh state reset:scaledState, rawState, [repVeh, nudgeVeh]
    vehDict[egoVeh][4] = {"rep":[[],[],[],[]],"nudge":[[],[],[],[],[]]} # egoVeh nudgeRep_dict reset
    vehDict[egoVeh][6] = [] # egoVeh spNudgeEffect reset
    #vehDict[egoVeh][7] = 0 # egoVeh index in LaneData reset
   
revisedMaxSpeedRiseCoef = 1.1
#Mul_agent = load_Policy(spaceDim=8, random_seed=20, device=device)

def veh_execution(egoVeh, Mul_agent, maxLongLatSpeedAcc,latThreshold, episode_num=3000):
    # numNeighbours=8, tds=.5,NudgeCoef=.5,vehLength=3.2,veh_width=1.8,
    global vehDict
    #ego_index = vehDict[egoVeh][7]
    state, agent_num = vehDict[egoVeh][1][1], vehDict[egoVeh][2]
    
    #agent_num = 1
    actions = Mul_agent.agents[4].act(np.array(state), 300000)
    # lateral action execution
    if vehDict[egoVeh][1][0][-2] <= 0.02 and actions[0][1] > 0: actions[0][1] = 0
    latActSig = np.sign(actions[0][1])
    first_sign = actions[0][1] >= 0
    sameDir = all((lat_act >= 0) == first_sign for lat_act in latMov[egoVeh])
    #sameDir = np.sign(latMov[egoVeh][-1]) == latActSig
    #sameDir = sameDir == latActSig
    latActDiff = abs(latMov[egoVeh][-1] - actions[0][1]) 
    latAct = actions[0][1] if latActDiff > 3 or sameDir  else 0
    latMov[egoVeh].append(actions[0][1])
    
    lateralAction = maxLongLatSpeedAcc[1] * latAct
    speedChange = maxLongLatSpeedAcc[0] * actions[0][0]
    lateralAction = np.clip(lateralAction,-1.5,1.5 )
    #if vehDict[egoVeh][8][1] == 0:traci.vehicle.changeSublane(egoVeh, lateralAction)# if it is not emergency vehicle has sublane maneuver
    traci.vehicle.changeSublane(egoVeh, lateralAction)
    # Longitudinal action execution
    # step_time + acc.* deltaT = delta_Velocity
    # veh. actu.Speed at previous step_time + acc.* deltaT
    egoVeh_index = vehDict[egoVeh][7] # extract egoVeh_index from the egoVeh's vehDict Data 
    egoVeh_actSpeed = LaneData[egoVeh_index][3] # extract egoVeh speed from LaneData by egoVeh_index
        #revisedMaxSpeedRiseCoef = MaxSpeedRiseCoef #if max(np.abs(vehDict[egoVeh][4]['rep'][1])) > .05 else 1
    #if LaneData[ego_index][6] == "E_entrance" and LaneData[ego_index][1] < 300: speedChange = max(speedChange,-.12)  
    target_speed = egoVeh_actSpeed + speedChange
    revised_target_speed = np.clip(target_speed, 0,
                                   vehDict[egoVeh][0] * revisedMaxSpeedRiseCoef)  # desired speed can only be increased up to 10%
    # define actual speed for the next step_time
    traci.vehicle.setSpeed(egoVeh, revised_target_speed)
    vehDict[egoVeh][3] = [speedChange, lateralAction, revised_target_speed] # add vehice implemented action to its database 

Count, RunStep = 0, 0
maxLatChange = 10.2 / 2 - 1.8/2
latLoc = np.arange(-maxLatChange, maxLatChange+.01, 4)
latLoc = collections.deque(latLoc)

routeList = ["straight", "OnMerge", "straight_offMerge", "OnMerge_offMerge"]
def laneFreeExecution(stepNum, edgeID, denum, den_merge,dubVehDen,entVehNum, latThreshold, initEgoVehSpeed=20,
                      neigboursNum=8, tds=.5,NudgeCoef=.5,nudge_SpeedCoef=.1,a=3,
                      vehLength=3.2 ,veh_width=1.8, speedRange=[25, 35],
                      distance=4990,routeID="straight",P_ent = [.9, .1], P_onMerge=[.7, .3], insStep=50):
    n_Num = 5 # number of neigbour vehicle considered as agents in the vecinity of egoVeh
    global vehDict, LaneData, Count, timeStep, RunStep, latLoc
    laneLength, neiVehNum = 50, 4
    #P = [.81, .08, .05, .06] # ["straight", "OnMerge", "straight_offMerge", "OnMerge_offMerge"]
    #P = [.9, .1]# ["straight","straight_offMerge"]
    meanSpeedrange = (speedRange[0] + speedRange[1]) / 2
    timeStep = traci.simulation.getDeltaT()
    laneID = edgeID + "_" + str(traci.edge.getLaneNumber(edgeID) - 1)
    laneWidth = traci.lane.getWidth(laneID)
    maxLatChange = laneWidth / 2 - .9
    #LaneData = LaneScan(laneID, vehDict)
    LaneScan2()
    if RunStep == 0: pass#macro_Prop(denum, den_merge)
    # VAR_SPEED = 64, VAR_LANEPOSITION = 86 ,VAR_LANEPOSITION_LAT = 184
    #traci.edge.subscribeContext(edgeID, tc.CMD_GET_VEHICLE_VARIABLE,5000, [86,184,64])
    # initLanePos =
    speedDevList, collisionList, meanSpeedList, i = [], [], [], 0
    _, detected_veh = LFree_indLoop(distance, detected_vehNum=0, detected_veh=[])
    detected_vehNum = 0
    onRamp = 0
    #addVehNum = 1
    
    while i <= stepNum:
        if i > insStep:denum = 300
        addVehNum = 1
        if denum == 1:
            insVehNum = i % dubVehDen # if it's requierd to add more than one vehicle at eache stepTime in entrance edge
            addVehNum = 1 if insVehNum != 0 else entVehNum # entVehNum: number of veh must be inserted at each step time
            #initLatPos = np.arange(-maxLatChange, maxLatChange, 2.7)
            #initLatPosEgo = [-4.2, -1.4,  1.4,  4.2][vehNumP % 4]
        i += 1
        if i % denum == 0:
            
            onRamp += 1
            try:
                #routeID =  routeID if onRamp % 8 !=0 else "OnMerge"
                #routeID = np.random.choice(routeList, 1, p = P)[0] # ["straight", "OnMerge", "straight_offMerge", "OnMerge_offMerge"]
                
                while addVehNum > 0:
                    routeID = np.random.choice(["straight","straight_offMerge"], 1, p = P_ent)[0]
                    #routeID = "straight0"
                    desireSp = random.randint(speedRange[0], speedRange[1])
                    #initLatPosEgo = random.uniform(-1.2, maxLatChange) if desireSp > meanSpeedrange else random.uniform(-maxLatChange, 1.2) 
                    color = random.sample(
                        [(255, 0, 0), (0, 255, 255), (255, 0, 255)], 1)[0]
                    if routeID in ["straight_offMerge", "OnMerge_offMerge"]:color = (255, 255, 255) 
                    #if routeID in ["OnMerge", "OnMerge_offMerge"]:desireSp = 35
                    veh_name = str(Count) + "_" + str(desireSp) + "_" + str(addVehNum-1)
                    add_veh_func(veh_name, initEgoVehSpeed, edgeID, laneID, color, desireSp, laneLength, neiVehNum,
                                 routeID, nudgeCoef=NudgeCoef)  # a vehicle can violate from its desired speed up to 20%
                    Count += 1
                    addVehNum -= 1
                #vehDict[veh_name] = [[False,float("inf"),0], [], desireSp, [False,float("inf")], [0,0,0], 0,0]
                # [[closestFollower, distToClosedFollower],0,desireSpeed,[LeadVeh,DistToLeadVeh],accel,aLat,additive_accel]
            # vList = traci.lane.getLastStepVehicleIDs(laneID)
            except:
                i += 1
                while addVehNum > 0:
                    routeID = np.random.choice(["straight","straight_offMerge"], 1, p = P_ent)[0]
                    #routeID = "straight0"
                    desireSp = random.randint(speedRange[0], speedRange[1])
                    color = random.sample(
                        [(255, 0, 0), (0, 255, 255), (255, 0, 255)], 1)[0]
                    if routeID in ["straight_offMerge", "OnMerge_offMerge"]:color = (255, 255, 255) 
                    #if routeID in ["OnMerge", "OnMerge_offMerge"]:desireSp = 35
                    veh_name = str(Count) + "_" + str(desireSp) + "_" + str(addVehNum-1)
                    add_veh_func(veh_name, initEgoVehSpeed, edgeID, laneID, color, desireSp, laneLength, neiVehNum
                                 ,routeID, nudgeCoef=NudgeCoef)  # a vehicle can violate from its desired speed up to 20%
                    Count += 1
                    addVehNum -= 1
                #vehDict[veh_name] = [[False,float("inf"),0], [], desireSp, [False,float("inf")], [0,0,0], 0,0]
                # [[closestFollower, distToClosedFollower],0,desireSpeed,[LeadVeh,DistToLeadVeh],accel,aLat,additive_accel]
            # vList = traci.lane.getLastStepVehicleIDs(laneID)
                
        
        
        """
        # define flow of on_rampe entrance
        if i % den_merge == 0:#routeID =  routeID if onRamp % 8 !=0 else "OnMerge"
            
            routeID = np.random.choice(["OnMerge","OnMerge_offMerge"], 1, p = P_onMerge)[0] # ["straight", "OnMerge", "straight_offMerge", "OnMerge_offMerge"]
            #routeID = "straight0"
            desireSp = random.randint(speedRange[0], speedRange[1])
            color = random.sample(
                [(255, 0, 0), (0, 255, 255), (255, 0, 255)], 1)[0]
            if routeID in ["straight_offMerge", "OnMerge_offMerge"]:color = (255, 255, 255) 
            #if routeID in ["OnMerge", "OnMerge_offMerge"]:desireSp = 35
            veh_name = str(Count) + "_" + str(desireSp) + "_onMerge"
            #initLatPosEgo = 0
            add_veh_func(veh_name, initEgoVehSpeed, edgeID, laneID, color, desireSp, laneLength, neiVehNum,
                         routeID, nudgeCoef=NudgeCoef)  # a vehicle can violate from its desired speed up to 20%
            Count += 1
            #vehDict[veh_name] = [[False,float("inf"),0], [], desireSp, [False,float("inf")], [0,0,0], 0,0]
            # [[closestFollower, distToClosedFollower],0,desireSpeed,[LeadVeh,DistToLeadVeh],accel,aLat,additive_accel]
            # vList = traci.lane.getLastStepVehicleIDs(laneID)
        """
        global vList
        vList = LaneData[:, 0]
        # reset each veh Data in vehDict
        [resetVehDict(egoVeh) for egoVeh in vList]
        #state estimation for throughput vehicles
        [veh_neighbours(egoVeh, neigboursNum, veh_width, vehLength, tds,
                         NudgeCoef,nudge_SpeedCoef,a) for egoVeh in vList] # Update Nudge and repulsion and calculate egoVeh_index in LaneData]
        [spNudgeEffct_Func(egoVeh) for egoVeh in vList] # vehDict[egoVeh][6] = spNudgeEffect: calculate rewardSpeed affected by any nudge imposed on egoVeh
        [sp_nugde_func(egoVeh) for egoVeh in vList] # calculate nudge affected by dev. from desired speed for neighbour veh
        [get_veh_state(egoVeh, n_Num) for egoVeh in vList]
        [traci.vehicle.setSignals(egoVeh, 0) for egoVeh in vList]  # https://sumo.dlr.de/docs/TraCI/Vehicle_Signalling.html]
            
            # rest the vehicle's closest follower data and, a_lat, additive_accel
            #vehDict[egoVeh][0], vehDict[egoVeh][-2:] = [False,float("inf"), 0], [0, 0]
        for egoVeh in vList:
            veh_execution(egoVeh, Mul_agent, maxLongLatSpeedAcc,latThreshold)
        
        #with open(dataPath +'/vehDict.pkl', 'wb') as file:pickle.dump(vehDict, file)
        traci.simulationStep()
        RunStep += 1
        if len(traci.simulation.getCollisions()) > 0: break
        #LaneData = LaneScan(laneID, vehDict)
        LaneScan2()
        if RunStep % 4 == 0: pass#macro_Prop(denum, den_merge)
        speedDevList.append(np.mean(LaneData[:, 4]))
        
        # delet veh from vehDict if veh is not anymore in LaneData (eliminated from traffic scenario)
        if RunStep % 101 == 0:
            DictVehList = list(vehDict.keys())
            [vehDict.pop(veh) for veh in DictVehList if veh not in vList and veh not in DictVehList[-10:]]
        # collisionList.append(traci.simulation.getCollisions())
        meanSpeedList.append(traci.lane.getLastStepMeanSpeed(laneID))
        detected_vehNum, detected_veh = LFree_indLoop(distance, detected_vehNum, detected_veh)
    return [speedDevList, collisionList, detected_vehNum, meanSpeedList, detected_veh]


resultPath = r'D:\Python_Files\Traffic Research Topics\TU of Berlin\sumo-rl-master\nets\New folder (2)\circle_net\MultiAgentsSumoEnv0\September2022\On_off_ramp\FundamentalDiagResults/'
def saveFundDiagRe (resultPath):
    zoonEntData.to_csv(resultPath + 'zoonEntData.csv')
    zoonEntData.to_pickle(resultPath + 'zoonEntData.pkl')

    zoonOnMerData.to_csv(resultPath + 'zoonOnMerData.csv')
    zoonOnMerData.to_pickle(resultPath + 'zoonOnMerData.pkl')
    
    zoonCom1Data.to_csv(resultPath + 'zoonCom1Data.csv')
    zoonCom1Data.to_pickle(resultPath + 'zoonCom1Data.pkl')

    zoonCom2Data.to_csv(resultPath + 'zoonCom2Data.csv')
    zoonCom2Data.to_pickle(resultPath + 'zoonCom2Data.pkl')

    zoonOffMerData.to_csv(resultPath + 'zoonOffMerData.csv')
    zoonOffMerData.to_pickle(resultPath + 'zoonOffMerData.pkl')

    zoonEnd1Data.to_csv(resultPath + 'zoonEnd1Data.csv')
    zoonEnd1Data.to_pickle(resultPath + 'zoonEnd1Data.pkl')

    zoonEnd2Data.to_csv(resultPath + 'zoonEnd2Data.csv')
    zoonEnd2Data.to_pickle(resultPath + 'zoonEnd2Data.pkl')
    ####################### read ##########################
    # new_df = pd.read_csv(resultPath + 'zoonOnMerData.csv')
    # new_df2 = pd.read_pickle(resultPath + 'zoonOnMerData.pkl')

P_ent, P_onMerge = [.9, .1], [.7, .3]
dubVehDen,entVehNum = 3, 2
denum, den_merge = 1, 4
initEgoVehSpeed=25
neigboursNum=8
tds=.5
NudgeCoef=.5
nudge_SpeedCoef=.1
a=3
vehLength=3.2
veh_width=1.8
speedRange=[25, 35]
distance=4990
routeID="straight"
n_Num = 5
episode_num=3000
