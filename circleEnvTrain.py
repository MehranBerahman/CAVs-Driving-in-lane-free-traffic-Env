# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 19:52:14 2023

@author: mehran
"""
import time
import os
import sys
from laneFreeUtils3_ import *
import traci
import sumolib
import traci.constants as tc
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pandas as pd
from collections import namedtuple, deque
import torch, random
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
import torch.nn.functional as F
import math, pickle
sumoCfgFile = r"C:\Users\MADDPG_LF\circle_5.sumocfg"

import sys
#sys.path.insert(0,path)
from  MultiAgentsCirEnv import *
#from MultiAgentsDDPG_for_SUMO_Env import *
from MultiAgentsDDPG_UnitedRBuffer import *
from MA_Seperated_st import *
# from MultiAgWithSingleNet import *
# from MADDPG_for_SUMO_2 import *


use_gui = False
traci.close()
    
env = laneFreeMagent_CirEnv(sumoCfgFile=sumoCfgFile, agentNum=6, vehWidth=1.8, vehLength=3.2,
             out_csv_name=None, use_gui=use_gui, maxTimeSim=1000, edgeID="e1", Init_minMaxSpeed=[20, 40], Desired_minMaxSpeed=[20, 40],
             longLatCollisionDis=[.4, .2], initPos0=100, max_step=1000, speedRewardCoef=1, Nudge_ind=1,
             longLatSpeedRewardCoef=[.4, .4], print_initial_statuse=False, maxLongLatSpeedAcc=[4, 1.5], n_intrusion=.25, wrongLatActPun=-1, discelTimeCoeff=1, reqDisLen=4,
             MaxSpeedRiseCoef=1.1, single_agent=False, seed=0, NudgeCoef=.2, intrusionStressCoef=1, MinspeedReward=-.2, VehSignalling=True, aCoef=1.38, tds=.5, posiAccFac=1,
             jeLatCoef=1, JeLonCoef=1, circleLength=400, depSpeed=20, ignorCollision=False)

info = env.reset()
state_dim = len(info[0][0])
env.vehsInfo[env.vList[3]]["prim_lead&fol_veh"]
####################################
global stateScaleList
stateScaleList = [(-.1,.4), (-maxLongLatSpeedAcc[0],2*maxLongLatSpeedAcc[0]) ,
                  (-maxLongLatSpeedAcc[1], 2*maxLongLatSpeedAcc[1]),
                  (-.9,1.8), (-.9,1.8), (-.9,1.8) ,(-.9,1.8),(-.9,1.8),
                  (-.4,.8),(-.4,.8),(-.4,.8),(-.4,.8),(-.4,.8)]
def scaled_Data(state):
    State = []
    for i,s in enumerate(state):
        S = 2*((s - stateScaleList[i][0]) / stateScaleList[i][1]) - 1
        S = np.clip(S, -1, 1)
        State.append(S)
    return(State)
###################################
#state_dim = env.observation_space["agent_0"].shape[0]
state_dim = len(info[0][0])
action_dim = env.action_space["agent_0"].shape[0] 
max_action = env.action_space["agent_0"].high
agentsNum=6
#---------------------- Data Scaling -----------------------#
def statesVar_cal(env, stateDim=8, agentsNum=10, seedChange=1000, max_iteration=10000):
    #data = np.zeros((1*max_iteration,6,stateDim))
    data = []
    rewardsData = np.zeros((max_iteration,agentsNum))
    actionData = np.zeros((max_iteration,agentsNum,2))
    min_max_data = [[0,0] for _ in range(stateDim)]
    env_info = env.reset()
    for i in range(max_iteration):
        env.stepNum = 1
        if i % seedChange == 0:
            seed = random.randint(1,10)
        if np.any(env_info.dones ) or env.stepNum % 1000 == 0 :env_info = env.reset(seed)
        action = np.array([[np.clip(random.uniform(-1.0,1.0001),-1,1),np.clip(random.uniform(-1.0,1.0001),-1,1)]  for _ in range(agentsNum)])
        env_info = env.step(action)
        for ind in range(stateDim):
           max_ind = max(np.array(env_info.CurrentStates)[:,ind])
           min_ind = min(np.array(env_info.CurrentStates)[:,ind])
           min_max_data[ind][1] = max(max_ind, min_max_data[ind][1])
           min_max_data[ind][0] = min(min_ind, min_max_data[ind][0])
        #data[1*i] = np.array(env_info.CurrentStates)
        data.append(env_info.CurrentStates)
        #data.append(env_info.NextStates)
        #data[2*i+1] = np.array(env_info.NextStates)
    
        actionData[i] = np.array(action) 
        rewardsData[i] = env_info.rewards
    return min_max_data, data, rewardsData, actionData

#min_max_data, data, rewardsData, actionData = statesVar_cal(env, state_dim, seedChange=100, max_iteration=1000)
env.findPrimPred()
len(data)
dataPath0_ = r"C:\Users\MADDPG_LF"#"D:\Traffic Research Topics\LaneFreeEnv\circle\result\firstResult/"

"""
with open(dataPath0_ +'/Data.pkl', 'wb') as file:
    pickle.dump(data, file)
"""
with open(dataPath0_ +'/Data.pkl', 'rb') as file:
    data = pickle.load(file)
len(data[0][0]), len(data)
#np.save(dataPath0+dataPath, data)
#L_index, state_dim = 18, 13
reshaped_data= np.reshape(data, (-1,state_dim*agentsNum))

from sklearn import preprocessing
#state_Normalization = preprocessing.StandardScaler().fit(reshData)
#max_abs_scaler = preprocessing.MaxAbsScaler()
scaled_States = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scaled_Data1 = scaled_States.fit_transform(reshaped_data)

############################################################

random_seed=0
env_Info = env.reset()
BUFFER_SIZE=int(5e5)
batch_size=128
tau=0.01
UPDATE_EVERY=3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#state_dim = 13
#L_index = 18
i_episode = 1
TOTAL_EPISODES, max_t, print_every = 2000, 100, 10
scores_deque, EQ_scores_deque = deque(maxlen=10), deque(maxlen=10)
average_step, EQ_average_step = deque(maxlen=10), deque(maxlen=10)
all_scores, EQ_all_scores = [], []
#data_for_scale = data
rolling_avg_100, EQ_rolling_avg_100 = [],[]   
#total_state_size = state_dim*agentsNum 
total_action_size = action_dim * agentsNum
NOISE_DECAY_LIMIT= 100
BETA_EPISODES_LIMIT = 100
#agent = MADDPGAgent(state_dim, action_dim, agentsNum, random_seed, batch_size, tau, BUFFER_SIZE, device, UPDATE_EVERY)
agent = MADDPG_Sep(state_dim, action_dim, agentsNum, random_seed, batch_size, tau, BUFFER_SIZE, device, 
                   UPDATE_EVERY, NOISE_DECAY_LIMIT)
agent.reset()

time.ctime()
fileName =  r"C:\Users\MADDPG_LF\checkResult/"#"D:\Traffic Research Topics\LaneFreeEnv\circle/"

def scale_beta(episode_num):
    return min(1.0, ( episode_num / BETA_EPISODES_LIMIT))
def scaledDataFun(scaled_States,data, stateDim=8, agentsNum=6):
    scaledData = np.reshape(data, (-1,agentsNum*stateDim))
    scaledData = scaled_States.transform(scaledData)
    scaledData = np.reshape(scaledData, (agentsNum,stateDim))
    return scaledData


def weightedReward(env, states, agent_number, rewards):
    SpeedReward = [env.speedRewardDict[veh] for veh in env.agentList_old]
    Rewards = agent_number * [0]
    Rewards = agent_number * [0]
    for agent_num in range(agent_number):
        Reward_n = np.array(SpeedReward)[np.arange(len(rewards))!=agent_num]
        weight = np.abs(states[agent_num][8:13])# speed reward of following veh on leading veh. by means of Long. Nudge Effects
        Rewards[agent_num] = min(weight * Reward_n) + rewards[agent_num]   
    return Rewards

def loadData(episode):
    path = r"D:\Python_Files\Traffic Research Topics\TU of Berlin\sumo-rl-master\nets\New folder (2)\circle_net\MultiAgentsSumoEnv0\January2023\result"
    AveReward_PerStep_path = f"\AveReward_PerStep_In_EachEpisod_list_Ep{episode}.npy"
    avg100_path = f"\\rolling_avg_100_Ep_{episode}.npy"
    coliPath = f"\\collision_data_{episode}.npy"
    AveReward_PerStep_In_EachEpisod_list = np.load(path + AveReward_PerStep_path).tolist()
    rolling_avg_100 = np.load(path + avg100_path).tolist()
    collision_data  = np.load(path + coliPath).tolist()
    return AveReward_PerStep_In_EachEpisod_list, rolling_avg_100, collision_data
    
# AveReward_PerStep_In_EachEpisod_list, rolling_avg_100, collision_data = loadData(1400)
coef = 1
#np.array(env_info.CurrentStates)[-1,18:23]                          
AveReward_PerStep_In_EachEpisod, EQ_AveReward_PerStep_In_EachEpisod = [], [] 
AveReward_PerStep_In_EachEpisod_list, EQ_AveReward_PerStep_In_EachEpisod_list = [], []
Ave_speedDev_In_EachEpisod = []
collision_data, num_of_collision = [], 0
from datetime import datetime
datetime.now()
#seed = random.randint(1,10)
seed = random_seed
episode_num= i_episode
#overallStep = 6000
sacaledUpdateNum = 20000
rescale_Accurredurred = False
E_last_Step = []
updateScales = 50000
#indexList = [13,14,15,16,17]

def MasFunc(indexList, state_dim):
    indBool = []
    for index in range(state_dim):
        ind = False if index in indexList else True
        indBool.append(ind)
    return indBool


def nei_states_Func(stateDim=8):
    N_states = []
    for veh in env.vList:
        p_lead, p_fol = env.vehsInfo[veh]["prim_lead&fol_veh"]
        if p_lead != "None":
          lead_states = env.t_Observation[p_lead]
        else:lead_states = [env.t_Observation[veh][0]+2, 0, 0, 0, 4, 4, 0, 0]
        if p_fol != "None":
          fol_states = env.t_Observation[p_fol]
        else:fol_states = [env.t_Observation[veh][0]-2, 0, 0, 0, 4, 4, 0, 0]
        nei_states = [lead_states, fol_states]
        nei_states = scaledDataFun(scaled_States,nei_states,state_dim)
        nei_states = np.reshape(nei_states, (1, 2*stateDim))

episode_num, overallStep = 1, 1
#MaskList =  MasFunc(indexList, L_index)
Mode = 1

for i_episode in range(episode_num, TOTAL_EPISODES+1):
     #noise_decay = noise_decay_schedule(i_episode)
     beta = scale_beta(i_episode)
     seed = random.randint(1,100)
     try: 
         env_info = env.reset(seed) # reset the environment
     except:
         try:
             env_info = env.reset(seed)
         except: env_info = env.reset(seed)
     #states = env_info.CurrentStates 
     critic_next_states = scaledDataFun(scaled_States,env_info.CurrentStates,state_dim)# first env_states after rest function    
     
     #critic_next_states = np.array(env_info.CurrentStates)
     #critic_next_states = [critic_next_states[i][MaskList] for i in range(agentsNum)]
     #critic_next_states = [scaled_Data(critic_next_states[i]) for i in  range(agentsNum)]
     #critic_next_states = env_info.CurrentStates
     agent.reset() # rest noise function which affects on action
     #scores = np.zeros(agentsNum)
     scores, EQ_score = 0, 0
     t_step = 0
     speedDifSum = []
     MaxDistCause, env_done, stepConclude = False, False, False
     while stepConclude == False:
         overallStep += 1
         t_step += 1
         actor_critic_states = critic_next_states
         #actions = agent.act(actor_critic_states,i_episode)
         
         #if overallStep % 4 == 0 and overallStep < 5000:
             #actions = 1 * np.array([[np.clip(random.uniform(-1.0,1.0001),-1,1),np.clip(random.uniform(-1.0,1.0001),-1,1)]  for _ in range(6)])
             #randomAgentAct = random.randint(0,5)
         actions = agent.act(actor_critic_states,i_episode)

         if overallStep % Mode == 0 and Mode < 10:
            randomAgentAct = random.randint(0,5)
            randomAct = [np.clip(random.uniform(-1.0,1.0001),-1,1),np.clip(random.uniform(-1.0,1.0001),-1,1)]
            actions[randomAgentAct] = randomAct

         
         #actions = agent.act(actor_critic_states,i_episode)
      
         actions = actions.reshape(agentsNum,-1)
         
         #actions[:,0] = np.clip(actions[:,0], -1,.5)
         env_info = env.step(actions)      # send the action to the environment
         
         
         # define the speed diff criterion: summation of speed dif. from desired speed 
         speedDifSumStep = []
         [speedDifSumStep.append(abs(env_info.NextStates[i][1])) for i in range(agentsNum)]
         
         speedDifSum.append(sum(speedDifSumStep)/agentsNum)
         #########################
         
         if len(data) < 98000:
             data.append(env_info.CurrentStates)
             #data.append(env_info.NextStates)
         
         #########################
         
         actor_next_states = scaledDataFun(scaled_States,env_info.NextStates,state_dim)  # get the next state of the old_agentList            
         #actor_next_states = np.array(env_info.NextStates)
         #actor_next_states = [actor_next_states[i][MaskList] for i in range(agentsNum)]
         #actor_next_states = [scaled_Data(actor_next_states[i]) for i in  range(agentsNum)]
         #actor_next_states = env_info.NextStates
         W_rewards = env_info.rewards                   # get the reward of the old_agentList 
         #W_rewards = weightedReward(env, env_info.NextStates, agentsNum, env_info.rewards) # get the reward of the old_agentList
         #W_rewards = [sum(rewards) for i in range(agentsNum)]
         #W_rewards = agentsNum*[sum(W_rewards)]
         dones = env_info.dones                 # see if episode has finished 
         
         #critic_next_states = env_info.CurrentStates
         nextStates_act = scaledDataFun(scaled_States,env_info.NextStates,state_dim)
         #nextStates_act = np.array(env_info.NextStates)
         #nextStates_act = [nextStates_act[i][MaskList] for i in range(agentsNum)]
         #nextStates_act = [scaled_Data(nextStates_act[i]) for i in  range(agentsNum)]
         
         #*****************************************************
         for veh in env.vList:
             p_lead, p_fol = env.vehsInfo[veh]["prim_lead&fol_veh"]
             if p_lead != "None":
               lead_states = env.t_Observation[p_lead]
             else:lead_states = [env.t_Observation[veh][0]+2, 0, 0, 0, 4, 4, 0, 0]
             if p_fol != "None":
               fol_states = env.t_Observation[p_fol]
             else:fol_states = [env.t_Observation[veh][0]-2, 0, 0, 0, 4, 4, 0, 0]
                 
         
         
         # ****************************************************
         critic_next_states = scaledDataFun(scaled_States,env_info.CurrentStates,state_dim)
         #critic_next_states = np.array(env_info.CurrentStates)
         #critic_next_states = [critic_next_states[i][MaskList] for i in range(agentsNum)]
         #critic_next_states = [scaled_Data(critic_next_states[i]) for i in  range(agentsNum)]


         agent.step(actor_critic_states, actions, env_info.rewards, W_rewards, nextStates_act, nextStates_act, dones, beta) 

  
         scores += np.sum(env_info.rewards) # update the score
         EQ_score += np.sum(W_rewards)
         #states = env_info.CurrentStates  # set the current states of the env. as the states of agentList                                           
         #np.any(dones)
         #MaxDistCause = (abs(np.array(env_info.CurrentStates)[:,13:])>=MaxDistance).any()
         env_done = np.any(dones)
         if env.detect_collision()[0] == True : num_of_collision += 1
         if env_done:
             E_last_Step.append(t_step)
             terminate_Cause = env.doneCause
             stepConclude = True
             #break
             
         #Update Data for rescaling
         
         if  overallStep % 2000==0 and overallStep <= updateScales:
             reshaped_data_0= np.reshape(data, (-1,state_dim * agentsNum))
             scaled_States.fit(reshaped_data_0)
             #np.save(dataPath0+dataPath, data)
        

             #updateScales += 1
         if overallStep % 1000 == 0 and len(data) < 98000:
             with open(dataPath0_ +'/Data.pkl', 'wb') as file:
                 pickle.dump(data, file)
         
         if overallStep % 800 == 0 and Mode < 10: 
           Mode += 1
           print('\n ********* Mode:{}   overallStep:{})'.format(Mode, overallStep))  
                
       
     #mean_score = np.mean(scores)
     mean_score = scores
     AveReward_PerStep_In_EachEpisod = mean_score/t_step
     EQ_AveReward_PerStep_In_EachEpisod = EQ_score/t_step
     
     # Summation of abs(speedDiff criterion) at each episode
     EpisodMeanSpeedDiff = np.mean(speedDifSum)
     Ave_speedDev_In_EachEpisod.append(EpisodMeanSpeedDiff)

     AveReward_PerStep_In_EachEpisod_list.append(AveReward_PerStep_In_EachEpisod)
     EQ_AveReward_PerStep_In_EachEpisod_list.append(EQ_AveReward_PerStep_In_EachEpisod)              
     scores_deque.append(AveReward_PerStep_In_EachEpisod)
     EQ_scores_deque.append(EQ_AveReward_PerStep_In_EachEpisod)
     #all_scores.append(mean_score)   
     rolling_avg_100.append(np.mean(scores_deque))
     EQ_rolling_avg_100.append(np.mean(EQ_scores_deque))
     average_step.append(t_step)
     #print("\n--------------------------------------")
     #print('\nEpisode:{}  AveReward_PerStep_In_EachEpisod:{:.3f}  total_step:{}  TerminationCause: {}'.format(i_episode, AveReward_PerStep_In_EachEpisod ,t_step, terminate_Cause))        
     if i_episode % print_every == 0:
         #seed = random.randint(1,10)
         print("\n--------------------------------------")
         print('\nEpisode:{}   Episode_Score:[{:.3f},{}]   Average_Score:[{:.3f},{:.3f}]  Average Steps per Episod:{} total_step:{}  TerminationCause: {}  Collision_occurrence_Num: {}  E_last_Step: {} EpisodMeanSpeedDev: {}'.format(i_episode,AveReward_PerStep_In_EachEpisod,overallStep,
                                                                                                                                                                                                                                            np.mean(scores_deque),np.mean(EQ_scores_deque), np.mean(average_step),t_step,
                                                                                                                                                                                                                                            terminate_Cause, num_of_collision,E_last_Step, EpisodMeanSpeedDiff))
         
         time_now = datetime.now()                                                                                                                                                                                                                                 
         current_time = time_now.strftime("%H:%M:%S")
         print("\nThe current time:", current_time) 
         E_last_Step = []
         print(actions, env_info.rewards)
         print("\n--------------------------------------")
         [print([round(i,3) for i in env_info.CurrentStates[i]]) for i in range(agentsNum)]
         
         
     if i_episode % (.5*print_every) == 0:
        agent.save_checkpt(fileName+"model/")
        #seed = random.randint(1,10)
        print(f"\n###### collision Num. in last 5 Ep.: {num_of_collision} #########") 
        collision_data.append(num_of_collision)
        num_of_collision = 0
        pass

         
             
     if i_episode % (2*print_every) == 0:
         print(f"\n############### Total Step: {overallStep} ####################") 
         np.save(fileName+f"result/AveReward_PerStep_In_EachEpisod_list_Ep_T3",  AveReward_PerStep_In_EachEpisod_list)
         np.save(fileName+f"result/EQ_AveReward_PerStep_In_EachEpisod_list_Ep_T3",  EQ_AveReward_PerStep_In_EachEpisod_list)
         np.save(fileName+f"result/rolling_avg_100_Ep_T3",  rolling_avg_100)
         np.save(fileName+f"result/EQ_rolling_avg_100_Ep_T3",  EQ_rolling_avg_100)
         np.save(fileName+f"result/collision_data_T3",  collision_data)
         np.save(fileName+f"result/Ave_speedDev_In_EachEpisod_T3", Ave_speedDev_In_EachEpisod) 
         


loadMADDPG_path = r"C:\Users\MADDPG_LF\model/"
agent.load(loadMADDPG_path, device=device)
###########################************************************############################################
agent2 = MADDPG_Sep(state_dim, action_dim, agentsNum, random_seed, batch_size, tau, BUFFER_SIZE, device, 
                   UPDATE_EVERY, NOISE_DECAY_LIMIT)
loadMADDPG_path2 = r"C:\Users\MADDPG_LF\model/"
agent2.load(loadMADDPG_path2, device=device)
