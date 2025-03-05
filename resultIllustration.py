import time
import os
import sys
from laneFreeUtils3_ import *
from utilis_illustration import *
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

sumoCfgFile = r"C:\Users\MADDPG_LF/circle_5.sumocfg"
import sumolib
import sumolib
import sys
from MultiAgentsCirEnv import *
from MultiAgentsDDPG_UnitedRBuffer import *
from MA_Seperated_st import *
#traci.close()
use_gui = True
veh_num = 74
env = laneFreeMagent_CirEnv(sumoCfgFile=sumoCfgFile, agentNum=veh_num, vehWidth=1.8, vehLength=3.2,
                            out_csv_name=None, use_gui=use_gui, maxTimeSim=1000, edgeID="e1", Init_minMaxSpeed=[20, 40],
                            Desired_minMaxSpeed=[20, 40],
                            longLatCollisionDis=[.4, .2], initPos0=100, max_step=1000, speedRewardCoef=1, Nudge_ind=1,
                            longLatSpeedRewardCoef=[.4, .4], print_initial_statuse=False, maxLongLatSpeedAcc=[4, 1.5],
                            n_intrusion=.25, wrongLatActPun=-1, discelTimeCoeff=1, reqDisLen=4,
                            MaxSpeedRiseCoef=1.1, single_agent=False, seed=0, NudgeCoef=.2, intrusionStressCoef=1,
                            MinspeedReward=-.2, VehSignalling=True, aCoef=1.38, tds=.5, posiAccFac=1,
                            jeLatCoef=1, JeLonCoef=1, circleLength=400, depSpeed=15, ignorCollision=False)

info = env.reset()
state_dim = len(info[0][0])
action_dim = env.action_space["agent_0"].shape[0]
max_action = env.action_space["agent_0"].high
random_seed = 0
batch_size = 64
tau = 0.01
UPDATE_EVERY = 3
BUFFER_SIZE = int(5e5)
NOISE_DECAY_LIMIT = 100
BETA_EPISODES_LIMIT = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataPath0_ = r"C:\Users\MADDPG_LF/"
with open(dataPath0_ + '/Data.pkl', 'rb') as file:
    data = pickle.load(file)
from sklearn import preprocessing

len(data[0][0]), len(data)
reshaped_data = np.reshape(data, (-1, state_dim * 1))
scaled_States1 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scaled_Data1 = scaled_States1.fit_transform(reshaped_data)
fileName = r"C:\Users\MADDPG_LF/"


def scaledDataFun2(scaled_States1, data, stateDim=8, agentsNum=1):
    scaledData = np.reshape(data, (-1, agentsNum * stateDim))
    scaledData = scaled_States1.transform(scaledData)
    scaledData = np.reshape(scaledData, (agentsNum, stateDim))
    return scaledData


agent2 = MADDPG_Sep(state_dim, action_dim, agentsNum, random_seed, batch_size, tau, BUFFER_SIZE, device,
                    UPDATE_EVERY, NOISE_DECAY_LIMIT)
loadMADDPG_path2 = r"C:\Users\MADDPG_LF\model/"
agent2.load(loadMADDPG_path2, device=device)
#####################################################################
#env.NudgeCoef = 1
resultDict = {}
# env.aCoef=2.0
# env.N_ind=1
# env.tds = 0.5
# env.discelTimeCoeff = 2
import matplotlib.pyplot as plt

env.DesiredSpeed = [25, 35]
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

prop = fm.FontProperties(family="Times New Roman")  # ,weight='bold'
plt.rcParams["font.family"] = "Times New Roman"
env.depSpeed = 25
colour = ["r", "g", "b", "c", "m", "y", "tab:purple", "tab:brown", "k", "pink"]
fig, ax = plt.subplots(1, 1, figsize=[10, 6])
############################ Fundamental Diag  #################################
corList = [(-2.47, 61.64), (9.46, 104.81), (66.94, 118.69), (115.31, 92.01), (116.96, 25.51), (34.39, -0.18)
    , (0.27, 24.95), (63.62, -0.58), (118.46, 60.12), (99.74, 112.69), (55.86, 126.87), (18.42, 104.75)
    , (66.99, -6.92), (-6.24, 77.98), (126.16, 65.15), (120.35, 72.98), (114.49, 81.68), (100.68, 105.13)
    , (77.80, 121.58), (43.95, 120.63), (29.96, 119.31), (3.10, 90.94), (-9.38, 56.15), (-2.00, 41.21)
    , (6.32, 28.16), (8.78, 18.89), (20.12, 4.14), (84.61, -1.91), (93.69, 8.11), (119.59, 41.21)
    , (14.26, 17.38), (37.71, -4.37), (43.57, -0.21), (77.04, 1.12), (105.22, 12.09)
    , (105.41, 19.84), (125.46, 48.77), (122.81, 80.54), (86.31, 114.96), (4.24, 86.22), (-5.78, 70.71)
    , (-1.15, 51.57), (-8.13, 47.55), (8.62, 13.53), (23.10, 9.51), (48.40, -4.27), (93.42, 2.53)
    , (113.14, 30.80), (108.25, 95.71), (36.54, 116.30), (19.26, 112.64), (10.54, 98.85)
    , (0.42, 75.65), (109.38, 101.70), (-4.63, 37.41), (50.82, -7.41), (120.36, 33.02)
    , (53.36, 119.89), (-1.63, 88.93), (1.50, 39.16), (117.44, 47.12), (66.05, 123.81), (89.92, 118.07)
    , (41.24, 123.52), (56.93, -4.27), (14.19, 12.11), (109.91, 16.72), (121.77, 50.69),
           (118.34, 82.81), (92.32, 109.01), (78.52, 115.95), (27.07, 111.71), (9.60, 94.24)
    , (-0.82, 82.95), (-7.98, 65.39)
           ]

veh_position = [
    (60.792869268333405, 121.4523024994819), (114.81155556268503, 83.03154272983352),
    (38.53231122016434, 116.62054418878424),
    (11.225343225341343, 21.46148562615021),
    (3.0134522276293243, 87.99947452570444),
    (48.36861864360226, -2.691658986272122),
    (37.67446917589946, -5.294347960789027),
    (117.35816441753215, 43.463394629697),
    (123.28677587879095, 47.070364113314184),
    (100.95997615352746, 7.885226558334222),
    (126.08295912576452, 57.9019978112503),
    (79.97751409487499, 116.04165208014916),
    (114.30147077591913, 90.84444082996298),
    (120.10793042106076, 88.72064194237417),
    (68.02417221618983, 126.40145973658313),
    (-0.45341711548253727, 73.18778934871546),
    (-0.027460380164120224, 39.41896282956566),
    (-0.4348010380473376, 31.32732101000904),
    (-2.5358520759415044, 63.73495373685244),
    (-4.6934970006928065, 36.66859515106568),
    (-9.058848659368081, 63.25729566860551),
    (66.2572750147467, -1.6920759388819437),
    (110.2693920317369, 95.08227918348224),
    (23.20504261202325, 8.587110234417803),
    (-1.8889266878948048, 87.50863421619728),
    (5.17690577938286, 30.874059318557432),
    (75.0630165689231, -5.994894587187277),
    (82.28596763582969, 2.2554759686870383),
    (84.29133964712786, -1.8248606709917048),
    (2.2193711939251535, 21.491211134451486),
    (119.10884141115318, 53.459799432931156),
    (103.56676043848469, 100.25430435222555),
    (9.84537860018738, 94.79160788472761),
    (109.82631811631147, 23.32558566382435),
    (90.86445270352046, 113.40066687643795),
    (9.597313491622305, 102.82980903148409),
    (38.768251993757374, 121.88219157928422),
    (22.315266968523876, 115.59182261683873),
    (-1.8220157118870532, 53.80766467227322),
    (10.503199677134539, 15.187430959976066),
    (73.30486652644584, 0.06133737928555494),
    (16.5267832377481, 7.584597546978251),
    (120.27281986721971, 35.56231168114608),
    (124.66839100704028, 73.65916281368206),
    (90.05451608839725, 6.000326019376853),
    (101.1766429825871, 112.19593225063267),
    (21.152505377185186, 106.79619108165828),
    (84.1496823735802, 120.01215253041094),
    (67.1427260955553, -8.788879306221832),
    (2.036269469455991, 81.92717635652467),
    (-0.5803747880990793, 47.37515142968234),
    (56.4792227862119, -2.132947219579084),
    (117.60942846052278, 24.647616497844044),
    (97.43752543823601, 106.37981421849838),
    (28.77750499434503, 111.57491735396526),
    (30.993159189532754, 4.863224603636027),
    (53.67224822855117, -7.114723800681978),
    (108.09565136947985, 13.8311469619087),
    (103.15745456310133, 17.016207557065933),
    (108.84798459917803, 101.74512265897471),
    (39.44479588129935, 1.2220497230818363),
    (17.118256399054292, 14.472737953100992),
    (-5.973060731933417, 49.286953036743405),
    (26.33326428698389, 1.4062702124823803),
    (113.45243552192058, 32.128986768194146),
    (120.02316691849082, 62.92884069006825),
    (53.75214962551968, 125.12033760837492),
    (120.00614657606971, 73.1618530694208),
    (49.11571036519812, 118.81088599918945),
    (15.85529087514377, 102.09731165798237),
    (70.15310728161018, 118.95226309866476),
    (2.2138680230138754, 96.63870595128525),
    (-4.873697806495759, 77.92718514860702),
    (95.91470289364811, 10.001719617901042),
    (95.62736787469278, 3.7174510928267184),
    (123.30, 65.03), (120.41, 79.64), (115.03, 96.18), (75.03, 122.33), (31.00, 118.10), (-6.69, 71.95), (3.88, 25.80),
    (31.97, 0.03), (60.06, -5.20), (17.55, 110.41)]

veh_pos_2 = [(21.931828820547153, 107.38286644332011),
             (123.27394531313239, 76.6799754851901),
             (40.53426469143681, 116.97729212999393),
             (81.25314895590806, 116.7158987719385),
             (92.69981093340765, 116.05374792803093),
             (34.2351317983334, 120.2126386380531),
             (2.6596806484555007, 34.49070272747559),
             (25.056249548780656, 8.232732118303046),
             (6.804121125885631, 18.083034321232212),
             (6.856423069087416, 26.252153481114505),
             (1.677384578614563, 21.908210271103286),
             (91.62235922341041, 111.14556655091265),
             (-7.119924860528572, 45.56222774866337),
             (106.43389560369361, 20.895176437705206),
             (76.71076272849574, 124.06069246646365),
             (60.57855616277352, -2.1163837145244875),
             (-7.32140757465548, 62.48136014377227),
             (33.40856438681999, 3.9823132606547547),
             (104.58965283654246, 11.124731535982356),
             (115.82841774101693, 25.645128085976662),
             (114.07952729527953, 18.85222807580371),
             (12.67068332436437, 8.758103901979904),
             (7.169258925593034, 98.66816996067263),
             (106.80889696050363, 96.68973837985203),
             (69.79832996092463, 121.07230309039461),
             (-2.872114036265694, 65.66351777878447),
             (119.41031495839073, 53.35186968065269),
             (44.89289401397286, 120.93273843431952),
             (-2.0390443818077655, 86.00662376599317),
             (18.11229160253074, 110.31120718167831),
             (9.33337127640594, 104.85107827146264),
             (12.51169266304385, 20.117247761177577),
             (28.408895327904894, 1.313453360404384),
             (43.07540528305416, 0.4998157816909945),
             (111.28269710874244, 27.355370436346274),
             (39.151926908393825, -4.154234650369402),
             (120.04212479239706, 62.98441783719032),
             (114.27578968341078, 98.65982648115619),
             (115.8129789948147, 35.78732764727278),
             (61.45221568884425, 119.67196421237387),
             (3.51705950637552, 84.47173782206316),
             (50.71228017530044, 125.11961280176048),
             (95.57837130230145, 4.014831846983995),
             (-0.4422252303811631, 43.42582456779813),
             (-1.7787311249125475, 56.693954020928935),
             (17.06813083632874, 13.65712905443218),
             (99.77478532579217, 13.90835910212106),
             (119.60083029708326, 72.0693527972869),
             (53.89880966507385, 119.81163405812873),
             (3.7677015555589253, 90.57774644925226),
             (70.90323273361892, -0.6738803493850369),
             (90.17795596451602, -1.6916177916644823),
             (126.3232275193941, 65.93902303143865),
             (125.09278873624272, 53.49564771938815),
             (63.100505867458345, 124.21673963113267),
             (90.33792475807334, 6.394111381874968),
             (79.6572342852243, -3.0834125622298076),
             (50.527638741584425, -1.5108316233134094),
             (-2.92142083286883, 34.919087926720614),
             (51.11094457926903, -6.57940455666445),
             (112.56588735869126, 87.61939662295134),
             (108.47671357397998, 102.1128283959258),
             (27.4537970136869, 117.11558883548913),
             (100.21428435592621, 105.64810127559707),
             (10.839264557013333, 95.95671558595784),
             (-0.019994374456544062, 75.92785313890572),
             (73.78444033852213, -7.168301388029106),
             (30.89174980840552, 113.74295003132677),
             (-6.4530039943245505, 75.61066461436336),
             (-3.281789171559467, 49.01722371273522),
             (119.03054460736261, 45.41934990685698),
             (120.22605963955841, 32.51563715916066),
             (121.51150176123286, 84.84808207437804),
             (104.60679757607919, 107.53647439123215),
             (82.97275948344588, 122.62704624877878),
             (38.804467212173755, 124.05428727680868),
             (14.35006056303483, 100.25242311021468),
             (117.38013141724979, 89.54789094543695),
             (-9.051049910821614, 53.19493724239079),
             (18.825272280980734, 6.9700226489694375),
             (62.61319761538194, -6.503956217768432),
             (83.06960634109791, 3.0290092150985983),
             (1.9058077248414653, 96.32853867638694),
             (116.18618374588804, 79.08943397802068),
             (125.90706363715898, 47.169024255418414)]
env.agentNum = veh_num
env.corList = corList[:veh_num]
# env.corList = veh_position[:80]
# env.corList = veh_pos_2#[:7]


len(env.corList)
env.agentNum, agentNum = len(env.corList), len(env.corList)
info = env.reset()
vehDictData = {}
##############################################################
for episode in range(1):
    # print(f"\nEpisode num {episode}")
    seed = random.randint(1, 100)
    env_info = env.reset(seed)
    # env.aCoef=1.38
    # env.stepNum=1
    # env.midAccel = 1
    state_dim = len(info[0][0])
    action_dim = env.action_space["agent_0"].shape[0]
    speed_0 = 0
    vehDataFrame = pd.DataFrame(env.vehsInfo)
    vehDataFrame.sort_values(by='desiredSpeed', axis=1, ascending=False, inplace=True)
    traci.vehicle.setColor(vehDataFrame.loc["desiredSpeed"].index[0], (255, 0, 0))
    traci.vehicle.setColor(vehDataFrame.loc["desiredSpeed"].index[1], (0, 250, 50))
    latMov = {}
    latLoc, longLoc = {}, {}
    for veh in env.vList:
        latMov[veh] = deque(maxlen=2)
        [latMov[veh].append(0) for _ in range(2)]
        latLoc[veh] = []
        longLoc[veh] = []
        vehDictData[veh] = []

    vehInt_num, meanSpeedList = 0, []
    steps = 400
    env.stepNum = 1
    for step in range(steps):
        ##########################
        vehs_old = traci.edge.getLastStepVehicleIDs('e1')
        ##########################
        CurrentStates = copy.deepcopy(env_info.CurrentStates)
        for i in range(agentNum):
            # CurrentStates[i][-1] = .3 * CurrentStates[i][-1] if CurrentStates[i][-1] <.2 and CurrentStates[i][1] > .1 else CurrentStates[i][-1]
            CurrentStates[i][1] = np.clip(1.0 * CurrentStates[i][1], -1, 1)  # -.5, .3)
            # CurrentStates[i][0] = np.clip(CurrentStates[i][0], 25, 35)
            # CurrentStates[i][-1] = np.clip(CurrentStates[i][-1], 0, .2)
            # CurrentStates[i][-2] = np.clip(CurrentStates[i][-2], 0, .5)
            # pass
        # for ag in range(6): CurrentStates[ag][0] = min(20, CurrentStates[ag][0])
        # IntVeh = "veh_5_21"
        # vehIndex = env.vList.index(IntVeh)
        # CurrentStates[vehIndex][-3] = 0
        # CurrentStates[vehIndex][-2] = max(CurrentStates[vehIndex][-2], .05 )

        # critic_next_states = scaledDataFun2(scaled_States,CurrentStates,state_dim, agentNum)# first env_states after rest function
        # actions = agent.act(critic_next_states,3000)
        # len(critic_next_states)
        actions = np.array([agent2.agents[4].act(np.array(scaledDataFun2(scaled_States1, state)), 3000)
                            for state in CurrentStates])
        # actions = np.reshape(actions, (-1,2))
        # actions = agent2.act(critic_next_states,3000)
        # actions[vehIndex]= [-.1, 0]
        ##################################################################
        ##################   Lat act revising   #########################
        """
  lead_veh = vehDict[egoVeh][1][2][1]
  if lead_veh != "N":
      lead_index = vehDict[lead_veh][7]
      latPosLead = LaneData[lead_index][5][1]
      if latAct > 0 and latPosLead - LaneData[ego_index][5][1] > 0.2: latAct = 0
  """
        ###########################################
        speedList = []
        for ind, veh in enumerate(env.vList):

            #########################
            leadVeh = env.vehsInfo[veh]["prim_lead&fol_veh"][0]
            if leadVeh != "None":
                latPos = env.tdict[veh][184]
                latPos_L = env.tdict[leadVeh][184]
                # lead_index = vehDict[lead_veh][7]
                # latPosLead = LaneData[lead_index][5][1]
                if actions[ind][0][1] > 0 and latPos_L - latPos > 2: actions[ind][0][1] = 0

            # latActSig = np.sign(actions[ind][1])
            latActSig = np.sign(actions[ind][0][1])
            # first_sign = actions[ind][1] >= 0
            first_sign = actions[ind][0][1] >= 0
            sameDir = all((lat_act >= 0) == first_sign for lat_act in latMov[veh])
            # sameDir = np.sign(latMov[egoVeh][-1]) == latActSig
            # sameDir = sameDir == latActSig
            # latActDiff = abs(latMov[veh][-1] - actions[ind][1])
            latActDiff = abs(latMov[veh][-1] - actions[ind][0][1])
            # latAct = actions[ind][1] if latActDiff > 4 or sameDir  else 0
            latAct = actions[ind][0][1] if latActDiff > 4 or sameDir else 0
            # latMov[veh].append(actions[ind][1])
            latMov[veh].append(actions[ind][0][1])
            latLoc[veh].append(env.tdict[veh][184])
            longLoc[veh].append(env.vehsInfo[veh]["dis2ref"])
            speedList.append(env_info.CurrentStates[ind][0])

        meanSpeedList.append(np.mean(speedList))
        ##################################################################
        """
  for ind, veh in enumerate(env.vList):
      speedDif = 0
      leadVeh, folVeh = env.vehsInfo[veh]["prim_lead&fol_veh"]
      if folVeh != "None":
          speedDif = env.tdict[veh][64] - env.tdict[folVeh][64]
      actions[ind][1] = 0 if speedDif > 0 and actions[ind][1] < 0 \
          else actions[ind][1]

      if leadVeh != "None":
          speedDif =  env.tdict[leadVeh][64] - env.tdict[veh][64]
          actions[ind][1] = 0 if speedDif > 0 and actions[ind][1] > 0 \
              else actions[ind][1]

  """
        actions = actions.reshape(agentNum, -1)

        for i, veh in enumerate(env.vList):
            """ 
   actions[i][0] = .1*actions[i][0] if env_info.CurrentStates[i][-2] >.6 and actions[i][0] > 0 \
       else actions[i][0] 
   if env_info.CurrentStates[i][-2] >.6: actions[i][0] = min(actions[i][0], 0)
   """

            # ********************   action revision   **************************
            if step > 0:
                if vehDictData[veh][-1][0] * actions[i][0] < 0: actions[i][0] *= .7
                if vehDictData[veh][-1][1] * actions[i][1] < 0: actions[i][1] *= .7
                actions[i][0] = np.clip(actions[i][0], -.5, .65)
                actions[i][1] = np.clip(actions[i][1], -.5, .6)

            ####################################################################
            vehDictData[veh].append([actions[i][0], actions[i][1], env_info.CurrentStates[i][-2],
                                     env_info.CurrentStates[i][-1]])

        ######################################################################

        [traci.vehicle.setSignals(egoVeh, 0) for egoVeh in env.vList]
        env_info = env.step(actions)

        # if env.detect_collision()[0] == True: colisionNum += 1
        ##########################
        veh_intr = []
        vehs_new = traci.edge.getLastStepVehicleIDs('e1')
        [veh_intr.append(veh) for veh in vehs_new if veh not in vehs_old]
        vehInt_num += len(veh_intr)

        ##########################
        ########################################################################

seconds = (step + 1) / 4
h_coef = 3600 / seconds
flow_h = h_coef * vehInt_num
density = round(2.5 * len(vehDataFrame.columns) / 3)
print(f"simulation steps: {step + 1}, traffic flow: {flow_h}")
#####################
speedList = []
CurrentStates = env_info.CurrentStates
M_Speed = round(np.mean([CurrentStates[i][0] for i in range(env.agentNum)]), 1)
D_Speed = round(vehDataFrame.loc['desiredSpeed'].mean(), 1)
Dev_Speed = round(vehDataFrame.loc['devFromDesiredSpeed'].mean(), 1)
means = np.mean(meanSpeedList)
print(f'\n step:{step + 1}, flow_h:{flow_h}, MD_DSp:{D_Speed}, Dev_Sp:{Dev_Speed}, \
meanSp:{means}, vehNum:{env.agentNum}, Dens:{density}')

resultDict[str(env.agentNum)] = {'step': step + 1, 'flow_h': flow_h, 'Dev_DSp': Dev_Speed, \
                                 'meanSp': means, 'vehNum': env.agentNum, 'Dens': density}
FundaD = {}
FundaD['20'] = [20, 28.65, 5157]
flow = flowCom(26.02, 68)
vehicle_ids = traci.vehicle.getIDList()
veh_position2 = []
[veh_position2.append(traci.vehicle.getPosition(veh)) for veh in vehicle_ids]
fundoment = pd.DataFrame(FundaD)
path = r"D:\Traffic Research Topics\LaneFreeEnv\circle\Papers\m_AGENdrl PAPER\Plots\circle/fundamental.csv"
fundoment.to_csv(path, index=False)
#######################
x, y = 2.5 * fundoment.iloc[0][::-1].values, fundoment.iloc[1][::-1].values
x_title, y_title = 'Density (vehicles / km)', 'Flow (vehicles / hour)'
x_title, y_title = 'Density (vehicles / km)', 'Speed (m/s)'
singlePlot(x, y, x_title, y_title, col='blue', yspeed=True, marker='o')

##############################  lateral heatMap   ########################################
l_dataframe = pd.DataFrame(latLoc)
# l_dataframe = l_dataframe.iloc[:400]

# l_dataframe = l_dataframe.groupby(np.arange(len(l_dataframe.index)) // 4, axis=0).mean()
la = np.linspace(-4.3, 4.3, 7)
for i in range(len(la)): la[i] = round(la[i], 2)
lat_dict = {}
for i in range(1, len(la)): lat_dict[(la[i - 1], la[i])] = []
for ind in range(len(l_dataframe)):
    lat_dict_0 = {}
    for i in range(1, len(la)): lat_dict_0[(la[i - 1], la[i])] = []
    for col in l_dataframe.columns:
        for L in lat_dict.keys():
            if l_dataframe.iloc[ind][col] >= L[0] and \
                    l_dataframe.iloc[ind][col] < L[1]:
                lat_dict_0[L].append(float(col.split('_')[-1]))
    [lat_dict[L].append(np.mean(lat_dict_0[L])) for L in lat_dict.keys()]

lat_dictDataframe = pd.DataFrame(lat_dict)
lat_dictDataframe = lat_dictDataframe.groupby(np.arange(len(lat_dictDataframe.index)) // 25, axis=0).mean()

Z = np.array([lat_dictDataframe[k] for k in lat_dict.keys()])
np.shape(Z)
# for L in lat_dict.keys():[z.append(des_sp) for des_sp in lat_dict[L]]
#  ################   plotting mesh grid  #####################
x_sample = len(l_dataframe)
X, Y = np.meshgrid(np.arange(0, x_sample, x_sample / len(lat_dictDataframe)),
                   [round(np.mean(k), 2) for k in lat_dict.keys()])
# Z = np.array(z)
import matplotlib.font_manager as fm

prop = fm.FontProperties(family="Times New Roman")  # ,weight='bold'
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=[10, 6])
# cmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])
cm = plt.cm.get_cmap('rainbow')  # 'gist_rainbow')#'rainbow') #'RdYlBu_r')# 'CMRmap')
im = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cm)  # , vmin=25, vmax=35) # cmap='viridis'

cbar = fig.colorbar(im, ax=ax)
# .set_label('Label', fontsize=18, fontfamily="Times New Roman")

cbar.set_label('Average desired speed (m/s)', fontsize=22, fontfamily="Times New Roman", rotation=90)
cbar.ax.tick_params(axis='both', which='major', labelsize=18)

ax.tick_params(axis='both', which='major', labelsize=18)
# ax.set_title('Vehicles destribution')
# ax.xticks((road_length))
# ax.yticks((la[1:]))
ax.set_xlabel("Time Step (0.25s)", fontproperties=prop, size=22)  # fontproperties=prop
ax.set_ylabel("Lateral position", fontproperties=prop, size=22)
ax.set_xticks(np.arange(0, 1000, 100))
############################
# ax.set_yticks(10.2)#Road_width)
# ax.set_yticklabels([k for k in lat_dict.keys()])
fig.show()
####################################################################################
plotFunc(env, vehDataFrame, latLoc, legend=False)
# [print(env.vehsInfo[vehID]["speedRewards"]) for vehID in env.vList]
# env.pre_t_Observation["veh_3_36"][-4:-2]
