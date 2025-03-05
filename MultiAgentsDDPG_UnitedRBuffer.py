# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:37:29 2022

@author: mehran
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from UnitedReplayBuffer import *
#from UnitedReplayBuffer import UnitedPrioritizedReplayBuffer


#########################
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.95            # discount factor
TAU = 0.05              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1
NO_UPDATES = 1



EPSILON = 1e-5 # small amount to avoid zero priority
ALPHA = 0.6 # to adjust weight of TD error. PER 1 = full prioritization, 0 = no prioritization
TOTAL_EPISODES=1000000
NOISE_SCALE = 1
#NOISE_DECAY_LIMIT=300
#BETA_EPISODES_LIMIT = 300

GREEDY_EPSILON = 0.01 #1.0 # exploration probability at start
GREEDY_EPSILON_MIN = 0.01  
GREEDY_EPSILON_DECAY= 0.0005 # exponential decay rate for exploration prob

#########################


# Actor/Critic model:
    
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor Model - Used to update policy."""

    def __init__(self, state_size, action_size, seed, fc1_units=1024, fc2_units=512, fc3_units=128,fc4_units=64,fc5_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            agentsNum (int): Number of agents in our environment
            max_action: Maximum lateral and longitudinal speed acceleration changes
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """

        super(Actor, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        """
        self.bn0 = nn.BatchNorm1d(state_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)
        self.bn4 = nn.BatchNorm1d(fc4_units)
        self.bn5 = nn.BatchNorm1d(fc5_units)
        """
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.fc5 = nn.Linear(fc4_units, fc5_units)
        #.............................................
        self.fc6 = nn.Linear(fc5_units, action_size)
        #self.l5 = nn.Linear(50, action_dim)	
        #self.max_action = max_action
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(*hidden_init(self.fc5))

        self.fc6.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""        
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(self.bn1(x)))
        x = F.relu(self.fc3(self.bn2(x)))
        x = F.relu(self.fc4(self.bn3(x)))
        x = F.relu(self.fc5(self.bn4(x)))
        return torch.tanh(self.fc6(x))
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return torch.tanh(self.fc6(x))
        


class Critic(nn.Module):
    """Critic Model - used to evaluate Value"""

    def __init__(self, state_size, action_size, agentsNum, seed, fc1_units=1024, fc2_units=512, fc3_units=256,
                 fc4_units=128,fc5_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            agentsNum (int): Number of agents in our environment
            max_action: Maximum lateral and longitudinal speed acceleration changes
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            .
            .
            ......
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        """
        self.bn0 = nn.BatchNorm1d(state_size*agentsNum + action_size*agentsNum)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)
        self.bn4 = nn.BatchNorm1d(fc4_units)
        """
        
        self.fc1 = nn.Linear(state_size*agentsNum + action_size*agentsNum, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)   
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units) 
        self.fc5 = nn.Linear(fc4_units, fc5_units) 
        self.fc6 = nn.Linear(fc5_units, 1)                
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(*hidden_init(self.fc5))
        self.fc6.weight.data.uniform_(-3e-3, 3e-3) 
        
    def forward(self, states, actions):# states=state Dimention * agentsNum & actions=action dimention *agentsNum
        """Build a critic (value) network that maps (state, action) pairs -> Q-values.
        states = np.array(1,state_Dimention*agentsNum)
        actions = np.array(1,action_Dimention*agentsNum)
        """ 
        xs = torch.cat((states, actions), dim=1)
        """
        xs = F.relu(self.fc1(xs))
        xs = F.relu(self.fc2(self.bn1(xs)))
        xs = F.relu(self.fc3(self.bn2(xs)))
        xs = F.relu(self.fc4(self.bn3(xs)))
        xs = F.relu(self.fc5(self.bn4(xs)))
        """
        xs = F.relu(self.fc1(xs))
        xs = F.relu(self.fc2(xs))
        xs = F.relu(self.fc3(xs))
        xs = F.relu(self.fc4(xs))
        xs = F.relu(self.fc5(xs))
        
        return self.fc6(xs)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, agentsNum, random_seed, batch_size=256, device='cpu',
                 NOISE_DECAY_LIMIT=600, discount=0.99, tau=0.01, LR_ACTOR=1e-4, LR_CRITIC=1e-3, Weight_Decay=1e-3,
                 BUFFER_SIZE=int(1e6)):
        self.device = device 
        #self.max_action = torch.tensor(max_action, device=self.device)
        #self.max_action_cpu = max_action

        self.state_size = state_size
        self.action_size = action_size        
        self.seed = random.seed(random_seed)
        self.batch_size = batch_size   
        self.t_step = 0
        self.seed = random.seed(random_seed)
        self.decay_step = 0
        self.discount = discount # GAMA
        self.tau = tau
        self.NOISE_DECAY_LIMIT = NOISE_DECAY_LIMIT
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, agentsNum, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, agentsNum, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=Weight_Decay)
        

        
        # update target networks
        #self.soft_update(self.critic_local, self.critic_target, 1.0)
        #self.soft_update(self.actor_local, self.actor_target, 1.0)
        
        # Noise process
        #self.noise = OUNoise()
        self.noise = OUNoise( action_size , random_seed)        
      
    
     
    
    def numpy_to_torch(self,data):
        return torch.from_numpy(data).float().to(self.device)
                                
    def act(self, state, episode_num, add_noise=True ):
        """Returns actions for given state as per current policy."""
        #state = np.reshape(state, (1,self.state_size))                  

        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():              
            action = self.actor_local(state).cpu().data.numpy()                                            
        self.actor_local.train()               
        if add_noise:                   
            action = self.add_random_noise(action, episode_num)
            #action +=  noise_decay * self.noise.sample() #self.add_random_noise()          
        return np.clip(action, -1, 1)

    def noise_decay_schedule(self,episode_num):  
        return max(0.0, NOISE_SCALE * (1 - (episode_num / self.NOISE_DECAY_LIMIT)))

    def add_random_noise(self, action, episode_num):  
        if episode_num < self.NOISE_DECAY_LIMIT:
            #return np.random.randn(1,self.action_size)
            action +=   self.noise_decay_schedule(episode_num) * self.noise.sample()
        return action
                
        
    
    def reset(self):        
        self.noise.reset()
    
   
                           

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
class MADDPGAgent():
    def __init__(self,state_size, action_size, agentsNum, random_seed, batch_size=256, tau=0.01, BUFFER_SIZE=int(1e6),
                 device="cpu", UPDATE_EVERY=0, NOISE_DECAY_LIMIT=600):
        self.device = device
        self.UPDATE_EVERY = UPDATE_EVERY       
        self.agents = [Agent(state_size, action_size, agentsNum, random_seed, batch_size,self.device, NOISE_DECAY_LIMIT) for _ in range(agentsNum)]
        self.agentsNum = agentsNum
        self.state_size = state_size
        self.batch_size = batch_size
        self.total_state_size = state_size*agentsNum 
        self.total_action_size = action_size * agentsNum
        self.t_step = 0
        self.tau = tau
        torch.autograd.set_detect_anomaly(False)

        #self.memory =  PrioritizedReplayBuffer(BATCH_SIZE, BUFFER_SIZE, random_seed)
        #self.memory = UnitedPrioritizedReplayBuffer(BUFFER_SIZE, self.batch_size, random_seed, device)
        self.memory = ReplayBuffer(self.batch_size, BUFFER_SIZE,  random_seed, device)
    def reset(self):
        for agent in self.agents:
            agent.reset()
    def numpy_to_torch(self,data):
        return torch.from_numpy(data).float().to(self.device)
    
    def act(self,actor_critic_state,episode_num):        
        #all_states = np.reshape(states, (1,self.total_state_size))
        #actions = [self.agents[agent_num].act(state, episode_num) for agent_num,state in enumerate(states)] 
        #######################################################################
        # states = [[states_agent0], [states_agent_1], ......] 
        #states must be a numpy array -> (np.array(states)) as observed states output of MultAgentSUMO_Env is a list         
        actions = [self.agents[agent_num].act(np.array(state), episode_num) for agent_num,state in enumerate(actor_critic_state)]        
        ### actions = [self.agents[agent_num].act(all_states, episode_num) for agent_num in range(self.agentsNum)] 
        actions = np.array(actions)  
        #actions = np.reshape(actions, (1,self.total_action_size))          
        return actions
        
    """
    def step(self,states,actions,rewards,next_states,dones,beta):   
        states = np.reshape(states, (1,self.total_state_size))
        actions = np.reshape(actions, (1,self.total_action_size))
        next_states = np.reshape(next_states,(1,self.total_state_size))
        for agent_num in range(self.agentsNum):
            self.agents[agent_num].step(states,actions,rewards[agent_num],next_states,dones[agent_num],agent_num,beta)
    """
    ######################################################
    def step(self, actor_critic_state, actions, rewards, W_rewards, actor_next_state, critic_next_state, dones, beta, GAMMA=0.95):
        actor_critic_state = np.reshape(actor_critic_state, (1,self.total_state_size))
        actions = np.reshape(actions, (1,self.total_action_size))
        actor_next_state = np.reshape(actor_next_state,(1,self.total_state_size))
        critic_next_state = np.reshape(critic_next_state,(1,self.total_state_size))
        done = np.any(dones)
        #T_rewards = np.mean(rewards)            
        # Save experience / reward   
        #error = self.calculate_error(actor_critic_state, actions, T_rewards, actor_next_state, critic_next_state, done) 
        self.memory.add(actor_critic_state, actions, W_rewards, rewards, actor_next_state, critic_next_state, done)                  
        #self.memory.add(actor_critic_state, actions, T_rewards, rewards, actor_next_state, critic_next_state, done, error)
        # Learn, if enough samples are available in memory  
        # Learn every UPDATE_EVERY time steps.
        self.t_step +=1
        
        if self.t_step %self.UPDATE_EVERY == 0:
            if self.memory.is_filled():                
                self.learn( GAMMA, beta, self.tau) 
    
    def calculate_error(self, states, actions, T_rewards, actor_next_states,critic_next_states, done):             
        [agent.actor_target.eval() for agent in self.agents]
        [agent.critic_target.eval() for agent in self.agents]
        [agent.critic_local.eval()  for agent in self.agents]
        
        states = self.numpy_to_torch(states)
        actions = self.numpy_to_torch(actions)
        actor_next_states = self.numpy_to_torch(actor_next_states)
        critic_next_states = self.numpy_to_torch(critic_next_states)

        
        with torch.no_grad():
            agents_next_action = []
            for agent_number, agent in enumerate(self.agents):
            
                first_state_index = self.state_size * (agent_number)
                last_state_index = self.state_size * (agent_number + 1)
                agents_next_action.append(agent.actor_target(actor_next_states[:,first_state_index:last_state_index]))
                #first_act_index = self.action_size * agent_number
                #last_act_index = self.action_size * (agent_number + 1)
            next_actions = torch.cat([action for action in agents_next_action], dim=1)
        
                    # action_nex is created by means of actor_traget network combined with the rest actions of other agents
                    #error = torch.tensor(np.zeros((self.batch_size,1)))
            error = 0
            for agent_number, agent in enumerate(self.agents):
                Q_targets_ = agent.critic_target(critic_next_states,next_actions)
                #Q_targets = rewards[agent_number] + (GAMMA * Q_targets_ * (1 - dones[agent_number]))
                Q_targets = T_rewards + (GAMMA * Q_targets_ * (1 - done))
    
                # Actual Q value based on reward rec'd at next step + future expected reward from Critic target network
                Q_expected = agent.critic_local(states, actions)
                # Compute Q targets for current states (y_i)             
                #Q_targets = np.asarray(reward, dtype=np.float32) + (GAMMA * Q_targets_next.data.numpy() * (1 - np.asarray(done, dtype=np.float32)))
                # Compute critic loss         
                #Q_expected = self.critic_local(self.numpy_to_torch(state), torch.from_numpy(action).float().to(device))        
                #error =  np.power((Q_expected - self.numpy_to_torch(Q_targets)).data.numpy(),2)            
                #error = ((Q_expected - Q_targets)**2).data.numpy()
                #huber_loss=torch.nn.SmoothL1Loss()        
                #error=huber_loss(Q_expected, Q_targets.detach())
                error += (torch.abs(Q_expected - Q_targets)).cpu().data.numpy()
        #error = ((error / self.agentsNum)).data.numpy()
        error = error / self.agentsNum

        [agent.actor_target.train() for agent in self.agents]
        [agent.critic_target.train() for agent in self.agents]
        [agent.critic_local.train()  for agent in self.agents]
        return error

    def learn(self, gamma, beta, tau):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """                        
                                                  
        #states, actions, rewards, next_states, dones = self.memory.sample()
        #experiences,indices,weights = self.memory.sample(self.batch_size)                 
        
       
        #states, actions, rewards, next_states, dones = experiences
        actor_critic_state, actions, W_rewards, rewards, actor_next_state, critic_next_state, done = self.memory.sample()
            #indices, weights = self.memory.sample(beta)
        # "states, actions, rewards, next_states, dones ,indices,weights" are outputs of torch_from_numpy executed in memory.sampl function
        # Get predicted next-state actions and Q values from target models
                            
        #actions_next = self.actor_target(next_states) 

        
        #-------------------- Calculation of Next_actions & Mu_actions ----------------------#
        """
        agents_Next_action, agents_mu_action = [], []
        for agent_number, agent in enumerate(self.agents):
            
            first_state_index = self.state_size * (agent_number)
            last_state_index = self.state_size * (agent_number + 1)
            # predict next actions by considering next states of each agent and actor_tatget
            agents_Next_action.append(agent.actor_target(actor_next_state[:,first_state_index:last_state_index]))
            # predict mu_action by current individual agent's states and respective local actors
            agents_mu_action.append(agent.actor_local(actor_critic_state[:,first_state_index:last_state_index]))
        Next_Action = torch.cat([action for action in agents_Next_action], dim=1)
        Mu_Action = torch.cat([action for action in agents_mu_action], dim=1)
        """
        #######################################################################################    

        
        Error = np.zeros((self.batch_size,1))
        
        ####################  Update Critics & Actors ########################
        Next_Action, agents_mu_action = self.update_Act_Fun(actor_next_state,actor_critic_state)
        for agent_number, agent in enumerate(self.agents):

 
           # ---------------------------- update critic ---------------------------- #
           with torch.no_grad():
               Q_target_next = agent.critic_target(critic_next_state, Next_Action)
           #reward =   torch.reshape(rewards[:,agent_number],(self.batch_size,1))
           #done = torch.reshape(dones[:,agent_number],(self.batch_size,1))
           """
           act_rewards = rewards[:,torch.arange(rewards.size(1)) != agent_number] 
           Rewards = .5 * rewards[:,agent_number] + .5 * torch.mean(act_rewards, dim=1)
           Q_Target_Next = Rewards.reshape(self.batch_size, 1) + gamma * Q_target_next * (1 - done)
           """
           Q_Target_Next = W_rewards[:,agent_number].reshape(self.batch_size,1) + gamma * Q_target_next * (1 - done)

           #Q_target_next = agent.critic_target(critic_next_state, Next_Action)
           #Q_Target_Next = rewards[agent_number] + (GAMMA * Q_target_next * (1 - dones[agent_number]))

           # Actual Q value based on reward rec'd at next step + future expected reward from Critic target network
           Q_expected = agent.critic_local(actor_critic_state, actions)
           errors = torch.abs(Q_expected - Q_Target_Next).cpu().data.numpy()
           Error += errors
           critic_loss =  F.mse_loss(Q_expected, Q_Target_Next.detach())
           #critic_loss = ((Q_expected - Q_Target_Next.detach()) ** 2).mean()
           #huber_loss=torch.nn.SmoothL1Loss()        
           #critic_loss=huber_loss(Q_expected, Q_targets.detach())
           agent.critic_optimizer.zero_grad()
           critic_loss.backward(retain_graph=True)
           #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
           agent.critic_optimizer.step()
           #agent.soft_update(agent.critic_local, agent.critic_target, tau)

            
           # ---------------------------- update actor ---------------------------- #
           #Mu_Action = copy.deepcopy(agents_mu_action)
           Mu_Action = [agents_mu_action[agentNum].detach() for agentNum in range(self.agentsNum)]
           Mu_Action[agent_number] = agents_mu_action[agent_number]
           Mu_Action = torch.cat([action for action in Mu_Action], dim=1).contiguous()
           # update actors
           """
           Mu_Action = agents_mu_action.detach()
           Mu_Action[:, agent_number] = agents_mu_action[:, agent_number]
           Mu_Action = self._flatten(Mu_Action)
           #Mu_Action = torch.cat([action for action in Mu_Action], dim=1)
           """
           # Compute actor loss 
           actor_loss = agent.critic_local(actor_critic_state, Mu_Action).reshape(-1) 
           actor_loss = -torch.mean(actor_loss)
           # Minimize the loss
           agent.actor_optimizer.zero_grad()
           actor_loss.backward(retain_graph=True)
           agent.actor_optimizer.step()
            
           #Next_Action, Mu_Action = self.update_Act_Fun(actor_next_state,actor_critic_state)
           
            # ----------------------- update target networks ----------------------- #
        for agent_number, agent in enumerate(self.agents): 
            agent.soft_update(agent.actor_local, agent.actor_target, tau)
            agent.soft_update(agent.critic_local, agent.critic_target, tau)
                 
        ######################################################################
        Error /= self.agentsNum
        #self.memory.update(indices, Error)
         
    def update_Act_Fun(self,actor_next_state,actor_critic_state):
        [agent.actor_target.eval() for agent in self.agents]
        #[agent.actor_local.eval()  for agent in self.agents]
        agents_Next_action, agents_mu_action = [], []
        
        for agent_number, Agent in enumerate(self.agents):
            
            first_state_index =Agent.state_size * (agent_number)
            last_state_index = Agent.state_size * (agent_number + 1)
            # predict next actions by considering next states of each agent and actor_tatget
            agents_Next_action.append(Agent.actor_target(actor_next_state[:,first_state_index:last_state_index]))
            # predict mu_action by current individual agent's states and respective local actors
            agents_mu_action.append(Agent.actor_local(actor_critic_state[:,first_state_index:last_state_index]))
        
         #######################################################################################
        #agents_Next_action = torch.stack(agents_Next_action).transpose(1, 0)
        #agents_mu_action = torch.stack(agents_mu_action).transpose(1, 0)
        Next_Action = torch.cat([action for action in agents_Next_action], dim=1).contiguous()
        #Next_Action = self._flatten(agents_Next_action.contiguous())
        #agents_mu_action = agents_mu_action.contiguous()
        #Mu_Action = torch.cat([action for action in agents_mu_action], dim=1)
        [agent.actor_target.train() for agent in self.agents]
        #[agent.actor_local.train()  for agent in self.agents]
        return Next_Action, agents_mu_action #Mu_Action
    def _flatten(self,tensor):
        b, n_agents, d = tensor.shape
        return tensor.view(b, n_agents * d)
    def save_checkpt(self,fileName):
        #fileName = "./MultiAgentsSumoEnv0/model/Muti_agentSUMO" -> i.e., Muti_agentSUMO_critic_0
        for agent_number in range(self.agentsNum):
            #torch.save(self.agents[agent_number].actor_local.state_dict(), f'checkpoint_actor_{agent_number}')
            #torch.save(self.agents[agent_number].critic_local.state_dict(), f'checkpoint_critic_{agent_number}')
            
            torch.save(self.agents[agent_number].critic_local.state_dict(), fileName  + f"critic_{agent_number}")
            torch.save(self.agents[agent_number].critic_optimizer.state_dict(), fileName + f"critic_optimizer_{agent_number}")
            
            torch.save(self.agents[agent_number].actor_local.state_dict(), fileName + f"actor_{agent_number}")
            torch.save(self.agents[agent_number].actor_optimizer.state_dict(), fileName + f"actor_optimizer_{agent_number}")

        """    
        #torch.save(self.agents[0].actor_local.state_dict(), 'checkpoint_actor1_preplay.pth')
        #torch.save(self.agents[0].critic_local.state_dict(), 'checkpoint_critic1_preplay.pth')
        #torch.save(self.agents[1].actor_local.state_dict(), 'checkpoint_actor2_preplay.pth')
        #torch.save(self.agents[1].critic_local.state_dict(), 'checkpoint_critic2_preplay.pth')
        """
        # fileName = path to save MADDPG model data like: "./models/MADDPG"
        # f"./MultiAgentsSumoEnv0/result/{file_name}.npy"
        # f"./MultiAgentsSumoEnv0/model/{file_name}"
        
    def load(self, fileName, device="cpu"):
        # fileName = "./MultiAgentsSumoEnv0/model/Muti_agentSUMO"
        for agent_number in range(self.agentsNum):
            
            self.agents[agent_number].critic_local.load_state_dict(torch.load(fileName + f"critic_{agent_number}",device))
            self.agents[agent_number].critic_optimizer.load_state_dict(torch.load(fileName + f"critic_optimizer_{agent_number}",device))
            self.agents[agent_number].critic_target.load_state_dict(torch.load(fileName + f"critic_{agent_number}",device))
    
            self.agents[agent_number].actor_local.load_state_dict(torch.load(fileName + f"actor_{agent_number}" ,device))
            self.agents[agent_number].actor_optimizer.load_state_dict(torch.load(fileName + f"actor_optimizer_{agent_number}",device))
            self.agents[agent_number].actor_target.load_state_dict(torch.load(fileName + f"actor_{agent_number}" ,device))














