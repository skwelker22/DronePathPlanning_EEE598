# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:27:13 2022

@author: whitchurch85 (Whitchurch Muthumani)
"""


from UAV import UAV

import gym
import sys
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class DQN(nn.Module):
    def __init__(self,img_height,img_width):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=9)
        self.fc2 = nn.Linear(in_features=9,out_features=9)
        self.fc3 = nn.Linear(in_features=9,out_features=9)
        self.out = nn.Linear(in_features=9, out_features=9)
        
    def forward(self,t):
        t = t.flatten(start_dim=1)
        #print(t.shape)
        t = F.relu_(self.fc1(t))
        t = F.relu_(self.fc2(t))
        t = F.relu_(self.fc3(t))
        t = self.out(t)
        return t
    
Experience = namedtuple('Experience',('state','action','next_state','reward'))

# e = Experience(2,3,1,4)
# print(e)

class ReplayMemory():
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count%self.capacity] = experience
        self.push_count +=1
        
    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self,batch_size):
        return len(self.memory) >= batch_size
    

class EpsilonGreedyStrategy():
    def __init__(self,start,end,decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    
    def get_exploration_rate(self,current_step):
        return self.end+(self.start-self.end)*\
    math.exp(-1. *current_step*self.decay)
    
    
class Agent():
    def __init__(self, strategy, num_actions,device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        
    
    def select_action(self,state,policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step +=1
        
        if rate > random.random():
            action = random.randrange(self.num_actions) # explore
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device) #exploit
            

class CartPoleEnvManager():

     
    def __init__(self,device,lamb, T0, alpha_low, beta, b1, b2, b3):
        self.device = device
        self.env = UAV(lamb, T0, alpha_low, beta, b1, b2, b3)
        self.env.reset(0)
        self.current_screen = None
        self.done = False
     
    
    def get_reward(self):
        return self.env.reward
        
    def check_obj_collided(self):
        return self.env.check_obj_collided()
    
    def check_target_collided(self):
        return self.env.check_target_collided()
    
    def reset(self,episode):
        self.env.reset(episode)
        self.current_screen = None
        
    def close(self):
        self.env.close()
        
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def num_actions_available(self):
        return self.env.action_space.n
    
    def take_action(self,action):
        state_dot_grid, reward, state_dot_high, reward_high, self.done = self.env.step(action.item())
        #_, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward+reward_high], device=self.device)
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2-s1
        
    
    def get_screen_height(self):
        screen = self.get_processed_screen()
        #print(screen.shape)
        return screen.shape[2]
    
    def get_screen_width(self):
        screen = self.get_processed_screen()
        #print(screen.shape)
        return screen.shape[3]
    
    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2,0,1))
        #screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)
    
    def crop_screen(self,screen):
        screen_height = screen.shape[1]
        
        #Strip off top and bottom
        top = int(screen_height*0.4)
        bottom = int(screen_height*0.8)
        screen = screen[:,top:bottom,:]
        return screen
    
    def transform_screen_data(self,screen):
        screen = np.ascontiguousarray(screen,dtype=np.float32)/255
        screen = torch.from_numpy(screen)
        
        resize = T.Compose([
            T.ToPILImage(),
            T.Resize((40,90)),
            T.ToTensor()
            ])
        return resize(screen).unsqueeze(0).to(self.device)
    

def extract_tensors(experiences):
    
    batch = Experience(*zip(*experiences))
    
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    
    return(t1,t2,t3,t4)


class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_current(policy_net,states,actions):
        #print(actions)
        #tensor_actions = torch.tensor(actions,dtype=torch.int64)
        #print(tensor_actions)
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
        # pcurrent_q_values = []
        # result_1 = [int(item) for item in actions.tolist()]
        # #print(result_1)
        # #print(actions.tolist())
        # #print(result_1)
        # #extracted_elements = [pcurrent_q_values[index] for index in result_1] 
        # #print(extracted_elements)
        # Counter = 0;
        # print(states.shape)
        # print(temp_result)
        # for item in states:
        #     print(item)
        #     index = result_1[Counter]
        #     flat = torch.flatten(item,start_dim=0)
        #     print(item.shape)
        #     temp_result = policy_net(flat)
        #     #print(temp_result)
        #     temp_result_extracted =  temp_result[index]
        #     #print(temp_result_extracted.item())
        #     pcurrent_q_values.append(temp_result_extracted.item())
        #     Counter = Counter+1
   
        
        # #print(actions.unsqueeze(-1))
        # #print(pcurrent_q_values.gather(dim=0,index=actions.unsqueeze(-1)))
           
        # return pcurrent_q_values
            
        #return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
        #return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    
    @staticmethod
    def get_next(target_net,next_states):
        final_state_locations = next_states.flatten(start_dim=1)\
            .max(dim=1)[0].eq(0).type(torch.bool)
        
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values
  
    
random.seed(10)
    
#Leaving these in here, for backwards compatibility for the  UAV class
# Hyper parameters
alpha = 0.2 #learning rate
alpha_low = 0.1
beta = 0.9
gamma_ex = 0.4 #discount factor
epsilon = 0.05 #for epsilon-greedy
lamb = 0.9 #discount factor
T0 = 100.0 #initial value of temp param



#beta terms for higher layer reward
b1 = 0.6
b2 = 0.3
b3 = 0.1
####The above values have been left behind for backwards compatibility and for less code rework

batch_size = 64 #128 #1024 #512 #256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.0001 #0.001 #0.0001
target_update = 10
memory_size = 100000
lr = 0.1
num_episodes = 610
max_iter = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = CartPoleEnvManager(device,lamb, T0, alpha_low, beta, b1, b2, b3)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy,em.num_actions_available(),device)
memory = ReplayMemory(memory_size)

policy_net = DQN(em.get_screen_height(),em.get_screen_width()).to(device)
target_net = DQN(em.get_screen_height(),em.get_screen_width()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(),lr=lr)

#print(policy_net)
#print(target_net)

episode_durations = []
episode_reward = []
cumulative_reward = []
cumulative_reward_tracker = 0

for episode in range(num_episodes):
    em.reset(episode)
    state = em.get_state()
    epochs = 0
    done = False
    print(episode)
    
    
    
    while not done:
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states,actions,rewards,next_states = extract_tensors(experiences)
            
            #print(states)
            #print(actions)
            
            current_q_values = QValues.get_current(policy_net,states,actions)
            next_q_values = QValues.get_next(target_net,next_states)
            
            #print(current_q_values)
            #print(next_q_values)
            
            #print(rewards)
            
            target_q_values = (next_q_values*gamma)+rewards
            #print(target_q_values)
            
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epochs += 1
            #print(epochs)
            if epochs > max_iter or em.check_obj_collided() or em.check_target_collided():
                done = True
                
                episode_reward.append(em.get_reward())
                
                cumulative_reward_tracker = cumulative_reward_tracker+em.get_reward()
                
                cumulative_reward.append(cumulative_reward_tracker)
                
                # if(em.check_target_collided()):
                #     torch.save(policy_net.state_dict(),"filename")
                #     sys.exit("End of game")
                epochs = max_iter
                
            em.render()
            
            
        if em.done:
            #episode_durations.append(timestep)
            #plot(episode_durations,100)
            break
        
        if episode == 10 or episode == 20 or episode == 30 or episode == 40 or episode == 50 or episode == 60\
           or episode == 70 or episode == 80 or episode == 90 or episode == 100 or episode == 110 or episode == 120\
           or episode == 120 or episode == 130 or episode == 140 or episode == 150 or episode == 160 or episode == 170\
           or episode == 180 or episode == 190 or episode == 200 or episode == 210 or episode == 220 or episode == 230\
           or episode == 230 or episode == 240 or episode == 250 or episode == 260 or episode == 270 or episode == 280\
           or episode == 290 or episode == 300 or episode == 310 or episode == 320 or episode == 330 or episode == 340\
           or episode == 350 or episode == 360 or episode == 370 or episode == 380 or episode == 390 or episode == 400\
           or episode == 410 or episode == 420 or episode == 430 or episode == 440 or episode == 450 or episode == 460\
           or episode == 470 or episode == 480 or episode == 490 or episode == 500 or episode == 510 or episode == 520\
           or episode == 520 or episode == 530 or episode == 540 or episode == 560 or episode == 580 or episode == 590\
           or episode == 600 or episode == 610 or episode == 620 or episode == 630 or episode == 640 or episode == 650\
           or episode == 660 or episode == 670 or episode == 680 or episode == 690 or episode == 700 or episode == 710\
           or episode == 720 or episode == 730 or episode == 740 or episode == 750 or episode == 760 or episode == 770\
           or episode == 780 or episode == 790 or episode == 800 or episode == 810 or episode == 820 or episode == 830\
           or episode == 840 or episode == 850 or episode == 860 or episode == 870 or episode == 880 or episode == 890\
           or episode == 900 or episode == 910 or episode == 920 or episode == 930 or episode == 940 or episode == 950\
           or episode == 960 or episode == 970 or episode == 980 or episode == 990 or episode == 1000:
            target_net.load_state_dict(policy_net.state_dict())
            
em.close()

#


#Plot the episode rewards
df = pd.DataFrame (episode_reward, columns = ['episode_reward'])
df[df < 0 ] = 0



df_110 = df.loc[0:110:1]
plt.figure(figsize=(10,10))
plt.plot(df_110,'r',label='Target Hits')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Frequency of successful hits [0-110 episodes]')
plt.legend()
plt.show()


df_210 = df.loc[110:210:1]
plt.figure(figsize=(10,10))
plt.plot(df_210,'r',label='Target Hits')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Frequency of successful hits [111-210 episodes]')
plt.legend()
plt.show()




df_310 = df.loc[211:310:1]
plt.figure(figsize=(10,10))
plt.plot(df_310,'r',label='Target Hits')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Frequency of successful hits [211-310 episodes]')
plt.legend()
plt.show()

df_410 = df.loc[311:410:1]
plt.figure(figsize=(10,10))
plt.plot(df_410,'r',label='Target Hits')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Frequency of successful hits [311-410 episodes]')
plt.legend()
plt.show()


df_510 = df.loc[411:510:1]
plt.figure(figsize=(10,10))
plt.plot(df_510,'r',label='Target Hits')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Frequency of successful hits [411-510 episodes]')
plt.legend()
plt.show()

df_510 = df.loc[511:610:1]
plt.figure(figsize=(10,10))
plt.plot(df_510,'r',label='Target Hits')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Frequency of successful hits [511-610 episodes]')
plt.legend()
plt.show()








    


        
        

    
            
        
        


    