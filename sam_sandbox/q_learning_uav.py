# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:42:15 2022

@author: skwel
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 07:57:56 2022

@author: skwel
"""

from UAV import UAV
from IPython import display
#from time import sleep
import numpy as np
import random
import matplotlib.pyplot as plt

#define and reset environment
env = UAV()

# Hyper parameters
alpha = 0.1 #learning rate
gamma = 0.6 #discount factor
epsilon = 0.1 #for epsilon-greedy
nEpisodes = 10000

#initialize q table
n_states = int(( (0.9-0.1) * float(env.observation_space.shape[0]) ) * float(env.observation_space.shape[1]))
n_actions = env.action_space.n
q_table = np.zeros([n_states, n_actions])

all_epochs, all_penalties = [list() for i in range(2)]

for i in range(1, nEpisodes):
    #reset and render
    state_grid = env.reset()
    state_x, state_y = env.drone.get_position()
    del_grid_y = env.y_max - env.y_min
    
    #translate position into row number in the q-table
    state_idx = state_x * del_grid_y + state_y
    
    epochs, penalties, reward = [0 for i in range(3)]
    done = False
    
    while not done:
        #epsilon greedy strategy for action decision
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample() #explore action space
        else:
            action = np.argmax(q_table[state_idx]) #eploit learned values

        #state transition
        state_dot_grid, reward, done, _ = env.step(action)
        state_dot_x, state_dot_y = env.drone.get_position()
        state_dot_idx = state_dot_x * del_grid_y + state_dot_y
        
        old_value = q_table[state_idx, action]
        next_max = np.max(q_table[state_dot_idx])
        
        #q update
        new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state_idx, action] = new_value
        
        #check reward and apply penalties if needed
        if reward == -10:
            penalties += 1
        
        state_idx = state_dot_idx
        epochs += 1
    
        #render drone/target
        env.render()
    
    if i % 100 == 0:
        display.clear_output(wait=True)
        print(f"Episode: {i}")
    
    #save off epoch array
    all_epochs.append(epochs)
    all_penalties.append(penalties)
        

#q-training finished
print("Training finished.\n")

#create plots
fig, ax = plt.subplots(2)
ax[0].plot(all_epochs)
ax[0].set_ylabel('# Epochs')
ax[0].set_xlabel("Episode #")

ax[1].plot(all_penalties)
ax[1].set_ylabel("# Penalties")
ax[1].set_xlabel("Episode #")















