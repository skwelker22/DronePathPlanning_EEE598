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

# Hyper parameters
alpha = 0.3 #learning rate
alpha_low = 0.1
beta = 5
gamma = 0.6 #discount factor
epsilon = 0.07 #for epsilon-greedy
lamb = 0.9 #discount factor
T0 = 1e3 #initial value of temp param
max_iter = 500
nEpisodes = int(1e3)

#define and reset environment
env = UAV(lamb, T0, alpha_low, beta)

#initialize q table
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
n_dims = 2
q_table = np.zeros([n_states, n_states, n_actions])

all_epochs, all_penalties = [list() for i in range(2)]

for i in range(1, nEpisodes):
    #reset and render
    state_grid = env.reset(i)
    state_x, state_y = env.drone.get_position()
    
    #reinit current episode vars
    epochs, penalties, reward = [0 for i in range(3)]
    done = False
    
    while not done:
        #epsilon greedy strategy for action decision
        U = random.uniform(0,1)
        if U < epsilon:
            action = env.action_space.sample() #explore action space
        else:
            #action = env.genBoltzmann(q_table[state_x, state_y], U, i)
            action = np.argmax(q_table[state_x, state_y]) #eploit learned values

        #state transition
        state_dot_grid, reward, done, _ = env.step(action)
        state_dot_x, state_dot_y = env.drone.get_position()
        
        old_value = q_table[state_x, state_y, action]
        next_max = np.max(q_table[state_dot_x, state_dot_y])
        
        #q update
        new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state_x, state_y, action] = new_value
        
        #pass new states into old
        state_x, state_y = state_dot_x, state_dot_y
        epochs += 1
        
        #if we have reached max iterations or collided with an object
        if epochs > max_iter or env.check_obj_collided():
            done = True
            epochs = max_iter
            penalties += 1
    
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















