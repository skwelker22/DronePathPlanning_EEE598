# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 07:57:56 2022

@author: skwel
"""

import gym
import numpy as np
from time import sleep
import random
import matplotlib.pyplot as plt
# from IPython.display import clear_output

#setup the environment (taxi)
env = gym.make("Taxi-v3").env

#debug/plot vars
plot_env = False

# Hyper parameters
alpha = 0.1 #learning rate
gamma = 0.6 #discount factor
epsilon = 0.1 #for epsilon-greedy
#nEpisodes = 100000
nEpisodes = 10000

#initialize q table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

all_epochs, all_penalties = [list() for i in range(2)]

for i in range(1, nEpisodes):
    #reset and render
    state = env.reset()

    epochs, penalties, reward = [0 for i in range(3)]
    done = False
    
    while not done:
        #epsilon greedy strategy for action decision
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample() #explore action space
        else:
            action = np.argmax(q_table[state]) #eploit learned values

        #state transition
        next_state, reward, done, info = env.step(action)
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        #q update
        new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        
        #check reward and apply penalties if needed
        if reward == -10:
            penalties += 1
        
        state = next_state
        epochs += 1
        
        #render and wait a moment to view results
        if plot_env:
            env.render()
            sleep(0.25)
        
        
    if i % 100 == 0:
        # clear_output(wait=True)
        print(f"Episode: {i}")
    
    #save off epoch array
    all_epochs.append(epochs)
    all_penalties.append(penalties)
        

print("Training finished.\n")

#create plots
fig, ax = plt.subplots(2)
ax[0].plot(all_epochs)
ax[0].set_ylabel('# Epochs')
ax[0].set_xlabel("Episode #")

ax[1].plot(all_penalties)
ax[1].set_ylabel("# Penalties")
ax[1].set_xlabel("Episode #")
















