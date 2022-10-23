# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:42:15 2022

@author: skwel
@modified by Jian Meng 
"""

from UAV import UAV
# from IPython import display
#from time import sleep
import numpy as np
import random
import matplotlib.pyplot as plt

# Hyper parameters
alpha = 0.2 #learning rate
alpha_low = 0.1
beta = 0.9
gamma = 0.4 # discount factor
epsilon = 0.05 #for epsilon-greedy
lamb = 0.9 #discount factor
T0 = 100.0 #initial value of temp param
max_iter = 1000
nEpisodes = int(1e2)

#beta terms for higher layer reward
b1 = 0.6
b2 = 0.3
b3 = 0.1

#define and reset environment
env = UAV(lamb, T0, alpha_low, beta, b1, b2, b3)

#initialize q table
n_states_x = env.observation_space.shape[0]
n_states_y = env.observation_space.shape[1]
n_actions = env.action_space.n
n_dims = 2
q_table = np.zeros([n_states_x, n_states_y, n_actions])

#define dynamic obstacle q table
n_states_high = 8 + 2 + 8 + 8
high_tuple = (8,2,8,8,9)
#n_actions_high = env.action_space_high.n
q_table_high = np.zeros(high_tuple)

all_epochs, all_penalties, final_distance, all_reward = [list() for i in range(4)]

for i in range(1, nEpisodes+1):
    #reset and render
    state_grid, state_high = env.reset(i)
    state_x, state_y = env.drone.get_position()    
    
    #reinit current episode vars
    epochs, penalties, reward = [0 for i in range(3)]
    done = False
    
    while not done:
        
        #epsilon greedy strategy for action decision
        U = random.uniform(0,1)
        if U < epsilon:
            #explore action space
            action = env.action_space.sample() 
        #check if the moving obstacle is in the sensor FOV
        elif env.drone.checkObsInFov() == True:
            action = np.argmax(q_table[state_x, state_y] + \
                               q_table_high[state_high[0], state_high[1], state_high[2], state_high[3]])
        else: 
            # action = np.argmax(q_table[state_x, state_y])
            action = env.genBoltzmann(q_table[state_x, state_y], U, i)
        
        #state transition
        state_dot_grid, reward, state_dot_high, reward_high, done = env.step(action)
        
        #get new states after action has taken place
        state_dot_x, state_dot_y = env.drone.get_position()
        
        old_value = q_table[state_x, state_y, action]
        next_max = np.max(q_table[state_dot_x, state_dot_y])
        
        #q update
        new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state_x, state_y, action] = new_value
        
        #pass new states into old
        state_x, state_y = state_dot_x, state_dot_y
        
        #update higher layer q-network
        if env.drone.checkObsInFov() == True:
            next_max_high = np.max(q_table_high[state_dot_high[0], state_dot_high[1], \
                                                state_dot_high[2], state_dot_high[3]])
            #equation 16
            new_value_high = reward_high + gamma * next_max_high
            q_table_high[state_dot_high[0], state_dot_high[1], \
                         state_dot_high[2], state_dot_high[3], action] = new_value_high
        
        epochs += 1
        
        #if we have reached max iterations or collided with an object
        if epochs > max_iter or env.check_obj_collided():
            done = True
            epochs = max_iter
            penalties += 1
    
        #render drone/target
        env.render()
    
    if i % 100 == 0:
        print(f"Episode: {i}")
    
    #save off epoch array
    all_epochs.append(epochs)
    all_penalties.append(penalties)
    all_reward.append(reward)
    final_distance.append(env.final_distance)


#q-training finished
print("Training finished.\n")
all_penalties = np.array(all_penalties)
spars = len(all_penalties[all_penalties==0])/len(all_penalties)
print("Total number of zero penalities: {} after {} episode, spars={}".format(len(all_penalties[all_penalties==0]), nEpisodes, spars))

#create plots
fig, ax = plt.subplots(3)
ax[0].plot(all_epochs)
ax[0].set_ylabel('# Epochs')
ax[0].set_xlabel("Episode #")

ax[1].plot(all_penalties)
ax[1].set_ylabel("# Penalties")
ax[1].set_xlabel("Episode #")

ax[2].plot(all_reward)
ax[2].set_ylabel("Total Reward")
ax[2].set_xlabel("Episode #")

fig.savefig(f"./results_alpha{alpha}_alpha_low{alpha_low}_beta{beta}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)















