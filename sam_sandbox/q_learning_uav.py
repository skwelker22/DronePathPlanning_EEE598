# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:42:15 2022

@author: skwel
"""

from UAV import UAV
# from IPython import display
#from time import sleep
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import signal

# Hyper parameters
alpha = 0.2 #learning rate
alpha_low = 0.1
beta = 0.9
gamma = 0.4 #discount factor
epsilon = 0.1 #for epsilon-greedy
lamb = 0.2 #discount factor
T0 = 100 #initial value of temp param
max_iter = 500
nEpisodes = int(750)

#beta terms for higher layer reward
b1 = 0.55
b2 = 0.35
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
n_states_high = 8 + 2 + 8 + 4
high_tuple = (8,2,8,4,n_actions)
#n_actions_high = env.action_space_high.n
q_table_high = np.zeros(high_tuple)

all_epochs, all_penalties, final_distance = [list() for i in range(3)]

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
        
        #epsilon decay factor as a function of episodes
        iterScale = 0.01 * i
        if U < (epsilon/iterScale):
            #explore action space
            action = env.action_space.sample()
            maxQ = float('inf')
        #check if the moving obstacle is in the sensor FOV
        elif env.drone.checkObsInFov() == True:
            action = np.argmax(q_table[state_x, state_y] + \
                               q_table_high[int(state_high[0]), int(state_high[1]), \
                                            int(state_high[2]), int(state_high[3]) ])
            
            maxQ = np.max(q_table[state_x, state_y] + \
                               q_table_high[int(state_high[0]), int(state_high[1]), \
                                            int(state_high[2]), int(state_high[3])])
        else:
            #action = env.genBoltzmann(q_table[state_x, state_y], U, i)
            action = np.argmax(q_table[state_x, state_y])
            maxQ = np.max(q_table[state_x, state_y])
        
        #set current Q value to display
        env.setQ(maxQ)
        
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
        
        if (env.drone.checkObsInFov() == True):
            #update higher layer q learning
            state_high[0] = state_dot_high[0]
            state_high[1] = state_dot_high[1]
            state_high[2] = state_dot_high[2]
            state_high[3] = state_dot_high[3]
        
        #update higher layer q-network
        if env.drone.checkObsInFov() == True:
            print("drone in fov now")
            print("state_dot_high_0 = " + str(state_dot_high[0]))
            print("state_dot_high_1 = " + str(state_dot_high[1]))
            print("state_dot_high_2 = " + str(state_dot_high[2]))
            print("state_dot_high_3 = " + str(state_dot_high[3]))
            
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
    final_distance.append(env.final_distance)
        

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

"""
Action (low) space defined as:
0: N, 1: NE, 2: E, 3: SE
4: S, 5: SW, 6: W, 7: NW
8: Do Nothing
"""
#show heatmap with q tables
plt.figure(333)
plt.imshow( np.log(np.abs(np.transpose(q_table[:,:,0]))), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('North')

plt.figure(334)
plt.imshow( np.log(np.abs(np.transpose(q_table[:,:,1]))), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('North East')

plt.figure(335)
plt.imshow( np.log(np.abs(np.transpose(q_table[:,:,2]))), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('East')

plt.figure(336)
plt.imshow( np.log(np.abs(np.transpose(q_table[:,:,3]))), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('South East')

plt.figure(337)
plt.imshow( np.log(np.abs(np.transpose(q_table[:,:,4]))), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('South')

plt.figure(338)
plt.imshow( np.log(np.abs(np.transpose(q_table[:,:,5]))), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('South West')

plt.figure(339)
plt.imshow( np.log(np.abs(np.transpose(q_table[:,:,6]))), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('West')

plt.figure(340)
plt.imshow( np.log(np.abs(np.transpose(q_table[:,:,7]))), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('North West')

plt.figure(341)
plt.imshow( np.log(np.abs(np.transpose(q_table[:,:,8]))), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Do Nothing')

#plot the max for the lower layer q-table
plt.figure(555)
plt.imshow( np.log(np.abs(np.transpose(np.max(q_table[:,:,:], axis=2)))), cmap='hot', interpolation='nearest' )
plt.colorbar()
plt.title("$log( |max(Q_{low})| )$")

#plot upper layer q table
"""
pltMax = np.max(q_table_high[:, 0, :, 0, :], axis=4)
plt.figure(556)
plt.imshow( np.log( np.abs( pltMax )), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("$log( |max(Q_{high})| )$")
"""

## MAF FINAL DISTANCE
mafLen = 100.0
imgDims = 32.0
bb = np.repeat(1./mafLen, mafLen)

normFinalDist = np.subtract(np.asarray(final_distance), imgDims)
filtFinalDist = signal.lfilter(bb, 1, normFinalDist)
plt.figure(445)
plt.xlabel("Episode #", size = 16)
plt.ylabel("Miss Distance [pixels]", size = 16)
plt.xticks(size = 15), plt.yticks(size=15)
plt.grid(True)
plt.plot( filtFinalDist, linewidth = 4 ) 
plt.show()

#filter penalties
successes = np.subtract(1, all_penalties)
successFilt = signal.lfilter(bb, 1, successes)
plt.figure(446)
plt.xlabel("Episode #", size = 16)
plt.ylabel("Success Rate", size = 16)
plt.xticks(size = 15), plt.yticks(size=15)
plt.grid(True)
plt.plot( successFilt, linewidth = 4 )
plt.show()







