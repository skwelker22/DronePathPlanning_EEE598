import numpy as np
import gym
import random
import time

#Get the object_handle for the FrozenLake environment:
env = gym.make("FrozenLake-v1")

#Query for the states and actions
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

#print(action_space_size)
#print(state_space_size)

#Create the Q-table:
q_table = np.zeros((state_space_size,action_space_size))
print(q_table)

#Global settings/Hyperparameters for the Reinforcement Learning game:
num_episodes = 10000 #numer of episodes
max_steps_per_episode = 100 #number of actions/steps per episode

learning_rate = 0.1 #we prioritze past data more
discount_rate = 0.99 # usual discount rate

exploration_rate = 1 # This is the starting rate, which we will decay
max_exploration_rate = 1 # upper ceiling 
min_exploration_rate = 0.01 # lower ceiling
exploration_recay_rate = 0.01 # rate of decay.
