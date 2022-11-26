# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:13:27 2022

@author: skwel
"""

#majority of code taken from:
#https://keras.io/examples/rl/ddpg_pendulum/


#import all needed libraries
from UAV2 import UAV
from QUActionNoise import QUActionNoise
from Buffer import Buffer, update_target
from IPython import display
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy import signal
from actor_critic_policy import get_actor, get_critic, policy
from pdb import set_trace

# max iterations per episode
max_iter = 300
total_episodes = 500

#define number of states based on feature s_t
#s_t = [p_uav, (p_uav - p_target), v_uav] (x/y components)
n_states = 6

#sim time
dT = 1

#define bounds
angMin = -math.pi #maps to -pi
angMax = math.pi #maps to pi
action_bounds = [angMin, angMax]

#define and reset environment
env = UAV(n_states, action_bounds, dT)

#std_dev = 0.2
std_dev = 0.0
ou_noise = QUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

#create actor/critic models
actor_model = get_actor(n_states, angMax)
critic_model = get_critic(n_states, 1)

#instantiate actor/critic models for the target network
target_actor = get_actor(n_states, angMax)
target_critic = get_critic(n_states, 1)

#initialize target models to have same weights as actor/critic
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

#set learning rates
critic_lr = 0.002
actor_lr = 0.001

#set optimizers
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

#discount factor for future rewards
gamma = 0.99

#for target network update
tau = 0.005

#create replay buffer
buff_capacity = 50000
batch_size = 64
buffer = Buffer(buff_capacity, batch_size, n_states, 1)

#implement main training loop
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
for ep in range(total_episodes):

    #debug
    #set_trace()    

    prev_state = env.reset(ep)
    state_x, state_y = env.drone.get_position()
    episodic_reward = 0
    ep_iter = 0

    while True:

        #incremement iteration count
        ep_iter += 1
        
        #
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        #draw action according to the policy (using the actor network)
        action = policy(tf_prev_state, ou_noise, actor_model, action_bounds)
        
        # Recieve state and reward from environment.
        state, reward, done = env.step(action)
        
        #record the current MDP tuple into the replay memory buffer
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward
        
        #train the replay buffer
        buffer.learn(target_actor, target_critic, critic_model, 
                     critic_optimizer, actor_model, actor_optimizer)
        
        #update the target networks with actor/critic models
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break
        
        #check if uav collided with objects, if so end
        if env.check_obj_collided() or (ep_iter > max_iter):
            break
        
        #overwrite previous state with new state
        prev_state = state
        
        #render environment for visualization
        env.render()

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)



