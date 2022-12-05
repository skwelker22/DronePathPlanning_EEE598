# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:13:27 2022

@author: skwel
"""

#majority of code taken from:
#https://keras.io/examples/rl/ddpg_pendulum/

#import all needed libraries
from UAV2 import UAV
from IPython import display
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy import signal
from pdb import set_trace

#hack to fix plotting when tf is running
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#set print options
np.set_printoptions(suppress = True)

#class that implements Ornstei-Uhlenbeck process for generating noise in order
#to obtain better exploration from the Actor network
class QUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

#buffer class for replay memory
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64, 
                 num_states = 0, num_actions = 0
                 ):
        
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, 
              state_batch, 
              action_batch, 
              reward_batch, 
              next_state_batch):
        
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )
        
        #return gradients for monitoring
        return critic_grad, actor_grad

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        critic_grad, actor_grad = \
            self.update(state_batch, action_batch, reward_batch, next_state_batch)
            
        return critic_grad, actor_grad


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def get_actor(num_states, action_upper_bound):
    # Initialize weights between -3e-3 and 3-e3
    #last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    last_init = tf.random_uniform_initializer(minval=-0.000003, maxval=0.000003)
    #last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.00003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 3.14 for UAV
    outputs = outputs * action_upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic(num_states, num_actions):
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def policy(state, noise_object, action_bounds):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    
    """
    print("Sampled Action w/o noise [degrees] = " + str(math.degrees(sampled_actions.numpy())))
    print("Sampled Action w/o noise [rads] = " + str((sampled_actions.numpy())))
    print("Noise value = " + str(noise))
    """
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise
    
    #debug
    #print('Sampled Action from Actor Network = ' + str(math.degrees(sampled_actions[0])))

    # We make sure action is within bounds
    # action_bounds[0] = lower_bound
    # action_bounds[1] = upper_bound
    legal_action = np.clip(sampled_actions, action_bounds[0], action_bounds[1])
    
    #debug
    #print("Clipped Action from Actor Network = " + str(math.degrees(legal_action[0])))

    return [np.squeeze(legal_action)]

###################
## TRAIN NETWORK ##
###################
# max iterations per episode
max_iter = 400
total_episodes = 100

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

std_dev = math.pi/16
#std_dev = 0.2
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
#gamma = 0.99 #original
gamma = 0.98

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

all_penalties = []
final_distance = []

#create dictionary to save off actions for each episode
yaw_dict = {}
critic_dict = {}
actor_dict = {}

# Takes about 4 min to train
for ep in range(total_episodes):

    #debug
    #set_trace()    

    prev_state = env.reset(ep)
    state_x, state_y = env.drone.get_position()
    episodic_reward = 0
    ep_iter = 0
    penalties = 0
    yaw_ep = []

    while True:

        #incremement iteration count
        ep_iter += 1
        
        #
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        #draw action according to the policy (using the actor network)
        action = policy(tf_prev_state, ou_noise, action_bounds)
        
        #append current action to yaw list
        yaw_ep.append(math.degrees(float(action[0])))
        #print("current action chosen by policy is [deg] = " + str(math.degrees(float(action[0]))))
        #debug
        """
        print("current action chosen by policy is [rads] = " + str(action))
        print("current action chosen by policy is [deg] = " + str(math.degrees(float(action[0]))))
        print("Prev State = " + str(prev_state))
        """
        
        # Recieve state and reward from environment.
        state, reward, done = env.step(action)
        
        #record the current MDP tuple into the replay memory buffer
        """
        print("New State = " + str(state))
        print("Reward = " + str(reward))
        """
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward
        
        #train the replay buffer
        critic_grad, actor_grad = buffer.learn()
        
        #append critic/actor to dict
        critic_dict['ep'+str(ep)] = critic_grad
        actor_dict['ep'+str(ep)] = actor_grad
        
        #update the target networks with actor/critic models
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)
        
        #render environment for visualization
        env.render()

        # End this episode when `done` is True
        if done:
            break
        
        #check if uav collided with objects, if so end
        if env.check_obj_collided() or (ep_iter > max_iter) \
            or (episodic_reward < -350.0):
            penalties += 1
            break
        
        #overwrite previous state with new state
        prev_state[:] = state[:]

    ep_reward_list.append(episodic_reward)
    all_penalties.append(penalties)
    final_distance.append(env.final_distance)
    
    #save off yaw_ep to dict
    #yaw_dict['ep_' + str(ep)] = yaw_ep
    
    # Mean of last nAvg episodes
    nAvg = 100
    avg_reward = np.mean(ep_reward_list[-nAvg:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.figure(22)
plt.plot(avg_reward_list, linewidth = 4)
plt.xlabel("Episode", size = 16)
plt.ylabel("Avg. Epsiodic Reward", size = 16)
plt.xticks(size = 15), plt.yticks(size=15)
plt.grid(True)
plt.show()

#plot succes rate and miss distance
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




