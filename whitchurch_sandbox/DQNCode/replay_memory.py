# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 10:31:06 2022

@author: whitchurch85
"""
import random
import torch
import numpy as np

class ReplayMemory:
    def __init__(self,capacity):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        
        self.idx = 0
        
    def store(self,states,actions,next_states,rewards,dones):
        if len(self.states) < self.capacity:
            self.states.append(states)
            self.actions.append(actions)
            self.next_states.append(next_states)
            self.rewards.append(rewards)
            self.dones.append(dones)
        else:
            self.states[self.idx] = states
            self.actions[self.idx] = actions
            self.next_states[self.idx] = next_states
            self.rewards[self.idx] = rewards
            self.dones[self.idx] = dones
            
        'The modulu is taken to cycle back to 0, during overflow'
        'This will ensure we start rewriting older experience from the top'
        self.idx = (self.idx + 1) % self.capacity 
        
        
    'Code to sample from the replay memory, of a batch size'
    'This batch of randomly sampled replay memory items will serve as training data for the NN'
    
    def sample(self,batchsize,device):
        indices_to_sample = random.sample(range(len(self.states)), k=batchsize)
        
        states = torch.from_numpy(np.array(self.states)[indices_to_sample]).float().to(device)
        actions = torch.from_numpy(np.array(self.actions)[indices_to_sample]).float().to(device)
        next_states = torch.from_numpy(np.array(self.next_states)[indices_to_sample]).float().to(device)
        rewards = torch.from_numpy(np.array(self.rewards)[indices_to_sample]).float().to(device)
        done = torch.from_numpy(np.array(self.done)[indices_to_sample]).float().to(device)
        
        return states,actions,next_states,rewards,done
    
    def __len__(self):
        return len(self.states)
    
    

        
        
        
        
        
        
        
        
        