# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:26:45 2022

@author: skwel
"""

import gym
from gym import Env, spaces
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import time

class UAV(Env):
    
    def __init__(self):
        super(UAV, self).__init__()
        self.action_space = spaces.Discrete(6,)
        
        self.observation_shape = (600, 800, 3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape),
                                            high = np.zeros(self.observation_shape), 
                                            dtype = np.float16)
        
        #create canvas for image
        self.canvas = np.ones(self.observation_shape) * 1
        
        self.elements = []
        
        #set limits of the environment
        self.y_min = int (self.observation_shape[0] * 0.1)
        self.x_min = 0
        self.y_max = int (self.observation_shape[0] * 0.9)
        self.x_max = self.observation_shape[1]
        
        #init penalties
        self.penalties = 0
        
    def draw_elements_on_canvas(self):
        
        self.canvas = np.ones(self.observation_shape) * 1
        
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x,y = elem.x, elem.y
            self.canvas[y : y + elem_shape[1], x : x + elem_shape[0]] = elem.icon
            
        text = 'Rewards: {}, Penalties: {}'.format(self.ep_return, self.penalties)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        self.canvas = cv2.putText(self.canvas, text, (10,20), 
                                  font, 0.8,
                                  (0,0,0), 1, 
                                  cv2.LINE_AA)
        
    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], \
            "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)
        
        elif mode == "rgb_array":
            return self.canvas
        
    #dictionary that maps numbers to "action definitions"
    def get_action_meanings(self):
        return {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Do Nothing"}
    
    #check if two point objects have collided
    def has_collided(self, elem1, elem2):
        x_col = False
        y_col = False
        
        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()
        
        if 2 * abs(elem1_x - elem2_x) <= (elem1.icon_w + elem2.icon_w):
            x_col = True
            
        if 2 * abs(elem1_y - elem2_y) <= (elem1.icon_h + elem2.icon_h):
            y_col = True
            
        if x_col and y_col:
            return True
        
        return False

    def step(self, action):
        
        done = False
        
        assert self.action_space.contains(action), "Invalid Action"
        
        reward = 1
        penalties = 0
        
        if action == 0:
            reward, penalties = self.drone.move(0, 5)
        elif action == 1:
            reward, penalties = self.drone.move(0,-5)
        elif action == 2:
            reward, penalties = self.drone.move(5,0)
        elif action == 3:
            reward, penalties = self.drone.move(-5,0)
        elif action == 4:
            reward, penalties = self.drone.move(0,0)
            
        #check to see if target and drone have collided, if so, fin
        if self.has_collided(self.drone, self.target):
            done = True
            reward = 10
            
        #increment episodic return
        self.ep_return += 1
        self.penalties += penalties
        
        #draw elements on canvas
        self.draw_elements_on_canvas()
        
        return self.canvas, reward, done, []
    
    def reset(self):
        self.ep_return = 0
        
        #initialize the location of the drone on the grid
        x = random.randrange(int(self.observation_shape[0] * 0.05), 
                             int(self.observation_shape[0] * 0.10))
        y = random.randrange(int(self.observation_shape[1] * 0.15),
                             int(self.observation_shape[1] * 0.2))
        self.drone = Drone("drone", self.x_max, self.x_min, self.y_max, self.y_min)
        self.drone.set_position(x,y)
        
        #initialize the target location
        x_targ = int(self.observation_shape[0] * 0.95)
        y_targ = int(self.observation_shape[1] * 0.95)
        
        self.target = Target("target", self.x_max, self.x_min, self.y_max, self.y_min)
        self.target.set_position(x_targ, y_targ)
        
        #init elements
        self.elements = [self.drone, self.target]
        
        #init canvas
        self.canvas = np.ones(self.observation_shape) * 1
        
        #draw elements on canvas
        self.draw_elements_on_canvas()
        
        return self.canvas
        
class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name
        
    def set_position(self, x, y):
        self.x, _, _ = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y, _, _ = self.clamp(y, self.y_min, self.y_max - self.icon_h)
        
    def get_position(self):
        return (self.x, self.y)
    
    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y
        
        self.x, clamp_reward_x, penalties_x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y, clamp_reward_y, penalties_y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)
        
        return max(clamp_reward_x, clamp_reward_y), max(penalties_x, penalties_y)
    
    def clamp(self, n, minn, maxn):
        
        clamp_reward, penalties = [0 for i in range(2)]
        
        max_n = min(maxn, n)
        min_n = max(max_n, minn)
        
        #if we hit a wall, decrement the reward
        if max_n == maxn or min_n == minn:
            clamp_reward = -10
            penalties = 1
        
        return min_n, clamp_reward, penalties
        

class Drone(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Drone, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("drone_basic.png") / 255.0
        self.icon_w = 64
        self.icon_h = 64
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        
class Target(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Target, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("target_basic.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        
        