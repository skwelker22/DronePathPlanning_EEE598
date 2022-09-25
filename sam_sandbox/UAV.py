# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:26:45 2022

Main Reference: https://blog.paperspace.com/creating-custom-environments-openai-gym/

@author: skwel
"""

import gym
from gym import Env, spaces
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import time
from math import exp, sqrt

class UAV(Env):
    
    def __init__(self, lamb, T0, alpha_low, beta):
        super(UAV, self).__init__()
        
        #boltzman probability parameters
        self.lamb = lamb
        self.T0 = T0
        self.alpha_low = alpha_low
        self.beta = beta
        self.obj_collided = False
        self.reward = 0
        
        """
        Action space defined as:
        0: N
        1: NE
        2: E
        3: SE
        4: S
        5: SW
        6: W
        7: NW
        8: Do Nothing
        """
        self.nActions = 9
        self.nObstacles = 3
        self.action_space = spaces.Discrete(self.nActions,)
        
        self.observation_shape = (400, 400, 3)
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
        
        #init episode #
        self.episode = 0
        
        
    def draw_elements_on_canvas(self):
        
        self.canvas = np.ones(self.observation_shape) * 1
        
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x,y = elem.x, elem.y
            self.canvas[y : y + elem_shape[1], x : x + elem_shape[0]] = elem.icon
            
        #text = 'Rewards: {}, Penalties: {}, Episode #: {}'.format(self.ep_return, self.penalties, self.episode)
        text = 'Rewards: {:.1f}, Ep: {:.0f}'.format(self.reward, self.episode)
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
        
        reward = 0
        penalties = 0
        
        #calculate current distance from uav to target
        x_drone, y_drone = self.drone.get_position()
        x_targ, y_targ = self.target.get_position()
        d_s_minus = self.calcDistance(x_drone, x_targ, y_drone, y_targ)
        
        """
        Action space defined as:
        0: N
        1: NE
        2: E
        3: SE
        4: S
        5: SW
        6: W
        7: NW
        8: Do Nothing
        """
        if action == 0:
            self.drone.move(0,-5)
        elif action == 1:
            self.drone.move(5,-5)
        elif action == 2:
            self.drone.move(5,0)
        elif action == 3:
            self.drone.move(5,5)
        elif action == 4:
            self.drone.move(0,5)
        elif action == 5:
            self.drone.move(-5,5)
        elif action == 6:
            self.drone.move(-5,0)
        elif action == 7:
            self.drone.move(-5,-5)
        elif action == 8:
            self.drone.move(0,0)
            
        
        #calculate the distance after action
        x_drone_dot, y_drone_dot = self.drone.get_position()
        d_s = self.calcDistance(x_drone_dot, x_targ, y_drone_dot, y_targ)
        
        #calculate the distance to each obstacle and sum inverses
        inverseObsSum = 0
        for obj in self.obs:
            x_obs, y_obs = obj.get_position()
            obs_i = self.calcDistance(x_drone_dot, x_obs, y_drone_dot, y_obs)
            inverseObsSum += 1/obs_i
            
        #calculate the "static obstacle" reward
        reward = self.alpha_low * (d_s_minus - d_s) - self.beta * inverseObsSum
            
        #check to see if target and drone have collided, if so, fin
        if self.has_collided(self.drone, self.target):
            done = True
            
        #check collisions with all obstacles
        self.obj_collided = False
        for obj in self.obs:
            if self.has_collided(self.drone, obj):
                self.obj_collided = True
                reward = reward - 1e5
                break
            
        #increment episodic return
        #self.ep_return += 1
        self.penalties += penalties
        
        #draw elements on canvas
        self.draw_elements_on_canvas()
        
        self.reward += reward
        
        return self.canvas, reward, done, []
    
    def reset(self, episode):
        self.reward = 0
        self.episode = episode
        
        #initialize the location of the drone on the grid
        x = random.randrange(int(self.observation_shape[0] * 0.05), 
                             int(self.observation_shape[0] * 0.10))
        y = random.randrange(int(self.observation_shape[1] * 0.15),
                             int(self.observation_shape[1] * 0.2))
        
        self.drone = Drone("drone", self.x_max, self.x_min, self.y_max, self.y_min)
        self.drone.set_position(x,y)
        
        #initialize the target location
        x_targ = int(self.observation_shape[0] * 0.90)
        y_targ = int(self.observation_shape[1] * 0.90)
        
        self.target = Target("target", self.x_max, self.x_min, self.y_max, self.y_min)
        self.target.set_position(x_targ, y_targ)
        
        ##init obstacles
        x_obs, y_obs = [np.zeros(self.nObstacles, dtype = int) for i in range(2)]
        x_obs[0] = int(self.observation_shape[0] * 0.60)
        y_obs[0] = int(self.observation_shape[1] * 0.70)
        x_obs[1] = int(self.observation_shape[0] * 0.20)
        y_obs[1] = int(self.observation_shape[1] * 0.44)
        x_obs[2] = int(self.observation_shape[0] * 0.70)
        y_obs[2] = int(self.observation_shape[1] * 0.20)
        
        self.obs = []
        
        for i in range(self.nObstacles):
            self.obs.append(Obstacle("obs" + str(i), self.x_max, self.x_min, self.y_max, self.y_min))
        
        pos_cnt = 0
        for obj in self.obs:
            obj.set_position(x_obs[pos_cnt], y_obs[pos_cnt])
            pos_cnt += 1
        
        #init elements
        self.elements = [self.drone, 
                         self.target, 
                         self.obs[0], self.obs[1], self.obs[2]]
        
        #init canvas
        self.canvas = np.ones(self.observation_shape) * 1
        
        #draw elements on canvas
        self.draw_elements_on_canvas()
        
        return self.canvas
    
    def genBoltzmann(self, Q_table, U, iterNum):
        #generate a bolzman random variable using the discrete inverse
        #transform technique and probability formulation discussed in the paper
        # DIT: https://home.csulb.edu/~tebert/teaching/lectures/552/variate/variate.pdf
        # BOLTZMAN: https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/Exploration_QLearning.pdf
        # https://web.cs.umass.edu/publication/docs/1991/UM-CS-1991-057.pdf
        
        #min to prevent numerical instability
        eps = 1e-9 #small number to prevent numerical instability
        
        #calculate boltzman "temperature" parameter
        #T_k = np.array((self.lamb ** iterNum) * self.T0, dtype = float)
        T_k = float((self.lamb ** iterNum) * self.T0)
        
        #calculate the probabilties for each action
        probs = []
        for a in range(self.nActions):
            
            #numerator
            #exp_num = exp(float(Q_table[a])/T_k)
            exp_num = float(Q_table[a])/T_k
            
            #reset cumSum
            exp_den = 0.0
            
            for i in range(self.nActions):
                    #exp_den += exp(float(Q_table[i])/T_k)
                    exp_den += float(Q_table[i])/T_k
                    
            #append probability
            probs.append(exp_num/( exp_den + eps ))
            
        #apply discrete inverse transform technique to generate action
        #generate CDF
        for a in range(self.nActions):
            q_i = sum(probs[0:a+1])
            #check if uniform random input is less than cumulated probability
            if U < q_i:
                return a
            else:
                return self.nActions - 1
     
    def calcDistance(self, x1, x2, y1, y2):
        return sqrt( (x1 - x2)**2 + (y1 - y2)**2 )
    
    def check_obj_collided(self):
        return self.obj_collided
        
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
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)
        
    def get_position(self):
        return (self.x, self.y)
    
    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y
        
        #saturate if we are hitting a wall
        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)
        
    
    def clamp(self, n, minn, maxn):
        max_n = min(maxn, n)
        min_n = max(max_n, minn)
        
        return min_n
        

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
        
class Obstacle(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Obstacle, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("obstacle.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))        