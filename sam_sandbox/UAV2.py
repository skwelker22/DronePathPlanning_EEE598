# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 15:08:01 2022

@author: skwel
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:26:45 2022

Main Reference: https://blog.paperspace.com/creating-custom-environments-openai-gym/

@author: skwel
"""

from gym import Env, spaces
import numpy as np
#import imutils
import cv2
import random
import math
from math import exp, sqrt, atan2, pi, sin, cos

class UAV(Env):
    
    def __init__(self, num_states, action_bounds, dT):
        super(UAV, self).__init__()
        
        #set run time
        self.dT = dT
        
        #algorithm properties
        self.obj_collided = False
        self.cum_reward = 0
        self.final_distance = None
        self.distThreshold = 64 #starts givng + 10 reward when uav is within 64 pixels
        self.state_feature = np.zeros((num_states,))
        
        """
        Action space defined as: yaw angle [-pi, pi]
        """
        self.angMin = action_bounds[0]
        self.angMax = action_bounds[1]
        
        #define number of obstacles and observation space
        self.nObstacles = 3
        
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
        text = 'R: {:.1f}, Ep: {:.0f}'.format(self.cum_reward, self.episode)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        self.canvas = cv2.putText(self.canvas, text, (10,20), 
                                  font, 0.8,
                                  (0,0,0), 1, 
                                  cv2.LINE_AA)
        
    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], \
            "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Simple Drone Simulator", self.canvas)
            cv2.waitKey(10)
        
        elif mode == "rgb_array":
            return self.canvas
    
    #check if two point objects have collided
    def has_collided(self, elem1, elem2):
        x_col = False
        y_col = False
        
        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()
        
        #print('Obs x = ' + str(elem2_x))
        #print('Obs y = ' + str(elem2_y))
        
        if 2 * abs(elem1_x - elem2_x) <= (elem1.icon_w + elem2.icon_w):
            x_col = True
            
        if 2 * abs(elem1_y - elem2_y) <= (elem1.icon_h + elem2.icon_h):
            y_col = True
            
        if x_col and y_col:
            return True
        
        return False
    
    def reset(self, episode):
        self.cum_reward = 0
        self.episode = episode
        
        #initialize the location of the drone on the grid
        x = random.randrange(int(self.observation_shape[0] * 0.05), 
                             int(self.observation_shape[0] * 0.10))
        y = random.randrange(int(self.observation_shape[1] * 0.15),
                             int(self.observation_shape[1] * 0.2))
        
        self.drone = Drone(x, y,"drone", self.x_max, self.x_min, self.y_max, self.y_min, self.dT)
        
        #initialize drone states
        self.drone.state[0] = x
        self.drone.state[1] = y
        self.drone.state[2] = 1   #1 pixel / second
        self.drone.state[3] = 1   #1 pixel / second
        self.drone.state[4] = 0.5 #0.5 pixel / second/ second
        self.drone.state[5] = 0   
        
        #initialize the target location
        x_targ = int(self.observation_shape[0] * 0.90)
        y_targ = int(self.observation_shape[1] * 0.90)
        
        self.target = Target("target", self.x_max, self.x_min, self.y_max, self.y_min, self.dT)
        self.target.set_position(x_targ, y_targ)
        
        #initialize final distance
        self.final_distance = self.calcDistance(x, x_targ, y, y_targ)
        
        #calcualte initial state feature
        self.state_feature[0] = self.drone.state[0] #initial uav position (x)
        self.state_feature[1] = self.drone.state[1] #initial uav position (y)
        self.state_feature[2] = self.drone.state[0] - x_targ #initial relative position (x)
        self.state_feature[3] = self.drone.state[1] - y_targ #initial relative position (y)
        self.state_feature[4] = self.drone.state[2] # initial velocity (x)
        self.state_feature[5] = self.drone.state[3] # initial velocity (y)
        
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
            self.obs.append(Obstacle("obs" + str(i), self.x_max, self.x_min, self.y_max, self.y_min, self.dT))
        
        pos_cnt = 0
        for obj in self.obs:
            obj.set_position(x_obs[pos_cnt], y_obs[pos_cnt])
            pos_cnt += 1
        
        #init elements
        self.elements = [self.drone, 
                         self.target, 
                         self.obs[0], self.obs[1], self.obs[2]
                         ]
        
        #init canvas
        self.canvas = np.ones(self.observation_shape) * 1
        
        #draw elements on canvas
        self.draw_elements_on_canvas()
        
        return self.state_feature

    def step(self, action):
        
        #reset done, current rewards and penalties
        done = False
        reward_t = 0
        penalties = 0
        
        #check if the requested action is within bounds
        assert action[0] <= self.angMax and action[0] >= self.angMin, "Invalid Action"
        
        #propogate the drone given current yaw angle
        """
        Action space defined as:
        yaw angle in the interval [-pi:pi/15:pi]
        """
        print("Action in [degrees] = " + str(math.degrees(action[0])))
        self.drone.move(action[0])
        
        #create new state feature vector
        x_targ, y_targ = self.target.get_position()
        x_drone_dot, y_drone_dot = self.drone.get_position()
        vx_drone_dot, vy_drone_dot = self.drone.get_velocity()
        
        #set state features
        self.state_feature[0] = x_drone_dot
        self.state_feature[1] = y_drone_dot
        self.state_feature[2] = x_drone_dot - x_targ
        self.state_feature[3] = y_drone_dot - y_targ
        self.state_feature[4] = vx_drone_dot
        self.state_feature[5] = vy_drone_dot
        
        print("state feature px = " + str(self.state_feature[0]))
        print("state feature del-px = " + str(self.state_feature[2]))
        print("state feature vx = " + str(self.state_feature[4]))
        
        #calculate rewards
        #calculate the distance for Rd
        d_s = self.calcDistance(x_drone_dot, x_targ, y_drone_dot, y_targ)
        self.final_distance = d_s
        
        #set Rd part of reward
        Rd = -d_s
        reward_t += Rd
        
        #check to see if target and drone have collided, if so, fin
        if self.has_collided(self.drone, self.target) or (d_s < self.distThreshold):
            reward_t += 10
            done = True
            
        #check collisions with all obstacles
        self.obj_collided = False
        for obj in self.obs:
            if self.has_collided(self.drone, obj):
                reward_t += -2 #fixed -2 when uav collides with obstacle
                self.obj_collided = True
                break
        
        #increment cummulative reward
        self.cum_reward += reward_t
        
        #draw elements on canvas
        self.draw_elements_on_canvas()
        
        return self.state_feature, reward_t, done
    
    def calcDistance(self, x1, x2, y1, y2):
        return sqrt( (x1 - x2)**2 + (y1 - y2)**2 )
    
    def check_obj_collided(self):
        return self.obj_collided
        
class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min, dT):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name
        self.dT = dT
        
    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)
        
    def get_position(self):
        return (self.x, self.y)    
    
    def clamp(self, n, minn, maxn):
        max_n = min(maxn, n)
        min_n = max(max_n, minn)
        
        return min_n
        

class Drone(Point):
    def __init__(self, x0, y0, name, x_max, x_min, y_max, y_min, dT):
        super(Drone, self).__init__(name, x_max, x_min, y_max, y_min, dT)
        self.icon = cv2.imread("drone_top_down.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        
        #initialize position and set origin
        self.x_origin = x0
        self.y_origin = y0
        self.x = x0
        self.y = y0
        
        #sensor properties
        #half drone img dimension + half obj image dimension + 29 pixels
        #as quoted in the paper
        self.sensor_fov = 32/2+32/2 + 29
        self.obs_in_fov = False
        
        self.state = np.zeros((6,))
        self.cmd = np.zeros((6,))
        self.state_dot = np.zeros((6,))
        
    def get_velocity(self):
        return (self.state_dot[2], self.state_dot[3])
        
    def move(self, yaw):
        
        #force command to test steering
        #yaw = -0.0698
        
        #calculate the center of the image
        (h, w) = self.icon.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        
        #rotate by the yaw angle
        C = cv2.getRotationMatrix2D((cX, cY), math.degrees(yaw), 1.0)
        
        #warp using affine transform
        self.icon = cv2.warpAffine(self.icon, C, (w,h))
        
        #create C from scratch
        C = np.array([[cos(yaw), -sin(yaw)],[sin(yaw), cos(yaw)]])
        a_cmd = np.dot(C, self.state[-2:])
        
        #add in command x_dot = Ax + Bu
        #negate y component because opposite in grid
        #a_cmd[1] = -a_cmd[1]
        self.state[4:] = a_cmd
        
        #propogate forward using constant accleration model
        #position
        self.state_dot[0] = self.state[0] + self.state[2] * self.dT + 0.5 * self.dT * self.dT * self.state[4]
        self.state_dot[1] = self.state[1] - self.state[3] * self.dT + 0.5 * self.dT * self.dT * self.state[5]
        
        #velocity
        self.state_dot[2] = self.state[2] + self.dT * self.state[4]
        self.state_dot[3] = self.state[3] + self.dT * self.state[5]
        
        #acceleration
        self.state_dot[4] = self.state[4]
        self.state_dot[5] = self.state[5]
        
        #saturate if we are hitting a wall
        self.state_dot[0] = self.clamp(self.state_dot[0], self.x_min, self.x_max - self.icon_w)
        self.state_dot[1] = self.clamp(self.state_dot[1], self.y_min, self.y_max - self.icon_h)
        
        #overwrite old states with propogated
        self.state = self.state_dot
        
        print("yaw [degrees] = " + str(math.degrees(yaw)))
        print("x = " + str(self.state_dot[0]))
        print("y = " + str(self.state_dot[1]))
        
        #set x/y for drawing on canvas
        self.x = self.state_dot[0].astype(np.int64)
        self.y = self.state_dot[1].astype(np.int64)
    
class Target(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min, dT):
        super(Target, self).__init__(name, x_max, x_min, y_max, y_min, dT)
        self.icon = cv2.imread("target_basic.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        
class Obstacle(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min, dT):
        super(Obstacle, self).__init__(name, x_max, x_min, y_max, y_min, dT)
        self.icon = cv2.imread("obstacle.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        