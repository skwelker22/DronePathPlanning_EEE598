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
        #starts givng + 10 reward when uav gets close to target
        self.distThreshold = 64 + 32 
        self.state_feature = np.zeros((num_states,))
        self.init_feature = np.zeros((num_states,))
        
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
        
        #normalization factor for rewards and state features
        self.normFactor = sqrt( self.observation_shape[0] ** 2 + self.observation_shape[1] ** 2)
        
        
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
        text = 'R_sum: {:.1f}, Ep: {:.0f}'.format(self.cum_reward, self.episode)
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
    
    def has_collided_with_boundry(self, drone):
        x_col = False
        y_col = False
        
        #get drone position
        drone_x, drone_y = drone.get_position()
 
        #get grid world shape
        grid_x = self.observation_shape[0]
        grid_y = self.observation_shape[1]
        
        #check to see if the drone position is touching the boundry
        #define conditions
        left_edge = (drone_x - drone.icon_w) <= 0
        right_edge = (drone_x + drone.icon_w) >= grid_x
        bottom_edge = (drone_y + drone.icon_h) >= self.y_max
        top_edge = (drone_y) <= self.y_min
        
        if left_edge or right_edge:
            x_col = True
            
        if top_edge or bottom_edge:
            y_col = True
        
        #if left_edge or top_edge:
        if x_col or y_col:
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
        self.drone.state[2] = 1.0  #1 pixel / second
        self.drone.state[3] = 0.0  #1 pixel / second
        self.drone.state[4] = 0.0  #0.5 pixel / second/ second
        self.drone.state[5] = 0.0   
        
        #initialize the target location
        #x_targ = int(self.observation_shape[0] * 0.30)
        #y_targ = int(self.observation_shape[1] * 0.80)
        x_targ = int(self.observation_shape[0] * 0.90)
        y_targ = int(self.observation_shape[1] * 0.85)
        
        self.target = Target("target", self.x_max, self.x_min, self.y_max, self.y_min, self.dT)
        self.target.set_position(x_targ, y_targ)
        
        #initialize final distance
        self.final_distance = self.calcDistance(x, x_targ, y, y_targ)
        
        #calcualte initial state feature
        self.init_feature[0] = self.drone.state[0]/self.normFactor #initial uav position (x)
        self.init_feature[1] = self.drone.state[1]/self.normFactor #initial uav position (y)
        self.init_feature[2] = (self.drone.state[0] - x_targ)/self.normFactor #initial relative position (x)
        self.init_feature[3] = (self.drone.state[1] - y_targ)/self.normFactor #initial relative position (y)
        self.init_feature[4] = self.drone.state[2]/self.normFactor # initial velocity (x)
        self.init_feature[5] = self.drone.state[3]/self.normFactor # initial velocity (y)
        
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
        
        return self.init_feature

    def step(self, action):
        
        #reset done and current rewards
        done = False
        reward_t = 0
        
        #check if the requested action is within bounds
        assert action[0] <= self.angMax and action[0] >= self.angMin, "Invalid Action"
        
        #propogate the drone given current yaw angle
        """
        Action space defined as:
        yaw angle in the interval [-pi:pi/15:pi]
        """
        #print("Action in [degrees] = " + str(math.degrees(action[0])))
        self.drone.move(action[0])
        
        #create new state feature vector
        x_targ, y_targ = self.target.get_position()
        x_drone_dot, y_drone_dot = self.drone.get_position()
        vx_drone_dot, vy_drone_dot = self.drone.get_velocity()
        
        #set state features, normalize
        self.state_feature[0] = x_drone_dot/self.normFactor
        self.state_feature[1] = y_drone_dot/self.normFactor
        self.state_feature[2] = (x_drone_dot - x_targ)/self.normFactor
        self.state_feature[3] = (y_drone_dot - y_targ)/self.normFactor
        self.state_feature[4] = vx_drone_dot/self.normFactor
        self.state_feature[5] = vy_drone_dot/self.normFactor
        
        """
        print("state feature px = " + str(self.state_feature[0]))
        print("state feature del-px = " + str(self.state_feature[2]))
        print("state feature vx = " + str(self.state_feature[4]))
        """
        
        #calculate rewards
        #calculate the distance for Rd
        d_s = self.calcDistance(x_drone_dot, x_targ, y_drone_dot, y_targ)
        self.final_distance = d_s
        
        #set Rd part of reward
        #normalize to shape of space
        Rd = -d_s/self.normFactor
        reward_t += Rd
        
        ###################################################
        #####  add in T reward that describes the threat 
        ###################################################
        #calculate equation for LOS
        m_los = (y_drone_dot - y_targ) / (x_drone_dot - x_targ)
        b_los = y_drone_dot - m_los * x_drone_dot
        
        #calculate perpendicular slope
        m_perp = -1/m_los
        
        #calculate equation of uav line and target line
        #get y - intercepts
        b_uav  = y_drone_dot - m_perp * x_drone_dot
        b_targ = y_targ - m_perp * x_targ
        
        #find minimum distance from self to each obstacle
        d_min = 1e3 #init to something large
        T_reward = 0
        for obj in self.obs:
            x_obs, y_obs = obj.get_position()
            d_obs = self.calcDistance(x_drone_dot, x_obs, y_drone_dot, y_obs)
            if (d_obs < d_min):
                d_min = d_obs
            
            #check if the obstacles are in the threat area, closer to target
            # or closer to uav
            
            #generate second point on uav and targ lines
            x_uav2  = 100
            x_targ2 = 100
            y_uav2  = m_perp * x_uav2 + b_uav
            y_targ2 = m_perp * x_targ2 + b_targ
            
            #calculate determinant and check sign
            detSgnUav = (x_uav2 - x_drone_dot) * (y_obs - y_drone_dot) - \
                (x_obs - x_drone_dot) * (y_uav2 - y_drone_dot)
            detSgnTarg = (x_targ2 - x_targ) * (y_obs - y_targ) - \
                (x_obs - x_targ) * (y_targ2 - y_targ)
            
            if detSgnUav > 0: #obstacle left of uav line
                if (d_obs < d_min):
                    T_reward += exp(1 - d_min ** 2 / (d_min ** 2 - d_obs ** 2 + 1e-4))
            
            elif detSgnTarg < 0: #obstacle right of targ line
                d_obs_targ = self.calcDistance(x_obs, x_targ, y_obs, y_targ)
                if (d_obs_targ < d_min):
                    T_reward += exp(1 - d_min ** 2 / (d_min ** 2 - d_obs ** 2 + 1e-4))
                
            else:   
                #calculate the dl term for each obstacle
                #dl is the distance between the LOS line and obstacle
                #assume all obstacles are in the inner region
                b_obs = y_obs - m_perp * x_obs
            
                #find the a,b,c terms for the intersection point equation
                #https://www.cuemath.com/geometry/intersection-of-two-lines/
                x_intersect = (b_obs - b_los) / (m_los - m_perp)
                y_intersect = m_los * x_intersect + b_los
            
                d_intersect = self.calcDistance(x_obs, x_intersect, y_obs, y_intersect)
            
                #check if dl < d_min
                if d_intersect < d_min:
                    T_reward += exp(1 - d_min ** 2 / (d_min ** 2 - d_intersect ** 2 + 1e-4))
        
        #add in T_reward to current episode reward
        reward_t += -T_reward
        
        #if within dist threshold start giving big rewards
        if (d_s < self.distThreshold):
            reward_t += 10
        
        #check to see if target and drone have collided, if so, fin
        if self.has_collided(self.drone, self.target):
            reward_t += 50
            done = True
            
        #check collisions with all obstacles
        self.obj_collided = False
        for obj in self.obs:
            if self.has_collided(self.drone, obj):
                reward_t += -10 #fixed -2 when uav collides with obstacle
                self.obj_collided = True
                break
        
        #check for collisions with the boundry of the grid space
        if self.has_collided_with_boundry(self.drone):
            reward_t += -100 
            self.obj_collided = True
        
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
        self.x_origin = self.clamp(x0, self.x_min, self.x_max - self.icon_w)
        self.y_origin = self.clamp(y0, self.y_min, self.y_max - self.icon_h)
        self.x = self.clamp(x0, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y0, self.y_min, self.y_max - self.icon_h)
        
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
        #yaw = 10*math.pi/180
        
        #calculate the center of the image
        (h, w) = self.icon.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        
        #rotate by the yaw angle
        C = cv2.getRotationMatrix2D((cX, cY), math.degrees(yaw), 1.0)
        
        #warp using affine transform
        self.icon = cv2.warpAffine(self.icon, C, (w,h))
        
        #create C from scratch
        C = np.array([[cos(yaw), -sin(yaw)],[sin(yaw), cos(yaw)]])
        cmd_power = 0.001
        a_cmd = np.dot(C, cmd_power * np.array([1,0]))
        
        #add in command x_dot = Ax + Bus
        #negate the y component because y axis is oriented opposite of
        #conventional notation
        a_cmd[1] = -a_cmd[1]
        self.state[4:] += a_cmd
        
        #propogate forward using constant accleration model
        #position
        self.state_dot[0] = self.state[0] + self.state[2] * self.dT + 0.5 * self.dT * self.dT * self.state[4]
        self.state_dot[1] = self.state[1] + self.state[3] * self.dT + 0.5 * self.dT * self.dT * self.state[5]
        
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
        
        """
        print("yaw cmd [degrees] = " + str(math.degrees(yaw)))
        print("x = " + str(self.state_dot[0]))
        print("y = " + str(self.state_dot[1]))
        print("vx = " + str(self.state_dot[2]))
        print("vy = " + str(self.state_dot[3]))
        print("ax = " + str(self.state_dot[4]))
        print("ay = " + str(self.state_dot[5]))
        """
        
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
        