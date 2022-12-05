# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:26:45 2022

Main Reference: https://blog.paperspace.com/creating-custom-environments-openai-gym/

@authors: skwel (Sam Welker & Whitchurch Muthumani) (reward shaping and additional modifications by Whitchurch)
"""

from gym import Env, spaces
import numpy as np
import cv2
import random
import torch
from math import exp, sqrt, atan2, pi, asin, sin

class UAV(Env):
    

    def __init__(self, lamb, T0, alpha_low, beta, b1, b2, b3):
        super(UAV, self).__init__()
        
        self.current_step = 0
        #boltzman probability parameters
        self.lamb = lamb
        self.T0 = T0
        self.alpha_low = alpha_low
        self.beta = beta
        self.obj_collided = False
        self.target_collided = False
        self.reward = 0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.final_distance = None
        """
        Action (low) space defined as:
        0: N, 1: NE, 2: E, 3: SE
        4: S, 5: SW, 6: W, 7: NW
        8: Do Nothing
        
        Dynamic Action (high) space defined as:
        0: N, 1: NE, 2: E, 3: W
        4: NW, 5: Do Nothing        
        """
        self.nActions = 9
        self.nDynamicActions = 6
        self.nObstacles = 3
        self.action_space = spaces.Discrete(self.nActions,)
        self.action_space_high = spaces.Discrete(self.nDynamicActions,)
        
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
        
        
    def select_action(self,state_x,state_y,policy_net,pstrategy):
        strategy = pstrategy
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step +=1
        
        if rate > random.random():
            action = self.action_space.sample() # explore
            return action
        else:
            with torch.no_grad():
                return policy_net(state_x,state_y).argmax(dim=1).to(self.device) #exploit
    
        
        
    def draw_elements_on_canvas(self):
        
        self.canvas = np.ones(self.observation_shape) * 1
        
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x,y = elem.x, elem.y
            self.canvas[y : y + elem_shape[1], x : x + elem_shape[0]] = elem.icon
            
        #text = 'Rewards: {}, Penalties: {}, Episode #: {}'.format(self.ep_return, self.penalties, self.episode)
        text = 'Rewards: {:.1f}, Ep: {:.0f}'.format(self.reward, self.episode)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        #self.canvas = cv2.putText(self.canvas, text, (10,20), 
                                  #font, 0.8,
                                  #(0,0,0), 1, 
                                  #cv2.LINE_AA)
        
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
        
        #initialize dynamic states
        R_obs, D_obs, R_targ, A_targ_obs = [None for i in range(4)]
        
        reward = 0
        reward_high = 0
        penalties = 0
        
        #calculate current distance from uav to target
        x_drone, y_drone = self.drone.get_position()
        x_targ, y_targ = self.target.get_position()
        d_s_minus = self.calcDistance(x_drone, x_targ, y_drone, y_targ)
        
        """
        Action space defined as:
        0: N, 1: NE, 2: E, 3: SE
        4: S, 5: SW, 6: W, 7: NW
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
        self.final_distance = d_s
        
        #calculate the distance to each obstacle and sum inverses
        inverseObsSum = 0
        for obj in self.obs:
            x_obs, y_obs = obj.get_position()
            obs_i = self.calcDistance(x_drone_dot, x_obs, y_drone_dot, y_obs)
            inverseObsSum += 1/obs_i
            
        #calculate the "static obstacle" reward
        reward = self.alpha_low * (d_s_minus - d_s) - self.beta * inverseObsSum
        
        #check to see if target and drone have collided, if so, fin
        self.target_collided = False
        if self.has_collided(self.drone, self.target):
            self.target_collided = True
            done = True
            
        #check collisions with all obstacles
        self.obj_collided = False
        for obj in self.obs:
            if self.has_collided(self.drone, obj):
                reward = reward - 50
                self.obj_collided = True
                break
        
        #move moving obstacle (just move up and down for simplicity)
        #get current position
        curr_move_obs_x, curr_move_obs_y = self.moving_obs.get_position()
        d_uo = self.calcDistance(x_drone_dot, curr_move_obs_x, y_drone_dot, curr_move_obs_y)
        
        if curr_move_obs_x < int(self.observation_shape[0] * 0.40):
            self.moving_obs.set_move_dir(0)
        elif curr_move_obs_x > int(self.observation_shape[0] * 0.85):
            self.moving_obs.set_move_dir(1)
        
        if self.moving_obs.move_dir == 0:
            self.moving_obs.move(2, 0)
        elif self.moving_obs.move_dir == 1:
            self.moving_obs.move(-2, 0)
        
        next_move_obs_x, next_move_obs_y = self.moving_obs.get_position()
        d_uo_plus = self.calcDistance(x_drone_dot, next_move_obs_x, y_drone_dot, next_move_obs_y)
        
        #check if the moving obstacle is in the sensor FOV
        R_obs, D_obs, R_targ, A_targ_obs = \
            self.drone.checkSensor(self.moving_obs, self.target)
        
        obstacle_in_fov = self.drone.checkObsInFov()
        s_high = []
        if obstacle_in_fov == True:
            s_high.append(R_obs)
            s_high.append(D_obs)
            s_high.append(R_targ)
            s_high.append(A_targ_obs)
            
            #get current alpha
            alpha_o_line = self.drone.getAlpha()
            
            #calculate high layer reward
            reward_high = self.b1 * (d_s_minus - d_s) + \
                exp(self.b2 * (d_uo - d_uo_plus)) + \
                self.b3 * sin(alpha_o_line)
        
        #check collisions with moving obstacle
        if self.has_collided(self.drone, self.moving_obs):
            reward = reward - 50
            self.obj_collided = True
        
        #increment penalties return
        self.penalties += penalties
        
        #draw elements on canvas
        self.draw_elements_on_canvas()
        
        self.reward += (reward + reward_high)
        
        return self.canvas, reward, s_high, reward_high, done
    
    def reset(self,episode):
        self.reward = 0
        self.episode = episode
        state_high = []
        
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
        
        #initialize final distance
        self.final_distance = self.calcDistance(x, x_targ, y, y_targ)
        
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
        
        #initialize moving obstacles
        x_move_obs = int(self.observation_shape[0] * 0.45)
        y_move_obs = int(self.observation_shape[1] * 0.50)
        
        #set moving obstacle position
        self.moving_obs = MovingObstacle("moving_obstacle", self.x_max, self.x_min, self.y_max, self.y_min)
        self.moving_obs.set_position(x_move_obs, y_move_obs)
        
        #init elements
        self.elements = [self.drone, 
                         self.target, 
                         self.obs[0], self.obs[1], self.obs[2], 
                         self.moving_obs]
        
        #init canvas
        self.canvas = np.ones(self.observation_shape) * 1
        
        #draw elements on canvas
        self.draw_elements_on_canvas()
        
        #check sensor at beginning and get current initial high layer states
        R_obs,D_obs,R_targ,A_targ_obs = self.drone.checkSensor(self.moving_obs, self.target)
        if self.drone.checkObsInFov() == True:
            state_high.append(R_obs)
            state_high.append(D_obs)
            state_high.append(R_targ)
            state_high.append(A_targ_obs)
        
        return self.canvas, state_high
    
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
            exp_num = exp(-float(Q_table[a])/T_k)
            
            #reset cumSum
            exp_den = 0.0
            
            for i in range(self.nActions):
                    exp_den += exp(-float(Q_table[i])/T_k)
                    
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
   
    def check_target_collided(self):
        return self.target_collided
    
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
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        
        #sensor properties
        self.sensor_fov = 32/2+32/2
        self.obs_in_fov = False
    
    def checkSensor(self, moving_obstacle, target):
        
        #initialize obstacle region to empty
        R_obs = None
        R_targ = None
        D_obs = None
        A_targ_obs = None
        
        #assume obstacle not in fov
        self.obs_in_fov = False
            
        #check to see if the moving obstacle is within the sensor fov
        moving_obs_x, moving_obs_y = moving_obstacle.get_position()
        targ_x, targ_y = target.get_position()
        self_x, self_y = self.get_position()
        
        #calculate distance
        del_obs_x = (moving_obs_x - self_x)
        del_obs_y = (moving_obs_y - self_y)
        d_uo = sqrt( del_obs_x**2 + del_obs_y**2 )
        
        #check which region the target is in
        del_targ_x = (targ_x - self_x)
        del_targ_y = (targ_y - self_y)
        #d_ut = sqrt( del_targ_x**2 + del_targ_y**2 )
        
        #angle corresponding to the region the target is in
        alpha_targ = atan2(del_targ_x, del_targ_y) * (180.0/pi)
        
        #region check
        if (alpha_targ > 0 and alpha_targ <= 45):
            R_targ = 1
        elif (alpha_targ > 45 and alpha_targ <= 90):
            R_targ = 2
        elif (alpha_targ > 90 and alpha_targ <= 135):
            R_targ = 3
        elif (alpha_targ > 135 and alpha_targ <= 180):
            R_targ = 4
        elif (alpha_targ < 0 and alpha_targ >= -45):
            R_targ = 5
        elif (alpha_targ < -45 and alpha_targ >= -90):
            R_targ = 6
        elif (alpha_targ < -90 and alpha_targ >= -135):
            R_targ = 7
        elif (alpha_targ < -135 and alpha_targ >= -180):
            R_targ = 8
        
        if d_uo < self.sensor_fov:
            #object is in fov
            self.obs_in_fov = True
            
            #figure out which region the moving object is in
            #calculate the angle from the Y-axis
            alpha_obs = atan2(del_obs_x, del_obs_y) * (180.0/pi)
            
            #region check
            if (alpha_obs > 0 and alpha_obs <= 45):
                R_obs = 1
            elif (alpha_obs > 45 and alpha_obs <= 90):
                R_obs = 2
            elif (alpha_obs > 90 and alpha_obs <= 135):
                R_obs = 3
            elif (alpha_obs > 135 and alpha_obs <= 180):
                R_obs = 4
            elif (alpha_obs < 0 and alpha_obs >= -45):
                R_obs = 5
            elif (alpha_obs < -45 and alpha_obs >= -90):
                R_obs = 6
            elif (alpha_obs < -90 and alpha_obs >= -135):
                R_obs = 7
            elif (alpha_obs < -135 and alpha_obs >= -180):
                R_obs = 8
                
            #get direction of moving object
            D_obs = moving_obstacle.move_dir
            
            #calculate the angle betwee the target and moving obstacle
            k = (targ_y - self_y)/(targ_x - self_x);
            b = self_y - k * self_x
            
            d_o_line = abs( moving_obs_x - k * self_y - b)/sqrt(k**2 + b**2)
            
            #calculate the angle
            alpha_o_line = asin( d_o_line / d_uo );
            self.alpha_o_line = alpha_o_line
        
            #check the region alpha_o_line is in
            if alpha_o_line >= 0 and alpha_o_line < pi/8:
                A_targ_obs = 1
            elif alpha_o_line >= pi/8 and alpha_o_line < pi/4:
                A_targ_obs = 2
            elif alpha_o_line >= pi/4 and alpha_o_line < 3*pi/8:
                A_targ_obs = 3
            elif alpha_o_line >= 3*pi/8 and alpha_o_line < pi/2:
                A_targ_obs = 4
            elif alpha_o_line >= pi/2 and alpha_o_line < 5*pi/8:
                A_targ_obs = 5
            elif alpha_o_line >= 5*pi/8 and alpha_o_line < 3*pi/4:
                A_targ_obs = 6
            elif alpha_o_line >= 3*pi/4 and alpha_o_line < 7*pi/8:
                A_targ_obs = 7
            elif alpha_o_line >= 7*pi/8 and alpha_o_line < pi:
                A_targ_obs = 8
        
        return R_obs, D_obs, R_targ, A_targ_obs
        
    def checkObsInFov(self):
        return self.obs_in_fov
    
    def getAlpha(self):
        return self.alpha_o_line
    
    
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
        
class MovingObstacle(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(MovingObstacle, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("moving_obstacle.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        self.move_dir = 0
    
    def set_move_dir(self, move_dir):
        #legend: 0 = east, 1 = west (for simplicity)
        self.move_dir = move_dir
        