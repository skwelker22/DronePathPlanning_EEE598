"""
DQN Infrastructure
"""
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from models import NN


class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        #(state1,action1,reward1),(state2,acition2,reward2) => (state1,state2),(action1,action2),(reward1,reward2)
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = NN(input_size, nb_action)
        self.memory = ReplayMemory(1000) 
        #parameters of our model and the learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        #for pytorch it not only has to be torch tensor and one more dimension  that corresponds to the batch
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        #go straight
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = torch.argmax(probs)

        return action.item()
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward

        td_loss = F.smooth_l1_loss(outputs, target)
        # zero_grad reinitializes the optimizer in every iteration of loop
        self.optimizer.zero_grad()
        #free memory because we go several times on the loss
        td_loss.backward()
        #.step updates th weights by backpropagating
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        #convert signal to Tensor(float) since it is input of neural network and add dimention according to batch
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #torch.longtensor converts int to long in tensor
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            #we get 100 from each of them
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    