"""
Deep Neural Network for DQN
"""

import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):

    def __init__(self, input_size, nb_action):
        super(NN, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        # nn.Linear => all neuran in input layer are connected to hidden layer
        # fc1 , fc2 => full connections
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, nb_action)
    
    def forward(self, state):# function for propagation
        # we apply rectifier fuction to hidden neurons and we should give our input connection input states
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values