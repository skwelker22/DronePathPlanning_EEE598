"""
Utilities for DQN
"""

from collections import namedtuple, deque
import random

# transition buffer for the UAV
Transition = namedtuple('Transition',
                        ('states_x', 'states_y', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


