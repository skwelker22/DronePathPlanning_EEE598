"""
DQN for Dron path planning
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from utils import ReplayMemory, Transition

parser = argparse.ArgumentParser(description='DQN Learning')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.999, help='Gamma value')
parser.add_argument('--eps_start', type=float, default=0.9, help='Gamma value')
parser.add_argument('--eps_final', type=float, default=0.5, help='Gamma value')

parser.add_argument('--eps_decay', default=200, type=int, metavar='W', dest='eps_decay')
parser.add_argument('--log_file', type=str, default=None, help='path to log file')

def main():
    pass