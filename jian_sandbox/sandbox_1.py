# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:04:28 2022

@author: skwel
"""

#script to try out Q - learning strategy from paper
#https://ieeexplore-ieee-org.ezproxy1.lib.asu.edu/stamp/stamp.jsp?tp=&arnumber=8027884

from random import seed
from random import random
import math

#init network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

#define activation function
def activate(weights, inputs):
    #assume bias is the last weight
    activation = weights[-1]
    #calculate the summed weighted input
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

#neuron transfer function
def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))

#calculate neuron output derivative (assumes sigmoid transfer
def transfer_derivative(output):
    return output * (1.0 - output)

#forward propagation
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

## -----------------
## MAIN
## -----------------
#get random seed and setup the NN
seed(1)
network = initialize_network(2, 1, 2)

#print out what the network looks like
if 0:
    for layer in network:
        print(layer)

row = [1, 0, None]
output = forward_propagate(network, row)
print(output)







    
    










