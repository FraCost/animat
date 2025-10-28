import copy
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import math


# ----------------------------------------------------------
# RNN controller
# ----------------------------------------------------------
class RNN:
    def __init__(self, input_size, hidden_size, output_size, activation, alpha):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.alpha = alpha
        if activation == relu:
            self.init_fcn = he_init
        else:
            self.init_fcn = xavier_init
        self.init_weights()
        self.init_biases()
        self.init_state()

    def init_weights(self):
        self.W_in = self.init_fcn(n_in=self.input_size, n_out=self.hidden_size)
        self.W_h = self.init_fcn(n_in=self.hidden_size, n_out=self.hidden_size)
        self.W_out = self.init_fcn(n_in=self.hidden_size, n_out=self.output_size)

    def init_biases(self):
        self.b_h = np.zeros(self.hidden_size)
        self.b_out = np.zeros(self.output_size)

    def init_state(self):
        """Reset hidden state between episodes"""
        self.h = np.zeros(self.hidden_size)
        self.out = np.zeros(self.output_size)

    def step(self, obs):
        """Compute one RNN step"""
        self.h = (1 - self.alpha) * self.h + self.alpha * self.activation(
            self.W_in @ obs + self.W_h @ self.h + self.b_h
        )
        self.out = (1 - self.alpha) * self.out + self.alpha * logistic(
            self.W_out @ self.h + self.b_out
        )
        return self.out


# ----------------------------------------------------------
# Low-level controller
# ----------------------------------------------------------
class LowLevelController:
    def __init__(self, num_units, num_actuators, weigths_inhib=0.5):
        self.num_units = num_units
        self.num_actuators = num_actuators
        self.weigths_inhib = weigths_inhib
        self.weights = self.init_weights()

    def init_weights(self):
        if self.num_actuators % 2 != 0:
            raise ValueError("Number of actuators must be even")
        
        if self.num_actuators == 2:
            W0 = np.linspace(1.0, 0.0, self.num_units)
            W1 = np.linspace(0.0, 1.0, self.num_units)
            W = np.vstack([W0, W1])
        
        elif self.num_actuators == 4:
            resolution = math.isqrt(self.num_units)
            if resolution * resolution != self.num_units:
                raise ValueError(f"num_units ({self.num_units}) must be a perfect square.")
            
            gradient = np.linspace(1.0, 0.0, resolution)
            W0 = np.repeat(gradient, resolution)
            W1 = np.repeat(gradient[::-1], resolution)
            W2 = np.tile(gradient, resolution)
            W3 = np.tile(gradient[::-1], resolution)
            W = np.vstack([W0, W1, W2, W3]) 
                
        return W
    
    def step(self, activation, feedback):
        antagonists = [(2*i, 2*i + 1) for i in range(self.num_actuators // 2)]

        # Interneurons → gamma MN offsets
        gamma_activation = self.weights @ activation  

        # Compute alpha activations
        spindle_lengths = np.array(feedback)
        spindle_activation = spindle_lengths + gamma_activation 
        
        alpha_activation = np.zeros_like(spindle_activation)
        for f, e in antagonists:
            alpha_activation[f] = max(0.0, spindle_activation[f] - self.weigths_inhib * spindle_activation[e])
            alpha_activation[e] = max(0.0, spindle_activation[e] - self.weigths_inhib * spindle_activation[f])
        
        return alpha_activation