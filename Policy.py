import torch
import torch.nn as nn
import torch.distributions as distributions
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.autograd import Variable
import easydict
import math
from collections import deque
import time
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import json
import CNN_InAreaOnly

# abstract class for policies
class Policy():
    def __init__(self, env):
        self.env = env
    
    def get_action(self, state):
        raise NotImplementedError
    

    def index_to_action(self, index):
        action = self.env.possible_start_positions[index]
        action_x = action[0]/self.env.scaling + self.env.x_min
        action_y = action[1]/self.env.scaling + self.env.y_min
        np_action = np.array([action_x, action_y])
        return np_action
    
    
class RandomPolicy(Policy):
    def __init__(self, env):
        super(RandomPolicy, self).__init__(env)
    
    def get_action(self, state):
        index = np.random.randint(0, len(self.env.possible_start_positions))
        return self.index_to_action(index)


class GreedyPolicy(Policy):
    def __init__(self, env):
        super(GreedyPolicy, self).__init__(env)
    
    def get_action(self, state):
        random_index = np.random.randint(0, len(self.env.possible_start_positions))
        if self.env.coverage_channel_out[self.env.possible_start_positions[random_index][0], self.env.possible_start_positions[random_index][1]] == 1:
            return self.index_to_action(random_index)
        radius = 0
        max_radius = min(len(self.env.possible_start_positions) - 1 - random_index, random_index)
        while radius < max_radius:
            pos_position = self.env.possible_start_positions[random_index + radius]
            if self.env.coverage_channel_out[pos_position[0], pos_position[1]] == 1:
                return self.index_to_action(random_index + radius)
            neg_position = self.env.possible_start_positions[random_index - radius]
            if self.env.coverage_channel_out[neg_position[0], neg_position[1]] == 1:
                return self.index_to_action(random_index - radius)
            radius += 1
        # check remaining on left or right
        while random_index + radius < len(self.env.possible_start_positions):
            pos_position = self.env.possible_start_positions[random_index + radius]
            if self.env.coverage_channel_out[pos_position[0], pos_position[1]] == 1:
                return self.index_to_action(random_index + radius)
            radius += 1
        while random_index - radius >= 0:
            neg_position = self.env.possible_start_positions[random_index - radius]
            if self.env.coverage_channel_out[neg_position[0], neg_position[1]] == 1:
                return self.index_to_action(random_index - radius)
            radius += 1
        return self.index_to_action(random_index)

class RL_Policy(Policy):
    def __init__(self, env, model_file, device='cpu'):
        super(RL_Policy, self).__init__(env)
        self.model = CNN_InAreaOnly.CCPP_Actor().to(device)
        self.model.load_state_dict(torch.load(model_file, map_location=device))
    
    def get_action(self, state):
        policy = self.model(state)
        action, _ = CNN_InAreaOnly.get_action(policy.detach().cpu().numpy(), self.env)
        return action