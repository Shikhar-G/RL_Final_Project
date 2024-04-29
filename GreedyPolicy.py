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
from CCPP import CCPP_Env


# class RandPolicy(nn.Module):
#     def __init__(self):
#         super(RandPolicy, self).__init__()


args = {
    "max_episode_length": 1000,
    "num_episodes": 10
}

env = CCPP_Env(agent_dims=[0.2, 0.2], agent_loc=[7.5, 0])
state, info = env.reset()
done = False
total_reward = 0
curr_step = 0
prog_bar = tqdm(total=args["max_episode_length"])
num_invalid = 0
state_space = env.possible_start_positions
np.random.shuffle(state_space)
milestone = 0.1
while not done and curr_step < args["max_episode_length"]:
    for state in state_space:
        if env.coverage_channel_out[state[0], state[1]] == 1:
            action = state
            break
    action_x, action_y = env.transform_map_to_xy(*action)
    next_state, reward, done, truncated, info = env.step(np.array([action_x, action_y]))
    total_reward += reward
    state = next_state
    curr_step += 1
    prog_bar.update(1)
    if env.curr_coverage / env.coverage_possible >= milestone:
        print(f"Coverage: {milestone} hit at step {curr_step}")
        milestone += 0.1
    # env.render()
    # time.sleep(2)
env.render()
time.sleep(2)
print("")
print(f"Total Reward: {total_reward}\n")
