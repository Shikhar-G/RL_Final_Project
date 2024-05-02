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
import os
from Policy import RL_Policy, RandomPolicy, Policy, GreedyPolicy
from CCPP import CCPP_Env
import matplotlib.pyplot as plt
args = easydict.EasyDict(
    {
        "batch_size": 32,
        "gamma": 0.99,
        "lambda": 0.95,
        "eps_clip": 0.2,
        "buffer_size": 64,
        "epochs": 10,
        "lr": 1e-6,
        "max_episode_length": 512,
        "num_episodes": 5,
        "enable_cuda": True,
        "entropy_coef": 0.1,
    }
)

def eval(actor_file, args, env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policies = [RL_Policy(env, actor_file, device)] #RandomPolicy(env), GreedyPolicy(env),
    policy_types = ["RL"] # "Random", "Greedy", 

    for i, policy in enumerate(policies):
        print(f"Policy: {policy_types[i]}")
        final_rewards = np.zeros((args["num_episodes"]))
        final_coverage = np.zeros((args["num_episodes"]))
        final_time_steps = np.zeros((args["num_episodes"]))
        rewards = np.zeros((args["num_episodes"], args["max_episode_length"]))
        coverage = np.zeros((args["num_episodes"], args["max_episode_length"]))
        time_steps = np.zeros((args["num_episodes"], args["max_episode_length"]))
        for j in range(args["num_episodes"]):
            done = False
            curr_step = 0
            prog_bar = tqdm(total=args["max_episode_length"])
            total_reward = 0
            state, _ = env.reset()
            while not done and curr_step < args["max_episode_length"]:
                action = policy.get_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                rewards[j, curr_step] = total_reward
                coverage[j, curr_step] = env.curr_coverage
                time_steps[j, curr_step] = env.curr_time
                state = next_state
                curr_step += 1
                prog_bar.update(1)
            prog_bar.close()
            final_rewards[j] = total_reward
            final_coverage[j] = env.curr_coverage
            final_time_steps[j] = env.curr_time
        # save results
        if not os.path.exists("results"):
            os.makedirs("results")
        np.save(f"results/{policy_types[i]}_rewards.npy", rewards)
        np.save(f"results/{policy_types[i]}_coverage.npy", coverage)
        np.save(f"results/{policy_types[i]}_time_steps.npy", time_steps)
        print(f"Average Total Reward: {np.mean(final_rewards)}")
        print(f"Average Coverage: {np.mean(final_coverage)}")
        print(f"Average Time Steps: {np.mean(final_time_steps)}")
        print("\n")

env = CCPP_Env(agent_dims=[0.2, 0.2], agent_loc=[ 0, 8],map_file="maps/GDC1_Ground_top_only.vectormap.json",scaling=6, coverage_radius=2)

eval("checkpoints_area/actor_1preset_7687_.1entr.pth", args, env)