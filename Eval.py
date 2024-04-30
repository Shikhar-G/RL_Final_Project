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
from Policy import RL_Policy, RandPolicy, Policy, GreedyPolicy
from CCPP import CCPP_Env


def eval(actor_file, args, env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policies = [RandPolicy(env), GreedyPolicy(env), Policy(env, actor_file, device)]
    policy_types = ["Random", "Greedy", "RL"]

    for i, policy in enumerate(policies):
        print(f"Policy: {policy_types[i]}")
        for j in range(args["num_episodes"]):
            done = False
            curr_step = 0
            prog_bar = tqdm(total=args["max_episode_length"])
            total_reward = 0
            rewards = np.zeros((args["num_episodes"], args["max_episode_length"]))
            coverage = np.zeros((args["num_episodes"], args["max_episode_length"]))
            time_steps = np.zeros((args["num_episodes"], args["max_episode_length"]))
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
        # save results
        if not os.path.exists("results"):
            os.makedirs("results")
        np.save(f"results/{policy_types[i]}_rewards.npy", rewards)
        np.save(f"results/{policy_types[i]}_coverage.npy", coverage)
        np.save(f"results/{policy_types[i]}_time_steps.npy", time_steps)
        print(f"Average Total Reward: {np.mean(rewards[:, -1])}")
        print(f"Average Coverage: {np.mean(coverage[:, -1])}")
        print(f"Average Time Steps: {np.mean(time_steps[:, -1])}")
        print("\n")


