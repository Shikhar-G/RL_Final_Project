import torch
import torch.nn as nn
import torch.distributions as distributions
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.autograd import Variable
import easydict
import math
from collections import deque
from envs import env
import time
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import json

# argument parser
args = easydict.EasyDict({
"batch_size": 16,
"gamma": 0.99,
"lambda": 0.95,
"eps_clip": 0.2,
"buffer_size": 64,
"epochs": 4,
"lr": 3e-4,
"enable_cuda": True
})

class CCPP_Resnet(nn.Module):
    def __init__(self):
        super(CCPP_Resnet, self).__init__()
        resnet = resnet18(pretrained=True)
        resnet = resnet.float()
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # print(x.dtype)
        x = self.feature_extractor(x)
        return x


class CCPP_Actor(nn.Module):
    def __init__(self):
        super(CCPP_Actor, self).__init__()

        self.resnet = CCPP_Resnet().float()
        self.features = nn.Sequential()
        self.features.add_module("flatten", nn.Flatten())
        self.features.add_module("lin1", nn.Linear(512, 512))
        self.features.add_module("tanh1", nn.Tanh())
        self.features.add_module("lin2", nn.Linear(512, 256))
        self.features.add_module("tanh2", nn.Tanh())
        self.features.add_module("lin3", nn.Linear(256, 4))

    def forward(self, x):
        out = self.resnet(x)
        out = self.features(out)
        return out

class CCPP_Critic(nn.Module):
    def __init__(self):
        super(CCPP_Critic, self).__init__()

        self.resnet = CCPP_Resnet().float()
        self.features = nn.Sequential()
        self.features.add_module("flatten", nn.Flatten())
        self.features.add_module("lin1", nn.Linear(512, 512))
        self.features.add_module("tanh1", nn.Tanh())
        self.features.add_module("lin2", nn.Linear(512, 256))
        self.features.add_module("tanh2", nn.Tanh())
        self.features.add_module("lin3", nn.Linear(256, 1))

    def forward(self, x):
        out = self.resnet(x)
        out = self.features(out)
        return out


def get_action(policy_output):
    action_meanx, action_stdx, action_meany, action_stdy = (
        policy_output[:, 0],
        policy_output[:, 1],
        policy_output[:, 2],
        policy_output[:, 3],
    )
    action_stdx = torch.exp(action_stdx)
    action_stdy = torch.exp(action_stdy)
    distx = distributions.Normal(action_meanx, action_stdx)
    disty = distributions.Normal(action_meany, action_stdy)
    actionx = distx.sample()
    actiony = disty.sample()
    log_probx = distx.log_prob(actionx)
    log_proby = disty.log_prob(actiony)
    return torch.cat([actionx, actiony]), torch.tensor(log_probx + log_proby)


def compute_gae(
    next_value, rewards, masks, values, gamma=args["gamma"], lam=args["lambda"]
):
    values = values + [next_value]
    gae = 0
    returns = deque()
    advantages = deque()
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.appendleft(gae + values[step])
        advantages.appendleft(gae)
    return list(returns), list(advantages)


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.shape[0]
    for _ in range(batch_size // mini_batch_size):
        rand_ids = torch.randint(0, batch_size, (mini_batch_size,))
        yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[
            rand_ids
        ], advantage[rand_ids]


def preprocess_input(x):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x.float()),
        ]
    )
    return preprocess(x).unsqueeze(0)


def train(
    actor,
    critic,
    actor_optim,
    critic_optim,
    env,
    args,
    num_episodes=1000,
    max_episode_length=1000,
):
    for i in tqdm(range(num_episodes)):
        actor.train()
        critic.train()
        state, info = env.reset()
        # state_buffer_file = open("state_buffer.json", "w")
        # buffers
        state_buffer = []
        action_buffer = []
        log_prob_buffer = []
        reward_buffer = []
        mask_buffer = []
        value_buffer = []

        done = False
        curr_step = 0

        prog_bar = tqdm(total=max_episode_length)
        while not done and curr_step < max_episode_length:
            # change state to double np array
            # print(state)
            if len(state) == 2:
                state = state[0]

            state_buffer.append(torch.from_numpy(state))
            state = np.array(state, dtype=np.double)
            state = preprocess_input(state)
            policy, value = actor(state), critic(state)

            action, log_prob = get_action(policy)
            next_state, reward, done, truncated, info = env.step(
                action.detach().numpy()
            )
            mask = 1 if not done else 0

            action_buffer.append(action.detach())
            log_prob_buffer.append(log_prob.detach())
            reward_buffer.append(reward)
            mask_buffer.append(mask)
            value_buffer.append(value.detach())
            state = next_state
            curr_step += 1
            prog_bar.update(1)

            # if len(state_buffer) == args["buffer_size"]:
            #     # append to file
            #     json_string = state_buffer.tolist()
            #     state_buffer_file.append(json_string)
            #     state_buffer_file.flush()
            #     state_buffer = []

        # state_buffer_file.close()
        next_state = preprocess_input(next_state)
        next_value = critic(next_state)
        returns, advantages = compute_gae(
            next_value, reward_buffer, mask_buffer, value_buffer
        )

        states = torch.cat(state_buffer)
        actions = torch.stack(action_buffer)
        log_probs = torch.cat(log_prob_buffer)
        returns = torch.tensor(returns).float()
        advantages = torch.tensor(advantages).float()

        # print(states.shape)
        # print(actions.shape)
        # print(log_probs.shape)
        # print(returns.shape)
        # print(advantages.shape)

        # PPO update
        for _ in tqdm(range(args["epochs"])):
            for state, action, old_log_probs, return_, advantage in ppo_iter(
                args["batch_size"], states, actions, log_probs, returns, advantages
            ):
                state = np.array(state, dtype=np.double)
                state = preprocess_input(state)
                policy, value = actor(state), critic(state)

                action, log_prob = get_action(policy)

                ratio = torch.exp(log_prob - old_log_probs)
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1.0 - args["eps_clip"], 1.0 + args["eps_clip"])
                    * advantage
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

        if i % 10 == 0:
            print("Evaluation")
            torch.save(actor.state_dict(), "checkpoints/actor_{}.pth".format(i))
            torch.save(critic.state_dict(), "checkpoints/critic_{}.pth".format(i))
            # evaluate the model
            actor.eval()
            critic.eval()
            state, info = env.reset()
            done = False
            total_reward = 0
            curr_step = 0
            prog_bar = tqdm(total=max_episode_length)
            while not done and curr_step < max_episode_length:
                state = np.array(state, dtype=np.double)
                state = preprocess_input(state)
                policy, _ = actor(state), critic(state)
                action, _ = get_action(policy)
                action = action.detach().numpy()
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                state = next_state
                curr_step += 1
                prog_bar.update(1)
            print(f"Episode {i} Total Reward: {total_reward}")


actor = CCPP_Actor()
critic = CCPP_Critic()
actor = actor.float()
critic = critic.float()
actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr)


train(actor, critic, actor_optim, critic_optim, env, args, num_episodes=10)
