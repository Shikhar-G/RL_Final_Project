import torch
import torch.nn as nn
import torch.distributions as distributions
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.autograd import Variable
import easydict
import math
from collections import deque
from envs import env
import time
from torchvision import transforms
import numpy as np
from tqdm import tqdm

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
        self.features.add_module("lin3", nn.Linear(256, 5))

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
    mean, std, stop_cmd = policy_output[:,0], policy_output[:,1], policy_output[:,2]
    dist = distributions.Normal(mean, std)
    action = dist.sample()
    return action, stop_cmd

def compute_gae(next_value, rewards, masks, values, gamma=args['gamma'], tau=args['lambda']):
    values.append(next_value)
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    values.popleft()
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.shape[0]
    for _ in range(batch_size // mini_batch_size):
        rand_ids = torch.randint(0, batch_size, (mini_batch_size,))
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


actor = CCPP_Actor()
critic = CCPP_Critic()
actor = actor.float()
critic = critic.float()
actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr)

# buffers
state_buffer = []
action_buffer = []
log_prob_buffer = []
reward_buffer = []
mask_buffer = []
value_buffer = []


def preprocess_input(x):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.float())
        ])
    return preprocess(x).unsqueeze(0)

def train():
    state, info = env.reset()
    
    for _ in tqdm(range(args['buffer_size'])):
        # change state to double np array
        # print(state)
        if len(state) == 2:
            state = state[0]
        state = np.array(state, dtype=np.double)
        state = preprocess_input(state)
        policy, value = actor(state), critic(state)

        action_meanx, action_stdx, action_meany, action_stdy, stop_cmd = policy[:,0], policy[:,1], policy[:,2], policy[:,3], policy[:,4]
        action_stdx = torch.exp(action_stdx)
        action_stdy = torch.exp(action_stdy)
        distx = distributions.Normal(action_meanx, action_stdx)
        disty = distributions.Normal(action_meany, action_stdy)
        actionx = distx.sample()
        actiony = disty.sample()
        log_probx = distx.log_prob(actionx)
        log_proby = disty.log_prob(actiony)

        combined_action = np.concatenate((actionx.detach().numpy(), actiony.detach().numpy(), stop_cmd.detach().numpy()))
        # print(combined_action)
        next_state, reward, done, truncated, info = env.step(combined_action)
        mask = 1 if not done else 0

        state_buffer.append(state)
        action_buffer.append(combined_action)
        log_prob_buffer.append(log_probx.detach().numpy() + log_proby.detach().numpy())
        reward_buffer.append(reward)
        mask_buffer.append(mask)
        value_buffer.append(value.detach().numpy())
        if done:
            state = env.reset()
        else:
            state = next_state
        
    next_state = preprocess_input(next_state)
    next_value = critic(next_state)
    returns = compute_gae(next_value.detach().numpy(), reward_buffer, mask_buffer, value_buffer)
    advantages = np.array(returns) - np.array(value_buffer)


    # PPO update
    for _ in range(args['epochs']):
        for state, action, old_log_probs, return_, advantage in ppo_iter(args['batch_size'], states, actions, log_probs, returns, advantages):
            policy, value = actor(state), critic(state)

            action_meanx, action_stdx, action_meany, action_stdy, stop_cmd = policy[:,0], policy[:,1], policy[:,2], policy[:,3], policy[:,4]
            action_stdx = torch.exp(action_stdx)
            action_stdy = torch.exp(action_stdy)
            distx = distributions.Normal(action_meanx, action_stdx)
            disty = distributions.Normal(action_meany, action_stdy)
            actionx = distx.sample()
            actiony = disty.sample()
            log_probx = distx.log_prob(actionx)
            log_proby = disty.log_prob(actiony)

            ratio = torch.exp(log_probx + log_proby - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - args['eps_clip'], 1.0 + args['eps_clip']) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

train()
