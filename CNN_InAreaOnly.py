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
import matplotlib.pyplot as plt
from CCPP import CCPP_Env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argument parser
args = easydict.EasyDict(
    {
        "batch_size": 32,
        "gamma": 0.99,
        "lambda": 0.95,
        "eps_clip": 0.2,
        "buffer_size": 64,
        "epochs": 10,
        "lr": 1e-6,
        "max_episode_length": 128,
        "num_episodes": 32,
        "enable_cuda": True,
        "entropy_coef": 0.01,
        "device": device,
    }
)

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
        self.features.add_module("lin2", nn.Linear(512, 512))
        self.features.add_module("tanh2", nn.Tanh())
        self.features.add_module("lin3", nn.Linear(512, 2))

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
        self.features.add_module("lin2", nn.Linear(512, 512))
        self.features.add_module("tanh2", nn.Tanh())
        self.features.add_module("lin3", nn.Linear(512, 1))

    def forward(self, x):
        out = self.resnet(x)
        out = self.features(out)
        return out

def convert_position_to_action(position, env):
    return np.array(
        [
            position[0] / env.scaling + env.x_min,
            position[1] / env.scaling + env.y_min,
        ]
    )


def transform_action(action, env, ind = 0):
    tanh_action = torch.tanh(action)
    index_to_pos = len(env.possible_start_positions) * (tanh_action + 1) / 2
    env_index =(
            round(
                torch.clip(index_to_pos, 0, len(env.possible_start_positions) - 1)
                .detach()
                .cpu()
                .numpy()[ind]
            )
        )
    env_position = env.possible_start_positions[env_index]

    if env.coverage_channel_out[env_position[0], env_position[1]] == 1:
        return convert_position_to_action(env_position, env) , convert_index_to_action(env_index, env)
    else:
        pos_index = env_index
        neg_index = env_index
        pos_position = env_position
        neg_position = env_position
        steps_to_increment = min(len(env.possible_start_positions) - 1 - env_index, env_index)
        for i in range(steps_to_increment):
            pos_index += 1
            neg_index -= 1
            pos_position = env.possible_start_positions[pos_index]
            neg_position = env.possible_start_positions[neg_index]
            if env.coverage_channel_out[pos_position[0], pos_position[1]] == 1:
                return convert_position_to_action(pos_position, env) , convert_index_to_action(pos_index, env)
            if env.coverage_channel_out[neg_position[0], neg_position[1]] == 1:
                return convert_position_to_action(neg_position, env) , convert_index_to_action(neg_index, env)
        while pos_index < len(env.possible_start_positions):
            pos_position = env.possible_start_positions[pos_index]
            if env.coverage_channel_out[pos_position[0], pos_position[1]] == 1:
                return convert_position_to_action(pos_position, env) , convert_index_to_action(pos_index, env)
            pos_index += 1
        while neg_index >= 0:
            neg_position = env.possible_start_positions[neg_index]
            if env.coverage_channel_out[neg_position[0], neg_position[1]] == 1:
                return convert_position_to_action(neg_position, env) , convert_index_to_action(neg_index, env)
            neg_index -= 1
    return convert_position_to_action(env_position, env) , convert_index_to_action(env_index, env)

            

def convert_action(action, env):
    tanh_action = torch.tanh(action)
    index_to_pos = len(env.possible_start_positions) * (tanh_action + 1) / 2
    env_position = env.possible_start_positions[
        (
            round(
                torch.clip(index_to_pos, 0, len(env.possible_start_positions) - 1)
                .detach()
                .cpu()
                .numpy()[0]
            )
        )
    ]
    converted_action = np.array(
        [
            env_position[0] / env.scaling + env.x_min,
            env_position[1] / env.scaling + env.y_min,
        ]
    )
    return converted_action


def convert_index_to_action(index, env):
    index_scaled = index / len(env.possible_start_positions) * 2 - 1
    # # inverse tanh
    index_to_action = math.atanh(np.clip(index_scaled, -0.99, 0.99))
    return torch.tensor(index_to_action)


def get_action(policy_output, env):
    action_mean, action_std = (
        policy_output[:, 0],
        policy_output[:, 1]
    )
    action_std = torch.exp(action_std)
    dist = distributions.Normal(action_mean, action_std)
    action = 0

    action = dist.sample()
    raw_action = action
    action_out = action
    # for i in range(len(action)):
    #     action_out[i], raw_action[i] = transform_action(action[i], env)
    action_out , raw_action = transform_action(action, env)
    log_prob = dist.log_prob(raw_action)

    return action_out, log_prob
# def get_action(policy_output, env):
#     action_mean, action_std = (
#         policy_output[:, 0],
#         policy_output[:, 1]
#     )
#     action_std = torch.exp(action_std)
#     dist = distributions.Normal(action_mean, action_std)
#     action = 0

#     action = dist.sample()
#     log_prob = dist.log_prob(action)

#     return convert_action(action, env), log_prob

# def get_log_prob_entropy(policy_output, env):
#     action_mean, action_std = (policy_output[:, 0], policy_output[:, 1])
#     action_std = torch.exp(action_std)
#     dist = distributions.Normal(action_mean, action_std)

#     action = dist.sample()
#     raw_action = action
#     # print(len(action))
#     for i in range(len(action)):
#         _, raw_action[i] = transform_action(action, env, i)
#     log_prob = dist.log_prob(raw_action)
#     entropy = dist.entropy().mean()

#     return log_prob, entropy

def get_log_prob_entropy(policy_output, env):
    action_mean, action_std = (policy_output[:, 0], policy_output[:, 1])
    action_std = torch.exp(action_std)
    dist = distributions.Normal(action_mean, action_std)

    action = dist.sample()
    # raw_action = action
    # # print(len(action))
    # for i in range(len(action)):
    #     _, raw_action[i] = transform_action(action, env, i)
    log_prob = dist.log_prob(action)
    entropy = dist.entropy().mean()

    return log_prob, entropy


def compute_gae(
    next_value, rewards, masks, values, gamma=args["gamma"], lam=args["lambda"]
):
    values = values + [next_value]
    gae = 0
    returns = torch.zeros(len(rewards))
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns[step] = gae + values[step]
    return returns.detach()



def ppo_iter(mini_batch_size, states, log_probs, returns, advantage):
    batch_size = states.shape[0]
    # shuffle the states
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    for i in range(batch_size // mini_batch_size):
        # get the indices of the current mini-batch
        ind = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
        yield states[ind], log_probs[ind], returns[ind], advantage[ind]


def preprocess_input(x):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x.float()),
        ]
    )
    return preprocess(x).unsqueeze(0)


def preprocess_input_batched(x):
    # x is a batch of tensors of size (batch_size, 224, 224, 3), we want to normalize and convert values to float
    preprocess = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize(
                mean=np.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, -1)),
                std=np.array([0.229, 0.224, 0.225]).reshape((1, 1, 1, -1)),
            ),
        ]
    )
    return preprocess(x).permute(0, 3, 1, 2)


def train(actor, critic, actor_optim, critic_optim, env, args):
    device = args["device"]
    actor = actor.to(device)
    critic = critic.to(device)
    for i in range(args["num_episodes"]):
        actor.eval()
        critic.eval()
        state, info = env.reset()
        # state_buffer_file = open("state_buffer.json", "w")
        # buffers
        state_buffer = []
        action_buffer = []
        log_prob_buffer = []
        reward_buffer = []
        mask_buffer = []
        value_buffer = []
        total_reward = 0
        num_invalid = 0
        num_blocked = 0
        done = False
        curr_step = 0

        prog_bar = tqdm()
        while not done and curr_step < args["max_episode_length"]:
            # change state to double np array
            if len(state) == 2:
                state = state[0]

            state_buffer.append(torch.from_numpy(state))
            state = np.array(state, dtype=np.double)
            state = preprocess_input(state).to(device)
            policy, value = actor(state), critic(state)

            action, log_prob = get_action(policy.detach().cpu(), env)
            next_state, reward, done, truncated, info = env.step(
                action
            )
            mask = 1 if not done else 0

            total_reward += reward

            action_buffer.append(action)
            log_prob_buffer.append(log_prob.detach())
            reward_buffer.append(reward)
            mask_buffer.append(mask)
            value_buffer.append(value.detach().cpu())
            state = next_state
            curr_step += 1
            prog_bar.update(1)

        # state_buffer_file.close()
        next_state = preprocess_input(next_state).to(device)
        next_value = critic(next_state).detach().cpu()
        returns = compute_gae(next_value, reward_buffer, mask_buffer, value_buffer)

        states = torch.stack(state_buffer)
        # actions = torch.stack(action_buffer)
        log_probs = torch.cat(log_prob_buffer)
        advantages = returns - torch.tensor(value_buffer)
        # print(actions[0])
        # plt.imshow(env.get_observation() * 255)
        env.render()
        time.sleep(2)
        # plt.pause(1)
        print(f"\nEpisode {i} Total Reward: {total_reward}\n")
        ratio_coverage = env.curr_coverage / env.coverage_possible
        print(f"Coverage ratio: {ratio_coverage} \n")
        total_time = env.curr_time
        print(f"Total time: {total_time} \n")
        # PPO update
        actor.train()
        critic.train()
        for _ in tqdm(range(args["epochs"])):
            for (
                batch_state,
                batch_log_probs,
                batch_return,
                batch_advantage,
            ) in ppo_iter(
                args["batch_size"], states, log_probs, returns, advantages
            ):
                actor_optim.zero_grad()
                critic_optim.zero_grad()
                if len(batch_state.shape) == 3:
                    batch_state = np.array(batch_state, dtype=np.double)
                    batch_state = preprocess_input(batch_state).to(device)
                # batched
                else:
                    batch_state = preprocess_input_batched(batch_state).to(device)
                policy, value = actor(batch_state), critic(batch_state)

                log_prob, entropy = get_log_prob_entropy(policy, env)

                ratio = torch.exp(log_prob - batch_log_probs.to(device))
                surr1 = ratio * batch_advantage.to(device)
                surr2 = (
                    torch.clamp(ratio, 1.0 - args["eps_clip"], 1.0 + args["eps_clip"])
                    * batch_advantage.to(device)
                )

                actor_loss = (-torch.min(surr1, surr2)).mean() - args[
                    "entropy_coef"
                ] * entropy
                critic_loss = (batch_return.to(device) - value).pow(2).mean()

                actor_loss.backward()
                actor_optim.step()
                critic_loss.backward()
                critic_optim.step()
        
        # torch.save(actor.state_dict(), "checkpoints_area/actor_{}.pth".format(i))
        # torch.save(critic.state_dict(), "checkpoints_area/critic_{}.pth".format(i))


def eval(actor, env):
    state, info = env.reset()
    done = False
    total_reward = 0
    curr_step = 0
    prog_bar = tqdm(total=1000)
    milestone = 0.1
    num_invalid = 0
    rewards = []

    while not done and curr_step < 1000:
        state = np.array(state, dtype=np.double)
        state = preprocess_input(state).to(device)
        policy = actor(state)
        action, _ = get_action(policy, env)
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        if env.curr_coverage / env.coverage_possible >= milestone:
            print(f"Coverage: {milestone} hit at step {curr_step}")
            milestone += 0.1
        curr_step += 1
        prog_bar.update(1)
        if env.curr_coverage / env.coverage_possible >= milestone:
            print(f"Coverage: {milestone} hit at step {curr_step}")
            milestone += 0.1
        # env.render()
        # time.sleep(2)
    print(f"Total Reward: {total_reward}\n")


# actor.load_state_dict(torch.load("actor_9.pth", map_location=device))
# actor.eval()
# eval(actor, env)

env = CCPP_Env(agent_dims=[0.2, 0.2], agent_loc=[ 0, 8],map_file="maps/CDL_Ground.vectormap.json",scaling=6, coverage_radius=2)
actor = CCPP_Actor()
critic = CCPP_Critic()
# actor.load_state_dict(torch.load("checkpoints_area/actor_3.pth"))
# critic.load_state_dict(torch.load("checkpoints_area/critic_3.pth"))
actor = actor.float()
critic = critic.float()
actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr)


train(actor, critic, actor_optim, critic_optim, env, args)