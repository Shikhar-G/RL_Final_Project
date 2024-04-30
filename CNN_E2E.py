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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argument parser
args = easydict.EasyDict(
    {
        "batch_size": 16,
        "gamma": 0.99,
        "lambda": 0.95,
        "eps_clip": 0.2,
        "buffer_size": 64,
        "epochs": 10,
        "lr": 5e-5,
        "max_episode_length": 128,
        "num_episodes": 32,
        "entropy_coef": 0.01,
        "enable_cuda": True,
        "device": device,
    }
)


class CCPP_CNN(nn.Module):
    def __init__(self):
        super(CCPP_CNN, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module(
            "conv1", nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        )
        self.features.add_module("tanh1", nn.Tanh(inplace=True))
        self.features.add_module(
            "conv2", nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        )
        self.features.add_module("tanh2", nn.Tanh(inplace=True))
        # self.features.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.features.add_module(
            "conv3", nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)
        )
        self.features.add_module("tanh3", nn.Tanh(inplace=True))
        self.features.add_module("maxpool3", nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        # print(x.dtype)
        x = self.features(x)
        return x


class CCPP_Actor(nn.Module):
    def __init__(self, image_dims):
        super(CCPP_Actor, self).__init__()

        self.resnet = CCPP_CNN().float()
        self.features = nn.Sequential()
        self.features.add_module("flatten", nn.Flatten())
        self.features.add_module(
            "lin1", nn.LazyLinear(image_dims[0] / 2 * image_dims[1] / 2 * 32, 2048)
        )
        self.features.add_module("tanh1", nn.Tanh())
        self.features.add_module("lin2", nn.LazyLinear(2048, 2))

    def forward(self, x):
        out = self.resnet(x)
        out = self.features(out)
        return out


class CCPP_Critic(nn.Module):
    def __init__(self, image_dims):
        super(CCPP_Critic, self).__init__()

        self.resnet = CCPP_CNN().float()
        self.features = nn.Sequential()
        self.features.add_module("flatten", nn.Flatten())
        self.features.add_module(
            "lin1", nn.LazyLinear(image_dims[0] / 2 * image_dims[1] / 2 * 32, 2048)
        )
        self.features.add_module("tanh1", nn.Tanh())
        self.features.add_module("lin2", nn.LazyLinear(2048, 1))

    def forward(self, x):
        out = self.resnet(x)
        out = self.features(out)
        return out


def convert_index_to_action(index):
    pos = env.possible_start_positions(index)
    pos_to_action = pos / len(env.possible_start_action) * 200 - 100
    return pos_to_action


def convert_action(action):
    env.possible_start_positions
    clip_action = torch.clip(action, -100, 100)
    index_to_pos = len(env.possible_start_positions) * (clip_action + 100) / 200
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


def get_action(policy_output):
    action_mean, action_std = (policy_output[:, 0], policy_output[:, 1])
    action_std = torch.exp(action_std)
    dist = distributions.Normal(action_mean, action_std)

    # with probability epsilon, choose a greedy action instead
    epsilon = 0.1
    if np.random.rand() < epsilon:
        for i, state in enumerate(env.possible_start_positions):
            if env.coverage_channel_out[state[0], state[1]] == 1:
                action = convert_index_to_action(i)
    else:
        action = dist.sample()

    log_prob = dist.log_prob(action)

    return convert_action(action), log_prob


def get_log_prob_entropy(policy_output):
    action_mean, action_std = (policy_output[:, 0], policy_output[:, 1])
    action_std = torch.exp(action_std)
    dist = distributions.Normal(action_mean, action_std)

    action = dist.sample()
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


# def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
#     batch_size = states.shape[0]
#     for _ in range(batch_size // mini_batch_size):
#         rand_ids = torch.randint(0, batch_size, (mini_batch_size,))
#         yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[
#             rand_ids
#         ], advantage[rand_ids]


def ppo_iter(mini_batch_size, states, log_probs, returns, advantage):
    batch_size = states.shape[0]
    # shuffle the states
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    i = 0
    while i < batch_size // mini_batch_size:
        # get the indices of the current mini-batch
        ind = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
        yield states[ind], log_probs[ind], returns[ind], advantage[ind]
        i += 1
    if i * mini_batch_size < batch_size:
        ind = indices[i * mini_batch_size :]
        yield states[ind], log_probs[ind], returns[ind], advantage[ind]


def preprocess_input(x):
    return torch.from_numpy(x).float().unsqueeze(0)


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
    best_model = None
    best_reward = -1000000
    for i in tqdm(range(args["num_episodes"])):
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

        prog_bar = tqdm(total=args["max_episode_length"])
        while not done and curr_step < args["max_episode_length"]:
            # change state to double np array
            if len(state) == 2:
                state = state[0]

            state_buffer.append(torch.from_numpy(state))
            state = np.array(state, dtype=np.double)
            state = preprocess_input(state).to(device)
            policy, value = actor(state), critic(state)

            action, log_prob = get_action(policy.detach().cpu())
            next_state, reward, done, truncated, info = env.step(action)
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
        # actions = torch.stack(action_buffer)ls
        log_prob
        log_probs = torch.cat(log_prob_buffer)
        advantages = returns - torch.tensor(value_buffer)
        # print(actions[0])
        # env.render()
        # time.sleep(2)
        env.close()

        # PPO update
        actor.train()
        critic.train()
        for _ in tqdm(range(args["epochs"])):
            for (
                batch_state,
                batch_log_probs,
                batch_return,
                batch_advantage,
            ) in ppo_iter(args["batch_size"], states, log_probs, returns, advantages):
                actor_optim.zero_grad()
                critic_optim.zero_grad()
                if len(batch_state.shape) == 3:
                    batch_state = np.array(batch_state, dtype=np.double)
                    batch_state = preprocess_input(batch_state).to(device)
                # batched
                else:
                    batch_state = preprocess_input_batched(batch_state).to(device)
                policy, value = actor(batch_state), critic(batch_state)

                log_prob, entropy = get_log_prob_entropy(policy)

                ratio = torch.exp(log_prob - batch_log_probs.to(device))
                surr1 = ratio * batch_advantage.to(device)
                surr2 = torch.clamp(
                    ratio, 1.0 - args["eps_clip"], 1.0 + args["eps_clip"]
                ) * batch_advantage.to(device)
                # add entropy term

                actor_loss = (-torch.min(surr1, surr2)).mean() - args[
                    "entropy_coef"
                ] * entropy
                critic_loss = (batch_return.to(device) - value).pow(2).mean()

                actor_loss.backward()
                actor_optim.step()
                critic_loss.backward()
                critic_optim.step()

        # print("Evaluation")
        torch.save(actor.state_dict(), "checkpoints_area/actor_{}.pth".format(i))
        torch.save(critic.state_dict(), "checkpoints_area/critic_{}.pth".format(i))
        # # evaluate the model
        # actor.eval()
        # critic.eval()
        # state, info = env.reset()
        # done = False
        # total_reward = 0
        # curr_step = 0
        # prog_bar = tqdm(total=args["max_episode_length"])
        # num_invalid = 0
        # while not done and curr_step < args["max_episode_length"]:
        #     state = np.array(state, dtype=np.double)
        #     state = preprocess_input(state).to(device)
        #     policy, _ = actor(state), critic(state)
        #     action, _ = get_action(policy)
        #     action = action.detach().cpu().numpy()
        #     next_state, reward, done, truncated, info = env.step(action)
        #     total_reward += reward
        #     if reward == -50:
        #         num_invalid += 1
        #     state = next_state
        #     curr_step += 1
        #     prog_bar.update(1)
        #     # env.render()
        #     # time.sleep(2)
        # env.render()
        # time.sleep(2)
        # print("Number of invalid actions: ", num_invalid)
        # print(f"Episode {i} Total Reward: {total_reward}\n")


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
        action, _ = get_action(policy)
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


env = CCPP_Env(agent_dims=[0.2, 0.2], agent_loc=[7.5, 0])
actor = CCPP_Actor().to(device)
actor = actor.float()

actor.load_state_dict(torch.load("actor_9.pth", map_location=device))
actor.eval()
eval(actor, env)