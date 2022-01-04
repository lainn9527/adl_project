import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
import gym
import numpy as np


class Actor(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size*4),
            nn.Tanh(),
            nn.Linear(input_size*4, input_size*2),
            nn.Tanh(),
            nn.Linear(input_size*2, output_size),
            nn.Identity(),
        )
        self.embedding = None
    
    def forward(self, x):
        linear_output = self.net(x)
        return F.softmax(self.embedding(linear_output), dim = -1)
        


class Critic(nn.Module):
    def __init__(self, input_size: list):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.Tanh(),
            nn.Linear(input_size*2, input_size),
            nn.Tanh(),
            nn.Linear(input_size, 1),
            nn.Identity()
        )
    
    def forward(self, x):
        return self.net(x)

def get_action(policy, state):
    state = torch.from_numpy(state).float()
    prob = policy(state)
    m = Categorical(prob)
    action = m.sample()
    return action.item(), m.log_prob(action)

def get_value(critic, state):
    state = torch.from_numpy(state).float()
    return critic(state)

def train(env_name,
          discount_rate,
          lr_actor,
          lr_critic,
          num_epoch = 100,
          num_step = 1000,
          max_step = 2000):
    
    eps = 1e-8

    # env
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_num = env.action_space.n

    # model
    actor = Actor(input_size = obs_dim, output_size = act_num)
    critic = Critic(input_size = obs_dim)

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr = lr_actor)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr = lr_critic)

    for epoch in range(num_epoch):
        actions = []
        log_probs = []
        values = []
        rewards = []
        returns = []
        ep_rewards = []
        ep_lengths = []
        ep_length = 0

        state = env.reset()
        for t in range(num_step):
            # interact with env
            # select action
            action, log_prob = get_action(actor, state)
            value = get_value(critic, state)


            # interact and get env info
            state, reward, done, _  = env.step(action)

            # store infos
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            ep_length += 1
            if done:
                ep_rewards.append(sum(rewards))
                # calculate returns by reward of env
                for i in range(len(rewards)-2, -1, -1):
                    rewards[i] += discount_rate * rewards[i+1]
                returns += rewards
                ep_lengths.append(ep_length)

                rewards = []
                state = env.reset()
                ep_length = 0

        # update model by episode (question: update by episode or by predefined number of step?)
        actor_losses = []
        critic_losses = []

        # normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # calculate the loss
        for log_prob, value, _return in zip(log_probs, values, returns):
            actor_losses.append(-log_prob * (_return - value))
            critic_losses.append(F.smooth_l1_loss(value, torch.tensor([_return])))

        # update
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        loss = torch.stack(actor_losses).mean() + torch.stack(critic_losses).mean()
        loss.backward()
        optimizer_actor.step()
        optimizer_critic.step()
        print(f'Episode: {epoch}, Average reward: {sum(ep_rewards)/len(ep_rewards):7.5}, Actor loss: {sum(actor_losses).item()/len(actor_losses):7.5}, Critic loss: {sum(critic_losses)/len(critic_losses):7.5}')





if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    # parser.add_argument('--render', action='store_true')
    # parser.add_argument('--lr', type=float, default=1e-2)
    # args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name = 'CartPole-v0',
          discount_rate = 0.99,
          lr_actor = 0.01,
          lr_critic = 0.01,
          num_epoch = 100,
          num_step = 2000,
          max_step = 2000)