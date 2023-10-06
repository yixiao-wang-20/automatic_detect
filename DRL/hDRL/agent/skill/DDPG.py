import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):

    def __init__(self, state_space_n, action_space_n, max_action, h_size):

        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_space_n, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.l3 = nn.Linear(h_size, action_space_n)
        self.max_action = max_action

    def forward(self,x):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.hardtanh(self.l3(x), min_val = -1, max_val = self.max_action)
        return x

class Critic(nn.Module):

    def __init__(self, state_space_n, action_space_n, h_size):

        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_space_n+action_space_n, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.l3 = nn.Linear(h_size, 1)

    def forward(self,x,u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class DDPG_net():

    def __init__(self, h_size, action_space_n, state_space_n, max_action, tau, y, LR):

        self.action_space_n = action_space_n
        self.max_action = max_action
        self.tau = tau
        self.y = y
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_space_n, action_space_n, max_action, h_size).to(self.device)
        self.actor_target = Actor(state_space_n, action_space_n, max_action, h_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), LR)

        self.critic = Critic(state_space_n, action_space_n, h_size).to(self.device)
        self.critic_target = Critic(state_space_n, action_space_n, h_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), LR)

    def select_action(self, state, exploration_noise):

        if (exploration_noise == 1):
            action = np.random.rand(self.action_space_n) * self.max_action

        try:
            action = self.actor(torch.Tensor(np.array([state])).to(self.device)).detach().cpu().numpy() * (np.random.normal(0., exploration_noise, size = self.action_space_n) + [1.] * self.action_space_n)
        except:
            action = self.actor(torch.Tensor(np.array([state])).to(self.device)).detach().cpu().numpy()

        return action

    def update(self, s1, action, reward, s2, done):

        state = torch.FloatTensor(s1).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(s2).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)

        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + ((1-done) * self.y * target_Q).detach()
        current_Q = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = - self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param,target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param,target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) 