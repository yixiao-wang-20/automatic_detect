import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional
from agent.DQN import DQN_net
from agent.DDPG import DDPG_net

class experience_buffer():

    def __init__(self, buffer_entry_size, buffer_size=512):
        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_entry_size = buffer_entry_size

    def add(self, experience):
        #print(experience)
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        #print(self.buffer)
        size = min(size, len(self.buffer))
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, self.buffer_entry_size])
        #return random.sample(self.buffer, size)

class agent():

    def __init__(self, h_size, buffer_entry_size, action_space_n_discrete, action_space_n_continuous, state_space_n, max_action, batch_size, tau, y, LR):

        self.action_space_n_discrete = action_space_n_discrete
        self.action_space_n_continuous = action_space_n_continuous
        self.state_space_n = state_space_n
        self.batch_size = batch_size
        self.DQN_nets = []
        for i in range(len(action_space_n_discrete)):
            self.DQN_nets.append(DQN_net(h_size, action_space_n_discrete[i], state_space_n, batch_size, tau, y, LR))
        self.DDPG_net = DDPG_net(h_size, action_space_n_continuous, state_space_n, max_action, tau, y, LR)

        self.myBuffer = experience_buffer(buffer_entry_size = buffer_entry_size)

    def act_epsilon_greedy(self, s, e=0):

        action = []
        for i in range(len(self.DQN_nets)):
            action.append(self.DQN_nets[i].select_action(s, e))
        action.extend(self.DDPG_net.select_action(s, e)[0])
        return action

    def update(self):

        trainBatch = self.myBuffer.sample(self.batch_size)

        s1 = trainBatch[:, 0:self.state_space_n]
        a = trainBatch[:, self.state_space_n:self.state_space_n + len(self.action_space_n_discrete) + self.action_space_n_continuous]
        reward = trainBatch[:, self.state_space_n + len(self.action_space_n_discrete) + self.action_space_n_continuous]
        s2 = trainBatch[:, self.state_space_n + len(self.action_space_n_discrete) + self.action_space_n_continuous + 1:2 * self.state_space_n + len(self.action_space_n_discrete) + self.action_space_n_continuous + 1]
        done = trainBatch[:, 2 * self.state_space_n + len(self.action_space_n_discrete) + self.action_space_n_continuous + 1]

        for i in range(len(self.DQN_nets)):
            self.DQN_nets[i].update(s1, a[:, i], reward, s2, done)
        self.DDPG_net.update(s1, a[:, len(self.DQN_nets):], reward, s2, done)

    def save(self, directory, name):

        for i in range(len(self.DQN_nets)):
            torch.save(self.DQN_nets[i].mainQN.state_dict(), directory + name + '_DQN_net_' + str(i) + '.params')
        torch.save(self.DDPG_net.actor.state_dict(), directory + name + '_DDPG_net_actor.params')
        torch.save(self.DDPG_net.critic.state_dict(), directory + name + '_DDPG_net_critic.params')

    def load(self, directory, name):

        for i in range(len(self.DQN_nets)):
            self.DQN_nets[i].mainQN.load_state_dict(torch.load(directory + name + '_DQN_net_' + str(i) + '.params'))
        self.DDPG_net.actor.load_state_dict(torch.load(directory + name + '_DDPG_net_actor.params'))
        self.DDPG_net.critic.load_state_dict(torch.load(directory + name + '_DDPG_net_critic.params'))