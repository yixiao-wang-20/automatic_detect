import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional
import sys

sys.path.append('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\_code\\agent\\skill')
from DDPG import DDPG_net

class experience_buffer():

    def __init__(self, buffer_entry_size, buffer_size=512):
        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_entry_size = buffer_entry_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        #print(self.buffer)
        size = min(size, len(self.buffer))
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, self.buffer_entry_size])
        #return random.sample(self.buffer, size)

class agent():

    def __init__(self, h_size, buffer_entry_size, action_space_n_continuous, state_space_n, max_action, batch_size, tau, y, LR):

        self.action_space_n_continuous = action_space_n_continuous
        self.state_space_n = state_space_n
        self.batch_size = batch_size
        self.DDPG_net = DDPG_net(h_size, action_space_n_continuous, state_space_n, max_action, tau, y, LR)
        self.myBuffer = experience_buffer(buffer_entry_size = buffer_entry_size)

    def act_epsilon_greedy(self, s, e=0):

        return self.DDPG_net.select_action(s, e)[0][0]

    def update(self):

        trainBatch = self.myBuffer.sample(self.batch_size)

        s1 = trainBatch[:, 0:self.state_space_n]
        a = trainBatch[:, self.state_space_n:self.state_space_n + self.action_space_n_continuous]
        reward = trainBatch[:, self.state_space_n  + self.action_space_n_continuous]
        s2 = trainBatch[:, self.state_space_n  + self.action_space_n_continuous + 1:2 * self.state_space_n  + self.action_space_n_continuous + 1]
        done = trainBatch[:, 2 * self.state_space_n  + self.action_space_n_continuous + 1]

        self.DDPG_net.update(s1, a, reward, s2, done)

    def save(self, directory):

        torch.save(self.DDPG_net.actor.state_dict(), directory + 'actor.params')
        torch.save(self.DDPG_net.critic.state_dict(), directory + 'critic.params')

    def load(self, directorys):

        self.DDPG_net.actor.load_state_dict(torch.load(directory + 'actor.params'))
        self.DDPG_net.critic.load_state_dict(torch.load(directory + 'critic.params'))