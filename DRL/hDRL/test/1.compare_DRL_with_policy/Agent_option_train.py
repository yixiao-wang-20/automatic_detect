import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional
import sys

sys.path.append('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\_code\\agent\\option')
from Linear import linear_net
from Linear_for_skill1 import linear_net_for_skill1

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

    def __init__(self, h_size, input_size_B1_1, input_size_B1_2, input_size_B2, input_size_B_collusion_1, batch_size, LR):

        self.batch_size = batch_size
        self.linear_net_B1_1 = linear_net(h_size, input_size_B1_1, LR)
        self.linear_net_B1_2 = linear_net_for_skill1(h_size, input_size_B1_2, LR)
        self.linear_net_B2 = linear_net(h_size, input_size_B2, LR)
        self.linear_net_B_collusion_1 = linear_net(h_size, input_size_B_collusion_1, LR)
        self.myBuffer_B1_1 = experience_buffer(buffer_entry_size = input_size_B1_1 + 1)
        self.myBuffer_B1_2 = experience_buffer(buffer_entry_size = input_size_B1_2 + 1)
        self.myBuffer_B2 = experience_buffer(buffer_entry_size = input_size_B2 + 1)
        self.myBuffer_B_collusion_1 = experience_buffer(buffer_entry_size = input_size_B_collusion_1 + 1)


    def predict(self, s, net_chosen):

        if net_chosen == 'B1_1':
            return self.linear_net_B1_1.predict_the_reward(s)
        elif net_chosen == 'B1_2':
            return self.linear_net_B1_2.predict_the_reward(s)
        elif net_chosen == 'B2':
            return self.linear_net_B2.predict_the_reward(s)
        else:
            return self.linear_net_B_collusion_1.predict_the_reward(s)

    def update(self, net_chosen):

        if net_chosen == 'B1_1':

            trainBatch = self.myBuffer_B1_1.sample(self.batch_size)
            reward_expected = trainBatch[:, 0:1]
            input_vector = trainBatch[:, 1:]
            loss = self.linear_net_B1_1.update(input_vector, reward_expected)

        elif net_chosen == 'B1_2':

            trainBatch = self.myBuffer_B1_2.sample(self.batch_size)
            reward_expected = trainBatch[:, 0]
            input_vector = trainBatch[:, 1:]
            loss = self.linear_net_B1_2.update(input_vector, reward_expected)

        elif net_chosen == 'B2':

            trainBatch = self.myBuffer_B2.sample(self.batch_size)
            reward_expected = trainBatch[:, 0:1]
            input_vector = trainBatch[:, 1:]
            loss = self.linear_net_B2.update(input_vector, reward_expected)

        else:

            trainBatch = self.myBuffer_B_collusion_1.sample(self.batch_size)
            reward_expected = trainBatch[:, 0:1]
            input_vector = trainBatch[:, 1:]
            loss = self.linear_net_B_collusion_1.update(input_vector, reward_expected)

        return loss

    def save(self, directory, net_chosen):

        if net_chosen == 'B1_1':
            torch.save(self.linear_net_B1_1.predict_net.state_dict(), directory + 'predict_net_B1_1.params')
        elif net_chosen == 'B1_2':
            torch.save(self.linear_net_B1_2.predict_net.state_dict(), directory + 'predict_net_B1_2.params')
        elif net_chosen == 'B2':
            torch.save(self.linear_net_B2.predict_net.state_dict(), directory + 'predict_net_B2.params')
        else:
            torch.save(self.linear_net_B_collusion_1.predict_net.state_dict(), directory + 'predict_net_B_collusion_1.params')        

    def load(self, directory, net_chosen):

        if net_chosen == 'B1_1':
            self.linear_net_B1_1.predict_net.load_state_dict(torch.load(directory + 'predict_net_B1_1.params'))
        elif net_chosen == 'B1_2':
            self.linear_net_B1_2.predict_net.load_state_dict(torch.load(directory + 'predict_net_B1_2.params'))
        elif net_chosen == 'B2':
            self.linear_net_B2.predict_net.load_state_dict(torch.load(directory + 'predict_net_B2.params'))
        else:
            self.linear_net_B_collusion_1.predict_net.load_state_dict(torch.load(directory + 'predict_net_B_collusion_1.params'))                