import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional

class experience_buffer():

    def __init__(self, buffer_entry_size, buffer_size=640):
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

class Qnetwork(nn.Module):

    def __init__(self, h_size, action_space_n):

        super(Qnetwork, self).__init__()

        self.out1 = nn.Sequential(
            nn.Linear(4, h_size),
            #nn.BatchNorm1d(h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            #nn.BatchNorm1d(h_size),
            nn.ReLU(),
            #nn.Dropout(0.1)
        )
        self.out_A = nn.Sequential(
            nn.Linear(h_size // 2, action_space_n)
        )
        self.out_V = nn.Sequential(
            nn.Linear(h_size // 2, 1)
        )
        self.h_size=h_size

    def forward(self, x):

        hidden = self.out1(x)
        Advantage = self.out_A(hidden[:,0:self.h_size // 2])
        Value = self.out_V(hidden[:,self.h_size // 2:self.h_size])
        V_af=torch.add(Value,torch.neg(torch.mean(Value,dim=0)))
        output = torch.add(V_af,Advantage)

        return output  

class agent_Miner():

    def __init__(self, h_size, buffer_entry_size, action_space_n, batch_size, tau, y):

        self.action_space_n = action_space_n
        self.batch_size = batch_size
        self.tau = tau
        self.y = y

        self.mainQN = Qnetwork(h_size, action_space_n)
        self.targetQN = Qnetwork(h_size, action_space_n)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.mainQN = self.mainQN.to(self.device)
        self.targetQN = self.targetQN.to(self.device)

        self.optimizer = torch.optim.Adam(self.mainQN.parameters(),lr=0.001)

        self.myBuffer = experience_buffer(buffer_entry_size = buffer_entry_size)

    def updateTarget(self):

        main_weights = list(self.mainQN.parameters())
        target_weights = list(self.targetQN.parameters())

        for main_weight, target_weight in zip(main_weights, target_weights):
            target_weight=(main_weight * torch.tensor(self.tau) + target_weight * torch.tensor(1 - self.tau))

    def act_epsilon_greedy(self, s, e=0):

        if np.random.rand(1) < e:
            a = np.random.choice(self.action_space_n)
        else:
            Q = self.mainQN(torch.Tensor(np.array([s])).to(self.device))
            a = np.argmax(Q.detach().cpu().numpy())

        return a


    def train(self):

        trainBatch = self.myBuffer.sample(self.batch_size)

        Q1_out = self.mainQN(torch.Tensor(trainBatch[:, 6:10]).to(self.device))
        Q1= torch.argmax(Q1_out, dim=1).cpu().numpy()
        Q2 = self.targetQN(torch.Tensor(trainBatch[:, 6:10]).to(self.device))
        end_multiplier = torch.tensor(-(trainBatch[:, 10] - 1))
        doubleQ = torch.zeros(self.batch_size, dtype=torch.float32)
        for i in range(self.batch_size):
            doubleQ[i] = (Q2[i, Q1[i]])      
        targetQ = torch.tensor(trainBatch[:, 5]) + (torch.tensor(self.y) * doubleQ * end_multiplier)
        actions = torch.tensor(trainBatch[:, 4]).to(torch.int64)
        actions_onehot = functional.one_hot(actions, num_classes=self.action_space_n)
        Qout_train = self.mainQN(torch.Tensor(trainBatch[:, 0:4]).to(self.device))
        Q_train = Qout_train*actions_onehot.to(self.device)
        Q_train = torch.sum(Q_train,dim=1)
        td_error = torch.pow((targetQ.to(self.device) - Q_train),2)
        loss = torch.mean(td_error)
        
        loss.backward()
        self.optimizer.step()

        self.updateTarget()