import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional

class Qnetwork(nn.Module):

    def __init__(self, h_size, action_space_n, state_space_n):

        super(Qnetwork, self).__init__()

        self.out1 = nn.Sequential(
            nn.Linear(state_space_n, h_size),
            #nn.BatchNorm1d(h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            #nn.BatchNorm1d(h_size),
            nn.ReLU()
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
        Advantage = self.out_A(hidden[:, 0:self.h_size // 2])
        Value = self.out_V(hidden[:, self.h_size // 2:self.h_size])
        V_af=torch.add(Value,torch.neg(torch.mean(Value, dim=0)))
        output = torch.add(V_af, Advantage)

        return output  

class DQN_net():

    def __init__(self, h_size, action_space_n, state_space_n, batch_size, tau, y, LR):

        self.action_space_n = action_space_n
        self.batch_size = batch_size
        self.tau = tau
        self.y = y

        self.mainQN = Qnetwork(h_size, action_space_n, state_space_n)
        self.targetQN = Qnetwork(h_size, action_space_n, state_space_n)
        self.targetQN.load_state_dict(self.mainQN.state_dict())

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.mainQN = self.mainQN.to(self.device)
        self.targetQN = self.targetQN.to(self.device)

        self.optimizer = torch.optim.Adam(self.mainQN.parameters(), lr = LR)


    def select_action(self, s, e=0):

        if np.random.rand(1) < e:
            a = np.random.choice(self.action_space_n)
        else:
            Q = self.mainQN(torch.Tensor(np.array([s])).to(self.device))
            a = np.argmax(Q.detach().cpu().numpy())

        return a


    def update(self, s1, action, reward, s2, done):

        self.optimizer.zero_grad()
        Q1_out = self.mainQN(torch.Tensor(s2).to(self.device))
        Q1= torch.argmax(Q1_out, dim=1).cpu().numpy()
        Q2 = self.targetQN(torch.Tensor(s2).to(self.device))
        end_multiplier = torch.tensor(-(done - 1))
        doubleQ = torch.zeros(self.batch_size, dtype=torch.float32)
        for i in range(self.batch_size):
            doubleQ[i] = (Q2[i, Q1[i]])      
        targetQ = torch.tensor(reward) + (torch.tensor(self.y) * doubleQ * end_multiplier)
        actions = torch.tensor(action).to(torch.int64)
        actions_onehot = functional.one_hot(actions, num_classes=self.action_space_n)
        Qout_train = self.mainQN(torch.Tensor(s1).to(self.device))
        Q_train = Qout_train*actions_onehot.to(self.device)
        Q_train = torch.sum(Q_train,dim=1)
        td_error = torch.pow((targetQ.to(self.device) - Q_train),2)
        loss = torch.mean(td_error)
        
        loss.backward()
        self.optimizer.step()

        main_weights = list(self.mainQN.parameters())
        target_weights = list(self.targetQN.parameters())

        for main_weight, target_weight in zip(main_weights, target_weights):
            target_weight=(main_weight * torch.tensor(self.tau) + target_weight * torch.tensor(1 - self.tau))