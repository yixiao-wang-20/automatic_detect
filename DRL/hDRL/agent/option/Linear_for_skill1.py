import numpy as np
import torch
import torch.nn as nn

class linear_network(nn.Module):

    def __init__(self, h_size, input_size):

        super(linear_network, self).__init__()

        self.out1 = nn.Sequential(
            nn.Linear(input_size, int(h_size / 2)),
            nn.BatchNorm1d(int(h_size / 2)),
            nn.ReLU(),
            nn.Linear(int(h_size / 2), int(h_size / 4)),
            nn.BatchNorm1d(int(h_size / 4)),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.out2 = nn.Sequential(
            nn.Linear(int(h_size / 4), 2)
        )

    def forward(self, x):

        output = self.out1(x)
        output = self.out2(output)

        return nn.LogSoftmax(dim=-1)(output)

class linear_net_for_skill1():  

    def __init__(self, h_size, input_size, LR):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.predict_net = linear_network(h_size, input_size).to(self.device)

        self.optimizer = torch.optim.Adam(self.predict_net.parameters(), lr = LR)

    def predict_the_reward(self, s):

        self.predict_net.eval()
        Q = np.argmax(self.predict_net(torch.Tensor(np.array(s)).unsqueeze(dim = 0).to(self.device)).squeeze(dim = 0).detach().cpu().numpy())

        return Q

    def update(self, input_vector, lable):

        self.predict_net.train()
        input_vector = torch.Tensor(input_vector).to(self.device)
        lable = torch.Tensor(lable).to(self.device)
        pre = self.predict_net(input_vector)
        loss = nn.NLLLoss()(pre, lable.view(-1).long())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()