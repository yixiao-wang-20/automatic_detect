import numpy as np

class experience_buffer():

    def __init__(self, buffer_entry_size, buffer_size=640):

        pass
        
    def add(self, experience):

        pass

class agent_A():

    def __init__(self, h_size, buffer_entry_size, action_space_n_discrete, action_space_n_continuous, state_space_n, max_action, batch_size, tau, y, LR):

        self.state_space_n = state_space_n - 5
        self.max_action = max_action
        self.myBuffer = experience_buffer(buffer_entry_size = buffer_entry_size)

    def act_epsilon_greedy(self, s, e=0):

        action = 0

        if ((s[0] > 6) and (s[self.state_space_n] == -1)):

            ledger = np.random.rand(1)
            if (ledger < 0.9):
                action = self.max_action * np.random.rand(1).item()

        return action

    def update(self):

        pass

    def save(self, directory, name):

        pass

    def load(self, directory, name):

        pass