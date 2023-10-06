import numpy as np

class bribe_env_B2:

    def __init__(self, T, v_dep, v_col, A_bid_high, A_bid_low):

        self.T_slot = T
        self.v_dep = v_dep
        self.v_col = v_col
        self.A_bid_high = A_bid_high
        self.A_bid_low = A_bid_low

        self.slot = 1
        #self.A_can_bid = 0
        self.bid_of_B = -1

    def reset(self):

        self.slot = 1
        #self.A_can_bid = np.random.rand(1)
        self.bid_of_B = -1
        next_state = [-1, -1, -1]

        return next_state

    def step(self, state, action):

        next_state = state
        reset_flag = 0
        reward = 0

        if (action > 0 and action > self.bid_of_B):
            self.bid_of_B = action

#change: reward // state[0]
#action > 0
#state[0] state[1] state[2]
#A发布了  B发布了  未赎回  M赎回所有  r = -v-col
#A发布了  B未发布  未赎回  A赎回v-dep
#A未发布  B发布了  未赎回  已到T周期 赎回v-dep r = v-dep 


        if (state[2] != -1 and self.bid_of_B != -1 and state[0] == -1):
            reward = self.v_col * (-1)
            next_state[0] = 1

        if (state[2] != -1 and self.bid_of_B == -1 and state[0] == -1):
            next_state[0] = 1

        if (state[2] == -1 and self.bid_of_B != -1 and state[0] == -1 and state[1] == 1):
            reward = self.v_dep
            next_state[0] = 1
        
        self.slot += 1

        if (self.slot == self.T_slot):
            next_state[1] = 1

        if (self.slot == self.T_slot + 6):
            reset_flag = 1

        #if (self.slot > 4 and self.slot < self.T_slot and state[0] == -1 and self.A_can_bid < 0.7):
        if (self.slot < self.T_slot and state[0] == -1):

            ledger = np.random.rand(1)
            if (ledger < 0.8):
                A_bid = np.random.randint(self.A_bid_low, self.A_bid_high)
                if (A_bid > state[2]):
                    next_state[2] = A_bid

        return next_state, reset_flag, reward