import numpy as np

class bribe_env_B1:

    def __init__(self, number_miners, T, reverse_bribe_high, reverse_bribe_low, A_bid_high, A_bid_low):

        self.number_miners = number_miners
        self.T_slot = T
        self.reverse_bribe_high = reverse_bribe_high
        self.reverse_bribe_low = reverse_bribe_low
        self.A_bid_high = A_bid_high
        self.A_bid_low = A_bid_low

        self.mining_powers = np.zeros(number_miners)
        self.miner_collude_with = 0
        self.slot = 1
        self.negative_reward = 0
        #self.A_can_bid = 0

    def reset(self):

        self.slot = 1
        self.negative_reward = 0
        #self.A_can_bid = np.random.rand(1)
        mining_powers = np.random.rand(self.number_miners)
        self.mining_powers = mining_powers / np.sum(mining_powers)
        self.miner_collude_with = np.random.choice(self.number_miners)
        reverse_bribe_value = np.random.randint(self.reverse_bribe_low, self.reverse_bribe_high)
        next_state = [-1, reverse_bribe_value, -1]

        return next_state

    def step(self, state, action):

        next_state = state
        reset_flag = 0
        reward = 0
        miner_chosen = np.random.choice(len(self.mining_powers), p = self.mining_powers)

        if (miner_chosen == self.miner_collude_with):
            
            if (self.slot >= self.T_slot and state[0] == -1):
                next_state[0] = 1
                reward = state[1] - self.negative_reward
            else:
                if (action > 0):
                    self.negative_reward += action * (-1)
        
        else:

            if (state[0] == -1):
                if (state[2] != -1):
                    if (action > state[2]):
                        self.negative_reward += action * (-1)
                    else:
                        next_state[0] = 1
            else:
                if (action > 0):
                    self.negative_reward += action * (-1)

        self.slot += 1

        if (self.slot == self.T_slot + 6):
            reset_flag = 1

#        if (self.slot > 4 and self.slot < self.T_slot and state[0] == -1 and self.A_can_bid < 0.9):
        if (self.slot > 4 and self.slot < self.T_slot and state[0] == -1):

            ledger = np.random.rand(1)
            if (ledger < 0.8):
                A_bid = np.random.randint(self.A_bid_low, self.A_bid_high)
                if (A_bid > state[2]):
                    next_state[2] = A_bid

        return next_state, reset_flag, reward