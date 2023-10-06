import numpy as np

class bribe_env_B_collusion_1:

    def __init__(self, number_miners, T, reverse_bribe_high, reverse_bribe_low, A_bid_high, A_bid_low, v_dep, v_col):

        self.number_miners = number_miners
        self.T_slot = T
        self.reverse_bribe_high = reverse_bribe_high
        self.reverse_bribe_low = reverse_bribe_low
        self.A_bid_high = A_bid_high
        self.A_bid_low = A_bid_low
        self.v_dep = v_dep
        self.v_col = v_col

        self.mining_powers = np.zeros(number_miners)
        self.miner_collude_with = 0
        self.slot = 1

    def reset(self):

        self.slot = 1
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

        if (self.slot < self.T_slot and state[0] == -1 and state[2] != -1):

            if (miner_chosen == self.miner_collude_with):
                next_state[0] = 1
                reward = state[1]
            else:
                next_state[0] = 0

        if (self.slot == self.T_slot and state[0] == 0):
            reward = self.v_col

        self.slot += 1

        if (self.slot == self.T_slot + 2):
            reset_flag = 1

        if (self.slot > 2 and self.slot < self.T_slot and state[0] == -1):

            ledger = np.random.rand(1)
            if (ledger < 0.8):
                A_bid = np.random.randint(self.A_bid_low, self.A_bid_high)
                if (A_bid > state[2]):
                    next_state[2] = A_bid

        return next_state, reset_flag, reward