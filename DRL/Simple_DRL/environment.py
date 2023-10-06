import numpy as np

class bribe_env:

    def seed(self, sd):
        
        np.random.seed(sd)

    def __init__(self, T, v_dep, unrelated_bid, A_bid, B_bid, punishment, decay):

        self._action_space_n = [4,3]
        self._state_vector_n = 4
        self.slot_count = 1
        self.T_slot = T
        self.v_dep = v_dep
        self.unrelated_bid = unrelated_bid
        self.A_bid = A_bid
        self.B_bid = B_bid
        self.punishment = punishment
        self.decay = decay

    def reset(self):

        next_state = [0]*self._state_vector_n
        self.slot_count = 1
        return next_state

    def unmapped_step(self, state, action, id):

        next_state = state
        reset_flag = 0
        reward_B = 0
        reward_M = 0

        if (id == 1):

            if (action == 0):
                next_state = state
            elif (action == 1):
                if (state[2]<1):
                    next_state[2] = self.unrelated_bid
            elif (action == 2):
                if (state[2]<2):
                    next_state[2] = self.A_bid - self.decay
            elif (action == 3):
                if (state[2]<3):
                    next_state[2] = self.B_bid

        elif (id == 2):

            if (action == 0):
                next_state = state
                reward_B = 0
                reward_M = 1
            elif (action == 1 and state[3] == 0 and state[1] != 0):
                next_state[3] = 1
                reward_B = 0
                if (state[2] > state[1]):
                    reward_M = state[1] - state[2]
                else:
                    reward_M = state[1]
            elif (action == 2 and state[3] == 0 and state[0] == 1 and state[2] != 0):
                next_state = state
                reward_B = self.v_dep - state[2]
                reward_M = state[2]

            self.slot_count += 1
            
        if (id == 1 and self.slot_count == 1):
            ledger = np.random.rand(1)
            if (ledger < 0.2):
                next_state[1] = 0
            else:
                next_state[1] = self.A_bid

        if (id == 2):
            if (self.slot_count == self.T_slot):
                next_state[0] = 1
            elif (self.slot_count == self.T_slot+1):
                if (reward_B == 0):
                    reward_B = self.punishment
                reset_flag = 1

        return next_state, reset_flag, reward_B, reward_M

    def is_legal_move(self, s, a, id):

        legal = 0

        if (id == 1 and (a == 0 or a == 1 or a == 2 or a == 3)):
            legal = 1

        elif (id == 2):
            if (a == 0):
                legal = 1
            elif (a == 1 and s[3] == 0 and s[1] != 0):
                legal = 1
            elif (a == 2 and s[3] == 0 and s[0] == 1 and s[2] != 0):
                legal = 1
        
        return legal

    def step(self, state, action, id):

        if (self.is_legal_move(state, action, id) == 1):
            s1, d, r_B, r_M = self.unmapped_step(state, action, id)
            return s1, r_B, r_M, d, action

        for i in range(self._action_space_n[id-1]):
            if (self.is_legal_move(state, i, id) == 1):
                s1, d, r_B, r_M = self.unmapped_step(state, i, id)
                return s1, r_B, r_M, d, i