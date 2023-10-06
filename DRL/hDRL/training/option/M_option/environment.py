import numpy as np

class bribe_env:

    def seed(self, sd):
        
        np.random.seed(sd)

    def __init__(self, _state_vector_n, mining_powers, T, v_dep, v_col, unrelated_bid):

        self._state_vector_n = _state_vector_n
        self.mining_powers = mining_powers
        self.T_slot = T
        self.v_dep = v_dep
        self.v_col = v_col
        self.unrelated_bid = unrelated_bid

    def reset(self):

        miner_chosen = np.random.choice(len(self.mining_powers), p = self.mining_powers)
        next_state = [1, miner_chosen, self.T_slot, self.v_dep, self.v_col, self.unrelated_bid] + self.mining_powers
        next_state = next_state + [-1] * (self._state_vector_n - len(next_state))
        return next_state

    def step(self, state, action, stage):

        next_state = state
        reset_flag = 0
        reward = [0] * (2 + len(self.mining_powers))
        v_dep_mined_by = -1
        v_col_mined_by = -1

        if (stage == 0):

            if ((action > 0) and (action > state[6 + len(self.mining_powers)])):
                next_state[6 + len(self.mining_powers)] = action

        elif (stage == 1):

            if ((action[0] in range(0,len(self.mining_powers))) and (state[12 + 3 * len(self.mining_powers)] == -1) and (state[12 + len(self.mining_powers) + action[0]] != -1) and (state[9 + len(self.mining_powers)] == -1)):
                next_state[12 + 3 * len(self.mining_powers)] = action[0]
                state[12 + 3 * len(self.mining_powers)] = action[0]
                next_state[7 + len(self.mining_powers)] = -1
                state[7 + len(self.mining_powers)] = -1
                if (state[12 + 2 * len(self.mining_powers) + action[0]] == 1):
                    next_state[8 + len(self.mining_powers)] = -1
                    state[8 + len(self.mining_powers)] = -1

            if ((action[1] > 0) and (action[1] > state[7 + len(self.mining_powers)]) and (state[12 + 3 * len(self.mining_powers)] == -1) and (state[9 + len(self.mining_powers)] == -1)):
                next_state[7 + len(self.mining_powers)] = action[1]

            if ((action[2] > 0) and (action[2] > state[8 + len(self.mining_powers)]) and (state[10 + len(self.mining_powers)] == -1)):
                if (state[12 + 3 * len(self.mining_powers)] != -1):
                    if (state[12 + 2 * len(self.mining_powers) + state[12 + 3 * len(self.mining_powers)]] == 0):
                        next_state[8 + len(self.mining_powers)] = action[2]
                else:
                    next_state[8 + len(self.mining_powers)] = action[2]
            
            if ((action[3] > 0) and (state[12 + 3 * len(self.mining_powers)] != -1) and (state[12 + 2 * len(self.mining_powers) + state[12 + 3 * len(self.mining_powers)]] == 1)):
                next_state[11 + len(self.mining_powers)] = action[3]
            else:
                next_state[11 + len(self.mining_powers)] = -1

        elif (stage == 2):

            for i in range(len(self.mining_powers)):

                if (i == state[1]):
                    
                    #reverse bribery attack
                    if ((state[12 + 3 * len(self.mining_powers)] == i) and (state[6 + len(self.mining_powers)] > 0) and (state[9 + len(self.mining_powers)] == -1)):
                        if (state[12 + 2 * len(self.mining_powers) + i] == 0):
                            next_state[9 + len(self.mining_powers)] = 1
                            state[9 + len(self.mining_powers)] = 1
                            reward[1] += state[12 + len(self.mining_powers) + i]
                            reward[2 + i] += self.v_dep - state[12 + len(self.mining_powers) + i]
                            v_dep_mined_by = 3
                            if (next_state[7 + len(self.mining_powers)] == -1):
                                next_state[7 + len(self.mining_powers)] = 1
                        if ((state[12 + 2 * len(self.mining_powers) + i] == 1) and (state[0] >= self.T_slot)):
                            next_state[9 + len(self.mining_powers)] = 1
                            state[9 + len(self.mining_powers)] = 1
                            reward[1] += state[12 + len(self.mining_powers) + i]
                            reward[2 + i] += self.v_dep - state[12 + len(self.mining_powers) + i]
                            v_dep_mined_by = 3
                            if (next_state[7 + len(self.mining_powers)] == -1):
                                next_state[7 + len(self.mining_powers)] = 1
                            if ((state[10 + len(self.mining_powers)] == -1)):
                                next_state[10 + len(self.mining_powers)] = 1
                                state[10 + len(self.mining_powers)] = 1
                                reward[2 + i] += self.v_col
                                v_col_mined_by = 3

                    if (action[i][0] == 0):
                        reward[2 + i] += self.unrelated_bid

                    elif (state[9 + len(self.mining_powers)] == -1):
                        if ((action[i][0] == 1) and (state[6 + len(self.mining_powers)] > 0) and ((action[i][2] == 0) or (state[11 + len(self.mining_powers)] == -1))):
                            next_state[9 + len(self.mining_powers)] = 1
                            reward[2 + i] += state[6 + len(self.mining_powers)]
                            reward[0] += self.v_dep - state[6 + len(self.mining_powers)]
                            v_dep_mined_by = 0
                        elif ((action[i][0] == 2) and (state[7 + len(self.mining_powers)] > 0)):
                            next_state[9 + len(self.mining_powers)] = 1
                            reward[2 + i] += state[7 + len(self.mining_powers)]
                            reward[1] += self.v_dep - state[7 + len(self.mining_powers)]
                            v_dep_mined_by = 1
                        elif ((action[i][0] == 3) and (state[6 + len(self.mining_powers)] > 0) and (state[7 + len(self.mining_powers)] > 0)):
                            next_state[9 + len(self.mining_powers)] = 1
                            reward[2 + i] += self.v_dep
                            v_dep_mined_by = 2

                    if (action[i][1] == 0):
                        reward[2 + i] += self.unrelated_bid

                    elif ((state[10 + len(self.mining_powers)] == -1) and (state[0] >= self.T_slot)):
                        if ((action[i][1] == 1) and (state[8 + len(self.mining_powers)] > 0)):
                            next_state[10 + len(self.mining_powers)] = 1
                            reward[2 + i] += state[8 + len(self.mining_powers)]
                            reward[1] += self.v_col - state[8 + len(self.mining_powers)]
                            v_col_mined_by = 1
                        elif ((action[i][1] == 2) and (state[6 + len(self.mining_powers)] > 0) and (state[7 + len(self.mining_powers)] > 0)):
                            next_state[10 + len(self.mining_powers)] = 1
                            reward[2 + i] += self.v_col
                            v_col_mined_by = 2

                    if ((action[i][2] == 1) and (state[11 + len(self.mining_powers)] != -1)):
                        reward[2 + i] += state[11 + len(self.mining_powers)]
                        reward[1] -= state[11 + len(self.mining_powers)]

                    if (state[12 + 3 * len(self.mining_powers)] == -1):
                        if ((action[i][3] != 0) and (action[i][4] > 0)):
                            next_state[12 + len(self.mining_powers) + i] = action[i][4]
                            next_state[12 + 2 * len(self.mining_powers) + i] = action[i][3] - 1

                else:

                    if (state[12 + 3 * len(self.mining_powers)] == -1):
                        if ((action[i][3] != 0) and (action[i][4] > 0)):
                            next_state[12 + len(self.mining_powers) + i] = action[i][4]
                            next_state[12 + 2 * len(self.mining_powers) + i] = action[i][3] - 1

            next_state[0] += 1
            next_state[1] = np.random.choice(len(self.mining_powers), p = self.mining_powers)

            if (next_state[0] == self.T_slot + 6):
                reset_flag = 1

        return next_state, reset_flag, reward, v_dep_mined_by, v_col_mined_by