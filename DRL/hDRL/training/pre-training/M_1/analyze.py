import json
import matplotlib.pyplot as plt

class analyze_agent():

    def __init__(self, miner_number, record_path, path_for_plot):

        self.miner_number = miner_number 
        self.record_path = record_path
        self.path_for_plot = path_for_plot

        self.reward = [0] * (2 + miner_number)
        self.bribe_count = 0
        self.bribe_value = 0
        self.v_dep_mined_by = -1
        self.v_col_mined_by = -1

        self.A_rewards = []
        self.B_rewards = []
        self.method_of_r_attack = []
        self.r_attack_rewards = []

        print('state record as training goes', file=open(record_path, 'w'))

    def state_map(self, s, stage):

        if (stage == 0):

            s_observed = s[0:11 + self.miner_number]

        elif (stage == 1):
                
            s_observed = s[0:11 + self.miner_number]
            s_observed += s[12 + self.miner_number:13 + 3 * self.miner_number]

        elif (stage == 2):
                
            s_observed = []

            for i in range(self.miner_number):
                s_observed.append(s[0:12 + self.miner_number])
                s_observed[i].append(s[12 + self.miner_number + i])
                s_observed[i].append(s[12 + 2 * self.miner_number + i])
                s_observed[i].append((s[12 + 3 * self.miner_number] == i))

        return s_observed

    def update_record(self, s, a, r, done, v_dep_mined_by, v_col_mined_by):

        return_reward = 0

        if (v_dep_mined_by != -1):
            self.v_dep_mined_by = v_dep_mined_by
            
        if (v_col_mined_by != -1):
            self.v_col_mined_by = v_col_mined_by

        for i in range(0, (2 + self.miner_number)):
            self.reward[i] += r[i] 
        
        if ((s[11 + self.miner_number] != -1) and (a[s[1]][2] == 1)):
            self.bribe_count += 1
            self.bribe_value += s[11 + self.miner_number]

        if (done == 1):

            return_reward = self.reward
            record = []
            record.append('v_dep_mined_by')
            record.append(self.v_dep_mined_by)
            record.append('v_col_mined_by')
            record.append(self.v_col_mined_by)
            record.append('who')
            record.append(s[12 + 3 * self.miner_number])
            record.append('method')
            if (s[12 + 3 * self.miner_number] == -1):
                record.append(-1)
            else:
                record.append(s[12 + 2 * self.miner_number + s[12 + 3 * self.miner_number]])
            record.append('bribe_count')
            record.append(self.bribe_count)
            record.append('bribe_value')
            record.append(int(self.bribe_value))
            record.append('fee_a')
            record.append(int(s[6 + self.miner_number]))
            record.append('fee_b_dep')
            record.append(int(s[7 + self.miner_number]))
            record.append('fee_b_col')
            record.append(int(s[8 + self.miner_number]))
            print(record, file=open(self.record_path, 'a+'))
            record = []
            record.append('bid')
            record += [int(iter) for iter in s[12 + self.miner_number:12 + 2 * self.miner_number]]
            record.append('method')
            record += s[12 + 2 * self.miner_number:12 + 3 * self.miner_number]
            record.append('reward')
            record += [int(iter) for iter in self.reward]
            print(record, file=open(self.record_path, 'a+'))
            print('', file=open(self.record_path, 'a+'))

            self.A_rewards.append(int(self.reward[0]))
            self.B_rewards.append(int(self.reward[1]))
            if (s[12 + 3 * self.miner_number] != -1):
                self.method_of_r_attack.append(int(s[12 + 2 * self.miner_number + s[12 + 3 * self.miner_number]]))
                self.r_attack_rewards.append(int(self.reward[2 + s[12 + 3 * self.miner_number]]))
 
            self.reward = [0] * (2 + self.miner_number)
            self.bribe_count = 0
            self.bribe_value = 0
            self.v_dep_mined_by = 0
            self.v_col_mined_by = 0

        return return_reward

    def plot(self):

        data_for_plot = {'A_rewards': self.A_rewards, 'B_rewards': self.B_rewards, 'method_of_r_attack': self.method_of_r_attack, 'r_attack_rewards': self.r_attack_rewards}

        with open(self.path_for_plot + "data.json", 'w') as f:
            json.dump(data_for_plot, f)

        episode_record = list(range(1, len(self.A_rewards) + 1))
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(episode_record, self.A_rewards, color='red', linestyle='--', label='bid')
        ax1.set_xlabel('episode')
        ax1.set_ylabel('reward')
        ax1.set_title('A_reward')
        ax1.legend()
        ax2.plot(episode_record, self.B_rewards, color='blue', linestyle='-', label='decision')
        ax2.set_xlabel('episode')
        ax2.set_ylabel('reward')
        ax2.set_title('B_reward')
        ax2.legend()
        plt.tight_layout()
        fig.savefig(self.path_for_plot + 'A&B_result.png')

        episode_record = list(range(1, len(self.method_of_r_attack) + 1))
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(episode_record, self.method_of_r_attack, color='red', linestyle='--', label='bid')
        ax1.set_xlabel('episode')
        ax1.set_ylabel('method')
        ax1.set_title('r_M_method')
        ax1.legend()
        ax2.plot(episode_record, self.r_attack_rewards, color='blue', linestyle='-', label='decision')
        ax2.set_xlabel('episode')
        ax2.set_ylabel('reward')
        ax2.set_title('r_M_reward')
        ax2.legend()
        plt.tight_layout()
        fig.savefig(self.path_for_plot + 'r_M_result.png')