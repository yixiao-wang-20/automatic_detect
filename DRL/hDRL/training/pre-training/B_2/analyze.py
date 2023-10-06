import json

class analyze_agent():

    def __init__(self, record_path):

        self.record_path = record_path
        self.reward = 0
        self.bid_list = []
        print('state record as training goes', file=open(record_path, 'w'))

    def update_record(self, s, a, r, done):

        return_reward = 0
        self.reward += r
        self.bid_list.append(a)

        if (done == 1):

            return_reward = self.reward
            record = []
            record.append('final_reward_of_B')
            record.append(return_reward)
            record.append('bid_of_A')
            record.append(s[2])
            record.append('bid_of_B')
            record.append(self.bid_list)
            print(record, file=open(self.record_path, 'a+'))
            print('', file=open(self.record_path, 'a+'))
 
            self.reward = 0
            self.bid_list = []

        return return_reward