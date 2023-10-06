import os
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json
import sys

from environment_B1 import bribe_env_B1
from environment_B2 import bribe_env_B2
from environment_B_collusion_1 import bribe_env_B_collusion_1
sys.path.append('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\_code\\agent\\skill')
from DDPG import DDPG_net
from Agent_option_train import agent

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# testing params.
test_episode = 100000
best_model_path = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\checkpoint\\option\\B_option\\best\\"
record_path = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\result\\test\\1.compare_DRL_with_policy\\record.txt"
final_reward_path = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\result\\test\\1.compare_DRL_with_policy\\final_reward.txt"

# agent params for B.
h_size_B = 32 #The size of the hidden layer.
batch_size = 64 #How many experiences to use for each training step.
LR = 0.001 # learning rate
input_size_B1_1 = 2 # the number of state space
input_size_B1_2 = 1 # the number of state space
input_size_B2 = 2 # the number of state space
input_size_B_collusion_1 = 2 # the number of state space

# agent params for B1.
h_size_B1 = 32 #The size of the hidden layer.
action_space_n_continuous_B1 = 1 # the number of continuous action space
state_space_n_B1 = 3 # the number of state space
max_action_B1 = 3000 # the upper bound of B's action

# agent params for B2.
h_size_B2 = 32 #The size of the hidden layer.
action_space_n_continuous_B2 = 1 # the number of continuous action space
state_space_n_B2 = 3 # the number of state space
max_action_B2 = 500 # the upper bound of B's action

# params for all environments.
v_dep_and_v_col = 40000 # the deposit of B
number_miners = 3 # the number of miners
T = 7 # the number of time slots in a trial
A_bid_high = 400 # the upper bound of A's bid
A_bid_low = 10 # the lower bound of A's bid

# environment_B1 params.
reverse_bribe_high_B1 = 30000 # the upper bound of reverse bribe
reverse_bribe_low_B1 = 5000 # the lower bound of reverse bribe

# environment_B2 params.
v_dep = 39000 # the deposit of B
v_col = 1000 # the collateral of B

# environment_B_collusion_1 params.
reverse_bribe_high_B_collusion_1 = 4000 # the upper bound of reverse bribe
reverse_bribe_low_B_collusion_1 = 500 # the lower bound of reverse bribe

# for padding
useless = 0

#Construct environment and agent
with open(best_model_path + 'statistics.json', "r", encoding = "utf-8") as json_file:
    reward_statistics = json.load(json_file)
B = agent(h_size_B, input_size_B1_1, input_size_B1_2, input_size_B2, input_size_B_collusion_1, batch_size, LR)
B.load(best_model_path, 'B1_1')
B.load(best_model_path, 'B2')
B.load(best_model_path, 'B_collusion_1')
action_agent_B1 = DDPG_net(h_size_B1, action_space_n_continuous_B1, state_space_n_B1, max_action_B1, useless, useless, useless)
action_agent_B1.actor.load_state_dict(torch.load('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\z_back\\checkpoint\\B_1_[500,10000,30000,400]_4_A_can_bid\\new\\990\\actor.params'))
action_agent_B2 = DDPG_net(h_size_B2, action_space_n_continuous_B2, state_space_n_B2, max_action_B2, useless, useless, useless)
action_agent_B2.actor.load_state_dict(torch.load('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\z_back\\checkpoint\\B_2_not_bid_A_can_bid=0.9\\new\\990\\actor.params'))
env_B1 = bribe_env_B1(number_miners, T, reverse_bribe_high_B1, reverse_bribe_low_B1, A_bid_high, A_bid_low)
env_B2 = bribe_env_B2(T, v_dep, v_col, A_bid_high, A_bid_low)
env_B_collusion_1 = bribe_env_B_collusion_1(number_miners, T, reverse_bribe_high_B_collusion_1, reverse_bribe_low_B_collusion_1, A_bid_high, A_bid_low, v_dep, v_col)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# test.
print('testing')
print('reward record', file=open(record_path, 'w'))
print(' ', file=open(record_path, 'w'))
print('reward record', file=open(final_reward_path, 'a+'))
print(' ', file=open(final_reward_path, 'a+'))
greedy_total_reward = 0
DRL_total_reward = 0

for i in range(test_episode):

    mining_power = np.random.rand(number_miners)
    mining_powers = mining_power / np.sum(mining_power)
    collusion_methods = []
    collusion_values = np.zeros(number_miners)

    for j in range(number_miners):
        collusion_methods.append(np.random.choice(3))

    for j in range(number_miners):

        if collusion_methods[j] == 1:
            collusion_values[j] = (np.random.randint(reverse_bribe_low_B_collusion_1, reverse_bribe_high_B_collusion_1))
        elif collusion_methods[j] == 2:
            collusion_values[j] = (np.random.randint(reverse_bribe_low_B1, reverse_bribe_high_B1))
        else:
            collusion_values[j] = 0

    #greedy-policy

    greedy_chosen = np.argmax(collusion_values)

    if (collusion_methods[greedy_chosen] == 0 or collusion_values[greedy_chosen] < v_col):

        greedy_reward = v_col

    elif (collusion_methods[greedy_chosen] == 1):

        s_before = env_B_collusion_1.reset()
        env_B_collusion_1.mining_powers = mining_powers
        env_B_collusion_1.miner_collude_with = greedy_chosen
        s_before[1] = collusion_values[greedy_chosen]
        d = 0
        greedy_reward = 0

        while d == 0: 

            s_after, d, r = env_B_collusion_1.step(s_before, useless)
            s_before = s_after
            greedy_reward += r

    else:

        s_before = env_B1.reset()
        env_B1.mining_powers = mining_powers
        env_B1.miner_collude_with = greedy_chosen
        s_before[1] = collusion_values[greedy_chosen]
        d = 0
        greedy_reward = 0

        while d == 0: 

            a = np.random.randint(0, max_action_B1)
            s_after, d, r = env_B1.step(s_before, a)
            s_before = s_after
            greedy_reward += r

        #for_fixing
        if greedy_reward < 0:
            greedy_reward = 0

    #DRL-policy

    expected_reward = np.zeros(number_miners)

    for j in range(number_miners):

        if collusion_methods[j] == 1:
            expected_reward[j] = B.predict([mining_powers[j], collusion_values[j]], 'B_collusion_1')
        elif collusion_methods[j] == 2:
            if (reward_statistics[str(int(mining_powers[j] * 20))] == 0):
                expected_reward[j] = 0
            else:
                expected_reward[j] = B.predict([mining_powers[j], collusion_values[j]], 'B1_1')
        else:
            expected_reward[j] = 0
    
    DRL_chosen = np.argmax(expected_reward)

    if (collusion_methods[DRL_chosen] == 0 or expected_reward[DRL_chosen] < v_col):

        DRL_reward = v_col

    elif (collusion_methods[DRL_chosen] == 1):

        s_before = env_B_collusion_1.reset()
        env_B_collusion_1.mining_powers = mining_powers
        env_B_collusion_1.miner_collude_with = DRL_chosen
        s_before[1] = collusion_values[DRL_chosen]
        d = 0
        DRL_reward = 0

        while d == 0: 

            s_after, d, r = env_B_collusion_1.step(s_before, useless)
            s_before = s_after
            DRL_reward += r

    else:

        s_before = env_B1.reset()
        env_B1.mining_powers = mining_powers
        env_B1.miner_collude_with = DRL_chosen
        s_before[1] = collusion_values[DRL_chosen]
        d = 0
        DRL_reward = 0

        while d == 0: 

            a = action_agent_B1.select_action(s_before, 0)[0][0]
            s_after, d, r = env_B1.step(s_before, a)
            s_before = s_after
            DRL_reward += r

        #for_fixing
        if DRL_reward < 0:
            DRL_reward = 0

    greedy_total_reward += greedy_reward
    DRL_total_reward += DRL_reward
    print([i, greedy_reward, DRL_reward], file=open(record_path, 'a+'))
    print('', file=open(record_path, 'a+'))

print([greedy_total_reward / test_episode, DRL_total_reward / test_episode], file=open(final_reward_path, 'a+'))