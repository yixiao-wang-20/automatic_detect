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

# description of the game.
#B1: HyDBA
#B2: honest
#B_collusion_1: SDRBA

# phase chosen params.
to_test = 1 # whether to test the agent

# training params.
num_episodes_B1 = 50000 #How many episodes of game environment to train network with.
num_episodes_B2 = 5000 #How many episodes of game environment to train network with.
num_episodes_B_collusion_1 = 5000 #How many episodes of game environment to train network with.
record_path = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\result\\training\\option\\B_option\\"
path_every_episode = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\checkpoint\\option\\B_option\\new\\"
best_model_path = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\checkpoint\\option\\B_option\\best\\"

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
B = agent(h_size_B, input_size_B1_1, input_size_B1_2, input_size_B2, input_size_B_collusion_1, batch_size, LR)
action_agent_B1 = DDPG_net(h_size_B1, action_space_n_continuous_B1, state_space_n_B1, max_action_B1, useless, useless, useless)
action_agent_B1.actor.load_state_dict(torch.load('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\z_back\\checkpoint\\B_1_[500,10000,30000,400]_4_A_can_bid\\new\\990\\actor.params'))
action_agent_B2 = DDPG_net(h_size_B2, action_space_n_continuous_B2, state_space_n_B2, max_action_B2, useless, useless, useless)
action_agent_B2.actor.load_state_dict(torch.load('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\z_back\\checkpoint\\B_2_not_bid_A_can_bid=0.9\\new\\990\\actor.params'))
env_B1 = bribe_env_B1(number_miners, T, reverse_bribe_high_B1, reverse_bribe_low_B1, A_bid_high, A_bid_low)
env_B2 = bribe_env_B2(T, v_dep, v_col, A_bid_high, A_bid_low)
env_B_collusion_1 = bribe_env_B_collusion_1(number_miners, T, reverse_bribe_high_B_collusion_1, reverse_bribe_low_B_collusion_1, A_bid_high, A_bid_low, v_dep, v_col)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# train.
print('training B1')
B_min_loss = 10000000000
print('record as training goes', file=open(record_path + 'record_B1.txt', 'w'))
reward_statistics = {}
for i in range(20):
    reward_statistics[i] = [0, 0]

for episodes in range(num_episodes_B1):

    s_before = env_B1.reset()
    d = 0
    r_accumulated_B1 = 0

    while d == 0: 

        a = action_agent_B1.select_action(s_before, 0)[0][0]
        s_after, d, r = env_B1.step(s_before, a)
        s_before = s_after
        r_accumulated_B1 += r

    if r_accumulated_B1 > 0:
        B_experience_1 = [r_accumulated_B1, env_B1.mining_powers[env_B1.miner_collude_with], s_after[1]]
        B.myBuffer_B1_1.add(np.reshape(np.array(B_experience_1), [1, input_size_B1_1 + 1]))
        reward_statistics[int(env_B1.mining_powers[env_B1.miner_collude_with] * 20)][0] += 1

    else:
        reward_statistics[int(env_B1.mining_powers[env_B1.miner_collude_with] * 20)][1] += 1

    if episodes > 64:
        loss = B.update('B1_1')
        print([episodes, loss, env_B1.mining_powers[env_B1.miner_collude_with], r_accumulated_B1], file=open(record_path + 'record_B1.txt', 'a+'))
        print('', file=open(record_path + 'record_B1.txt', 'a+'))

    if episodes % 100 == 0 and episodes != 0:
        print("episode: " + str(episodes) + " predict_loss: " + str(loss))
        path_every = path_every_episode + 'B1\\' + str(episodes) + "\\"
        if not os.path.exists(path_every):
            os.makedirs(path_every)
        B.save(path_every, 'B1_1')

    if episodes > 64 and loss < B_min_loss:
        B_min_loss = loss
        B.save(best_model_path, 'B1_1')

for i in range(20):
    if reward_statistics[i][0] > reward_statistics[i][1]:
        reward_statistics[i] = 1
    else:
        reward_statistics[i] = 0

with open(best_model_path + 'statistics.json', "w", encoding = "utf-8") as json_file:
    json.dump(reward_statistics, json_file)

'''
print('training B2')
B_min_loss = 10000000000
print('record as training goes', file=open(record_path + 'record_B2.txt', 'w'))

for episodes in range(num_episodes_B2):

    s_before = env_B2.reset()
    d = 0
    r_accumulated_B2 = 0

    while d == 0: 

        a = action_agent_B2.select_action(s_before, 0)[0][0]
        s_after, d, r = env_B2.step(s_before, a)
        s_before = s_after
        r_accumulated_B2 += r

    r_accumulated_B2 += v_col
    B_experience = [r_accumulated_B2, v_dep, v_col]
    B.myBuffer_B2.add(np.reshape(np.array(B_experience), [1, input_size_B2 + 1]))

    if episodes > 64:
        loss = B.update('B2')
        print([episodes, loss, r_accumulated_B2], file=open(record_path + 'record_B2.txt', 'a+'))
        print('', file=open(record_path + 'record_B2.txt', 'a+'))

    if episodes % 100 == 0 and episodes != 0:
        print("episode: " + str(episodes) + " predict_loss: " + str(loss))
        path_every = path_every_episode + 'B2\\' + str(episodes) + "\\"
        if not os.path.exists(path_every):
            os.makedirs(path_every)
        B.save(path_every, 'B2')

    if episodes > 64 and loss < B_min_loss:
        B_min_loss = loss
        B.save(best_model_path, 'B2')


print('training B_collusion_1')
B_min_loss = 10000000000
print('record as training goes', file=open(record_path + 'record_B_collusion_1.txt', 'w'))

for episodes in range(num_episodes_B_collusion_1):

    s_before = env_B_collusion_1.reset()
    d = 0
    r_accumulated_B_collusion_1 = 0

    while d == 0: 

        s_after, d, r = env_B_collusion_1.step(s_before, useless)
        s_before = s_after
        r_accumulated_B_collusion_1 += r

    B_experience = [r_accumulated_B_collusion_1, env_B_collusion_1.mining_powers[env_B_collusion_1.miner_collude_with], s_after[1]]
    B.myBuffer_B_collusion_1.add(np.reshape(np.array(B_experience), [1, input_size_B_collusion_1 + 1]))

    if episodes > 64:
        loss = B.update('B_collusion_1')
        print([episodes, loss, env_B_collusion_1.mining_powers[env_B_collusion_1.miner_collude_with], r_accumulated_B_collusion_1], file=open(record_path + 'record_B_collusion_1.txt', 'a+'))
        print('', file=open(record_path + 'record_B_collusion_1.txt', 'a+'))

    if episodes % 100 == 0 and episodes != 0:
        print("episode: " + str(episodes) + " predict_loss: " + str(loss))
        path_every = path_every_episode + 'B_collusion_1\\' + str(episodes) + "\\"
        if not os.path.exists(path_every):
            os.makedirs(path_every)
        B.save(path_every, 'B_collusion_1')

    if episodes > 64 and loss < B_min_loss:
        B_min_loss = loss
        B.save(best_model_path, 'B_collusion_1')


# test
if to_test:

    file_reward = open(file_reward_path, "w")
    avg_r = 0
    avg_reverse_bribe = 0

    for i in range(rept):

        s_before = env.reset()
        d = 0
        r_every_episode = 0

        while d == 0:

            a = B.act_epsilon_greedy(s_before)
            s_after, d, r = env.step(s_before, a)
            s_before = s_after
            r_every_episode += r

        avg_r += r_every_episode
        avg_reverse_bribe += s_after[1]
    
    avg_r = avg_r / rept
    avg_reverse_bribe = avg_reverse_bribe / rept
        
    print("final average reward = ", avg_r)
    print("final average reverse bribe = ", avg_reverse_bribe)
    print("final average reward = ", avg_r, file = file_reward)
    print("final average reverse bribe = ", avg_reverse_bribe, file = file_reward)

    file_reward.close()
'''