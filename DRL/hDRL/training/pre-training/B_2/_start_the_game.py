import os
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from environment_B2 import bribe_env_B2
from Agent_pre_train import agent
from analyze import analyze_agent

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# phase chosen params.
to_test = 1 # whether to test the agent

# training params.
num_episodes = 1000 #How many episodes of game environment to train network with.
startE = 1 #Starting chance of random action
endE = 0 #Final chance of random action
annealing_steps = 400. #How many steps of training to reduce startE to endE.
stepDrop = (startE - endE)/annealing_steps #Reduction rate of random action
record_path = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\result\\training\\pre-training\\B_2\\record.txt"
path_every_episode = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\checkpoint\\pre-training\\B_2\\new\\"
best_model_path = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\checkpoint\\pre-training\\B_2\\best\\"

# test params.
rept = 50 # final test repetition time
file_reward_path = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.2_attack_on_mad_hRL\\result\\training\\pre-training\\B_2\\reward_result.txt"

# agent params for B.
h_size = 32 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
batch_size = 64 #How many experiences to use for each training step.
tau = 0.01 #Rate to update target network toward primary network
y = .99 #Discount factor on the target Q-values
LR = 0.001 # learning rate
action_space_n_continuous_B = 1 # the number of continuous action space
state_space_n_B = 3 # the number of state space
buffer_entry_size_B = 2 + action_space_n_continuous_B + 2 * state_space_n_B #The size of the experience buffer.
B_max_action = 500 # the upper bound of B's action

# environment params.
v_dep_and_v_col = 40000 # the deposit and the collateral of B 35000 5000?
v_dep = 39000 # the deposit of B
v_col = 1000 # the collateral of B
T = 12 # the number of time slots in a trial
A_bid_high = 400 # the upper bound of A's bid
A_bid_low = 10 # the lower bound of A's bid

#Construct environment and agent
B = agent(h_size, buffer_entry_size_B, action_space_n_continuous_B, state_space_n_B, B_max_action, batch_size, tau, y, LR)
env = bribe_env_B2(T, v_dep, v_col, A_bid_high, A_bid_low)
analyzer = analyze_agent(record_path)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# train.
e = startE
B_best_reward = 0

for episodes in range(num_episodes):

    #if episodes % 2 == 0:
    #    e_used = e
    #else:
    #    e_used = 1
    e_used = e
    s_before = env.reset()
    d = 0

    while d == 0: 

        a = B.act_epsilon_greedy(s_before, e_used)
        s_after, d, r = env.step(s_before, a)

        record_of_reward = analyzer.update_record(s_after, a, r, d)

        B_experience = s_before + [a] + [r] + s_after + [d]
        B.myBuffer.add(np.reshape(np.array(B_experience), [1, buffer_entry_size_B]))

        s_before = s_after

        if e > endE:
            e -= stepDrop
            if e < 0:
                e = 0

        if episodes > 30:
            B.update()

    if episodes % 30 == 0:
        print("episode: " + str(episodes) + " B_total_reward: " + str(record_of_reward))
        path_every = path_every_episode + str(episodes) + "\\"
        if not os.path.exists(path_every):
            os.makedirs(path_every)
        B.save(path_every)

    if record_of_reward > B_best_reward:
        B_best_reward = record_of_reward
        B.save(best_model_path)

# test
if to_test:

    file_reward = open(file_reward_path, "w")
    avg_r = 0

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
    
    avg_r = avg_r / rept
        
    print("final average reward = ", avg_r)
    print("final average reward = ", avg_r, file = file_reward)

    file_reward.close()