import os
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from environment import bribe_env
from agent.Agent import agent
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
record_path = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\result\\record.txt"
path_every_episode = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\checkpoint\\new\\"
best_model_path = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\checkpoint\\best\\"

# test params.
rept = 50 # final test repetition time
file_reward_path = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\result\\reward_result.txt"

# plot params.
path_for_plot = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\result\\figure\\"

# agent params for all.
h_size = 128 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
batch_size = 64 #How many experiences to use for each training step.
tau = 0.01 #Rate to update target network toward primary network
y = .99 #Discount factor on the target Q-values
LR = 0.001 # learning rate

# agent params for B.
action_space_n_continuous_B = 3 # the number of continuous action space
state_space_n_B = 12 + 3 * len(mining_powers) # the number of state space
buffer_entry_size_B = 2 + len(action_space_n_discrete_B) + action_space_n_continuous_B + 2 * state_space_n_B #The size of the experience buffer.

# environment params.
state_space_n_env = 13 + 3 * len(mining_powers) # the number of state space
T = 20 # the number of time slots in a trial
v_dep = 30000 # the deposit
v_col = 15000 # the collateral
unrelated_bid = 1 # the unrelated bid
mining_powers = [0.5, 0.3, 0.2] # the mining power of the miners
bribe_miner
bribe_value

#Construct environment and agent
B = agent(h_size, buffer_entry_size_B, action_space_n_continuous_B, state_space_n_B, v_dep/10, batch_size, tau, y, LR)
env = bribe_env(state_space_n_env, mining_powers, T, v_dep, v_col, unrelated_bid, mining_powers, bribe_value, bribe_miner)
analyzer = analyze_agent(len(mining_powers), record_path, path_for_plot)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# train.
e = startE
B_best_reward = 0

for episodes in range(num_episodes):

    if episodes % 2 == 0:
        e_used = e
    else:
        e_used = 1
    #e_used = 1
    s_before = env.reset()
    d = 0

    while d == 0: 

        a = B.act_epsilon_greedy(s_before, e_used)[0]
        s_after, d, r = env.step(s_before, a)
        record_of_reward = analyzer.update_record(s_after, a, r, d)

        B_experience = s_before + [a] + [r] + s_after + [d]
        B.myBuffer.add(np.reshape(np.array(B_experience), [1, buffer_entry_size_B]))

        if e > endE:
            e -= stepDrop

        if episodes > 25:
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
    env.seed(100)
    avg_r = [0] * (2 + len(mining_powers))

    for i in range(rept):

        s_A = env.reset()
        d_M = 0
        r_every_episode = [0] * (2 + len(mining_powers))

        while d_M == 0:

            s_B_mapped = analyzer.state_map(s_B, 1)
            a_B = B.act_epsilon_greedy(s_B_mapped, e)
            s_M, d_B, r_B, _, _ = env.step(s_B, a_B, 1)
            for i in range(0, (2 + len(mining_powers))):
                r_every_episode[i] += r_M[i]

        for i in range(0, (2 + len(mining_powers))):
            avg_r[i] += r_every_episode[i]
    
    for i in range(0, (2 + len(mining_powers))):
        avg_r[i] = avg_r[i] / rept
        
    print("final average reward = ", avg_r)
    print("final average reward = ", avg_r, file = file_reward)

    file_reward.close()