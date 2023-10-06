import os
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from environment import bribe_env
from agent.Agent import agent
from agent.Agent_A import agent_A
from analyze import analyze_agent

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# phase chosen params.
to_test = 1 # whether to test the agent
to_plot = 1 # whether to plot the result

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
mining_powers = [0.5, 0.3, 0.2] # the mining power of the miners

# agent params for A.
action_space_n_discrete_A = [] # the number of discrete action space
action_space_n_continuous_A = 1 # the number of continuous action space
state_space_n_A = 11 + len(mining_powers) # the number of state space
buffer_entry_size_A = 2 + len(action_space_n_discrete_A) + action_space_n_continuous_A + 2 * state_space_n_A #The size of the experience buffer.

# agent params for B.
action_space_n_discrete_B = [len(mining_powers) + 1] # the number of discrete action space
action_space_n_continuous_B = 3 # the number of continuous action space
state_space_n_B = 12 + 3 * len(mining_powers) # the number of state space
buffer_entry_size_B = 2 + len(action_space_n_discrete_B) + action_space_n_continuous_B + 2 * state_space_n_B #The size of the experience buffer.

# agent params for Miners.
action_space_n_discrete_M = [4, 3, 2, 3] # the number of discrete action space
action_space_n_continuous_M = 1 # the number of continuous action space
state_space_n_M = 15 + len(mining_powers) # the number of state space
buffer_entry_size_M = 2 + len(action_space_n_discrete_M) + action_space_n_continuous_M + 2 * state_space_n_M #The size of the experience buffer.

# environment params.
state_space_n_env = 13 + 3 * len(mining_powers) # the number of state space
T = 20 # the number of time slots in a trial
v_dep = 30000 # the deposit
v_col = 15000 # the collateral
unrelated_bid = 1 # the unrelated bid

#Construct environment and agent
A = agent_A(h_size, buffer_entry_size_A, action_space_n_discrete_A, action_space_n_continuous_A, state_space_n_A, v_dep/10, batch_size, tau, y, LR)
B = agent(h_size, buffer_entry_size_B, action_space_n_discrete_B, action_space_n_continuous_B, state_space_n_B, v_dep/10, batch_size, tau, y, LR)
Miners = []
for i in range(len(mining_powers)):
    Miners.append(agent(h_size, buffer_entry_size_M, action_space_n_discrete_M, action_space_n_continuous_M, state_space_n_M, v_dep/3, batch_size, tau, y, LR))
env = bribe_env(state_space_n_env, mining_powers, T, v_dep, v_col, unrelated_bid)
analyzer = analyze_agent(len(mining_powers), record_path, path_for_plot)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# train.
e = startE
A_best = 0
B_best = 0
Miner_best = [0] * len(mining_powers)
v_dep_3 = 0
v_col_3 = 0
#print(' ', file=open('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\result\\debug.txt', 'w'))

for episodes in range(num_episodes):

    if episodes % 2 == 0:
        e_used = e
    else:
        e_used = 1
    #e_used = 1
    s_A = env.reset()
    d_M = 0

    while d_M == 0: 
        #print(' ', file=open('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\result\\debug.txt', 'a+'))
        s_A_mapped = analyzer.state_map(s_A, 0)
        #print(s_A, file=open('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\result\\debug.txt', 'a+'))
        a_A = A.act_epsilon_greedy(s_A_mapped, e_used)
        #print(a_A, file=open('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\result\\debug.txt', 'a+'))
        s_B, d_A, r_A, _, _ = env.step(s_A, a_A, 0)
        s_B_mapped = analyzer.state_map(s_B, 1)
        #print(s_B, file=open('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\result\\debug.txt', 'a+'))
        a_B = B.act_epsilon_greedy(s_B_mapped, e_used)
        #print(a_B, file=open('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\result\\debug.txt', 'a+'))
        s_M, d_B, r_B, _, _ = env.step(s_B, a_B, 1)
        #print(s_M, file=open('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\result\\debug.txt', 'a+'))
        s_M_mapped = analyzer.state_map(s_M, 2)
        a_M = []
        for i in range(len(mining_powers)):
            a_M.append(Miners[i].act_epsilon_greedy(s_M_mapped[i], e_used))
        #print(a_M, file=open('C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\4.attack_on_mad\\result\\debug.txt', 'a+'))
        s_A, d_M, r_M, v_dep_mined_by, v_col_mined_by = env.step(s_M, a_M, 2)
        record_of_reward = analyzer.update_record(s_A, a_M, r_M, d_M, v_dep_mined_by, v_col_mined_by)

        if ((s_A_mapped[0] != 1)):
            A_experience = s_A_mapped_back + [a_A_back] + [r_M_back[0]] + s_A_mapped + [d_A_back]
            A.myBuffer.add(np.reshape(np.array(A_experience), [1, buffer_entry_size_A]))
            B_experience = s_B_mapped_back + a_B_back + [r_M_back[1]] + s_B_mapped + [d_B_back]
            B.myBuffer.add(np.reshape(np.array(B_experience), [1, buffer_entry_size_B]))
            for i in range(len(mining_powers)):
                Miner_experience = s_M_mapped_back[i] + a_M_back[i] + [r_M_back[2 + i]] + s_M_mapped[i] + [d_M_back]
                Miners[i].myBuffer.add(np.reshape(np.array(Miner_experience), [1, buffer_entry_size_M]))

        if(d_M == 1):
            A_experience = s_A_mapped + [a_A] + [r_M[0]] + s_A_mapped + [d_M]
            A.myBuffer.add(np.reshape(np.array(A_experience), [1, buffer_entry_size_A]))
            B_experience = s_B_mapped + a_B + [r_M[1]] + s_B_mapped + [d_M]
            B.myBuffer.add(np.reshape(np.array(B_experience), [1, buffer_entry_size_B]))
            for i in range(len(mining_powers)):
                Miner_experience = s_M_mapped[i] + a_M[i] + [r_M[2 + i]] + s_M_mapped[i] + [d_M]
                Miners[i].myBuffer.add(np.reshape(np.array(Miner_experience), [1, buffer_entry_size_M]))

        r_M_back = r_M
        s_A_mapped_back, a_A_back, d_A_back = s_A_mapped, a_A, d_A
        s_B_mapped_back, a_B_back, d_B_back = s_B_mapped, a_B, d_B
        s_M_mapped_back, a_M_back, d_M_back = s_M_mapped, a_M, d_M

        if (v_dep_mined_by == 3):
            v_dep_3 += 1
        if (v_col_mined_by == 3):
            v_col_3 += 1

        if e > endE:
            e -= stepDrop

        if episodes > 25:
            A.update()
            B.update()
            for i in range(len(mining_powers)):    
                Miners[i].update()

    if episodes % 30 == 0:
        print("episode: " + str(episodes) + " A_total_reward: " + str(record_of_reward[0]) + " B_total_reward: " + str(record_of_reward[1]) + " Miner_total_reward: ", record_of_reward[2:], " v_dep_3: " + str(v_dep_3) + " v_col_3: " + str(v_col_3))
        path_every = path_every_episode + str(episodes) + "\\"
        if not os.path.exists(path_every):
            os.makedirs(path_every)
        A.save(path_every, "A")
        B.save(path_every, "B")
        for i in range(len(mining_powers)):
            Miners[i].save(path_every, "Miner_" + str(i))

    if record_of_reward[0] > A_best:
        A_best = record_of_reward[0]
        A.save(best_model_path, "A")
    if record_of_reward[1] > B_best:
        B_best = record_of_reward[1]
        B.save(best_model_path, "B")
    for i in range(len(mining_powers)):
        if record_of_reward[2 + i] > Miner_best[i]:
            Miner_best[i] = record_of_reward[2 + i]
            Miners[i].save(best_model_path, "Miner_" + str(i))

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

            s_A_mapped = analyzer.state_map(s_A, 0)
            a_A = A.act_epsilon_greedy(s_A_mapped, e)
            s_B, d_A, r_A, _, _ = env.step(s_A, a_A, 0)
            s_B_mapped = analyzer.state_map(s_B, 1)
            a_B = B.act_epsilon_greedy(s_B_mapped, e)
            s_M, d_B, r_B, _, _ = env.step(s_B, a_B, 1)
            s_M_mapped = analyzer.state_map(s_M, 2)
            a_M = []
            for i in range(len(mining_powers)):
                a_M.append(Miners[i].act_epsilon_greedy(s_M_mapped[i], e))
            s_A, d_M, r_M, v_dep_mined_by, v_col_mined_by = env.step(s_M, a_M, 2)
            for i in range(0, (2 + len(mining_powers))):
                r_every_episode[i] += r_M[i]

        for i in range(0, (2 + len(mining_powers))):
            avg_r[i] += r_every_episode[i]
    
    for i in range(0, (2 + len(mining_powers))):
        avg_r[i] = avg_r[i] / rept
        
    print("final average reward = ", avg_r)
    print("final average reward = ", avg_r, file = file_reward)

    file_reward.close()

# plot
if to_plot:
    analyzer.plot()