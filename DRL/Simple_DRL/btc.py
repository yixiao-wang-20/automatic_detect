import os
import json
import numpy as np
import random
from environment import bribe_env
from Miner import agent_Miner
from B import agent_B
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# plot params.
to_plot = 1 # whether to plot the result
to_test = 1 # whether to test the agent

# training params.
num_episodes = 500 #How many episodes of game environment to train network with.
startE = 1 #Starting chance of random action
endE = 0 #Final chance of random action
annealing_steps = 200. #How many steps of training to reduce startE to endE.

# agent params.
h_size = 64 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
buffer_entry_size = 11 #The size of the experience buffer. # 4 state + 1 action + 1 reward + 4 next state + 1 done
action_space_n_B = 4 #The number of actions available to the agent
action_space_n_Miner = 3 #The number of actions available to the agent
batch_size = 64 #How many experiences to use for each training step.
tau = 0.001 #Rate to update target network toward primary network
y = .99 #Discount factor on the target Q-values

# environment params.
T = 5 # the number of time slots in a trial
v_dep = 25 # the deposit
unrelated_bid = 1 # the unrelated bid
A_bid = 5 # the bid of A
B_bid = 15 # the bid of B
punishment = -10 # the punishment
decay = 1 # the decay

# test params.
rept = 50 # final test repetition time

env = bribe_env(T, v_dep, unrelated_bid, A_bid, B_bid, punishment, decay)

Miner = agent_Miner(h_size, buffer_entry_size, action_space_n_Miner, batch_size, tau, y)
B = agent_B(h_size, buffer_entry_size, action_space_n_B, batch_size, tau, y)

#Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE)/annealing_steps


# train params.
episodes = 0
B_best = 0
Miner_best = 0

# plot params.
path_for_plot = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\bribe_pytorch\\result\\figure\\"
episode_record = []
B_bid_record = []
Miner_decision_record = []

# train.
file_action = open("C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\bribe_pytorch\\result\\action_result.txt", "w")

for i in range(num_episodes):

    s = env.reset()

    d = 0

    B_total_reward = 0
    Miner_total_reward = 0

    s_M_back = [-1, -1, -1, -1]
    a_M_back = -1
    r_M_back = -1
    d_M_back = -1

    B_actions_record = []
    B_actions_record.append(i)
    B_actions_record.append('B')
    Miner_actions_record = ['Miner']

    while d == 0: 
        
        a_B = B.act_epsilon_greedy(s, e)

        s1, r_B1, r_M1, d1, a1 = env.step(s, a_B, 1)

        a_M = Miner.act_epsilon_greedy(s1, e)

        s2, r_B2, r_M2, d2, a2 = env.step(s1, a_M, 2)

        B_experience = s + [a1] + [r_B2] + s2 + [d2]
        B.myBuffer.add(np.reshape(np.array(B_experience),[1,buffer_entry_size])) #Save the experience to our episode buffer.

        if (s_M_back[0] != -1):
            Miner_experience = s_M_back + [a_M_back] + [r_M_back] + s1 + [d_M_back]
            Miner.myBuffer.add(np.reshape(np.array(Miner_experience),[1,buffer_entry_size]))
        if (d2 == 1):
            Miner_experience = s1 + [a2] + [r_M2] + [-1, -1, -1, -1] + [d2]
            Miner.myBuffer.add(np.reshape(np.array(Miner_experience),[1,buffer_entry_size]))

        s_M_back = s1
        a_M_back = a2
        r_M_back = r_M2
        d_M_back = d2

        s = s2
        d = d2

        B_actions_record.append(a1)
        Miner_actions_record.append(a2)

        B_total_reward += r_B2
        Miner_total_reward += r_M2

        if e > endE:
            e -= stepDrop

        if episodes > 67:
            B.train()
            Miner.train()

        if (d2 == 1):  
            Miner_actions_record.append("A'bid:")
            Miner_actions_record.append(s2[1])
            Miner_actions_record.append("A redeems:")
            Miner_actions_record.append(s2[3] == 1)
            Miner_actions_record.append("B'bid:")
            Miner_actions_record.append(s2[2])
            Miner_actions_record.append("B redeems:")
            Miner_actions_record.append(a2 == 2)
            if to_plot:
                episode_record.append(i)
                B_bid_record.append(s2[2])
                if a2 == 2:
                    Miner_decision_record.append(2)
                elif s2[3] == 1:
                    Miner_decision_record.append(1)
                else:
                    Miner_decision_record.append(0)

    B_actions_record.extend(Miner_actions_record)
    print(B_actions_record, file=file_action)
    print('  ', file=file_action)

    episodes += 1

    if episodes % 30 == 0:
        print("episode: " + str(episodes) + " B_total_reward: " + str(B_total_reward) + " Miner_total_reward: " + str(Miner_total_reward) + "  A_has_published: " + str(s2[1]) + " tx_a_has_been_included: " + str(s2[3]))
        path_every = "C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\bribe_pytorch\\checkpoint\\new\\" + str(episodes) + "\\"
        if not os.path.exists(path_every):
            os.makedirs(path_every)
        torch.save(B.mainQN.state_dict(), path_every + 'model_B.params')
        torch.save(Miner.mainQN.state_dict(), path_every + 'model_M.params')

    if B_total_reward > B_best:
        B_best = B_total_reward
        torch.save(B.mainQN.state_dict(), 'C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\bribe_pytorch\\checkpoint\\best\\best_B.params')
    if Miner_total_reward > Miner_best:
        Miner_best = Miner_total_reward
        torch.save(Miner.mainQN.state_dict(), 'C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\bribe_pytorch\\checkpoint\\best\\best_M.params')

file_action.close()

# test
if to_test:

    file_reward = open("C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment\\bribe_pytorch\\result\\reward_result.txt", "w")
    env.seed(100)
    avg_B = 0
    avg_M = 0

    for i in range(rept):

        s = env.reset()
        d = 0
        r_B = 0
        r_M = 0

        while d == 0:
            a_B = B.act_epsilon_greedy(s, e)
            s1, r_B1, r_M1, d1, a1 = env.step(s, a_B, 1)
            a_M = Miner.act_epsilon_greedy(s1, e)
            s2, r_B2, r_M2, d2, a2 = env.step(s1, a_M, 2)
            s = s2
            d = d2
            r_B += r_B2
            r_M += r_M2
        avg_B += r_B
        avg_M += r_M

    avg_B = avg_B / rept
    avg_M = avg_M / rept

    print("final average reward of B = ", avg_B)
    print("final average reward of Miner = ", avg_M)
    print("final average reward of B = ", avg_B, file = file_reward)
    print("final average reward of Miner = ", avg_M, file = file_reward)

    file_reward.close()

# plot
if to_plot:

    data_for_plot = {'episode': episode_record, 'B_bid': B_bid_record, 'Miner_decision': Miner_decision_record}
    with open(path_for_plot + "data.json", 'w') as f:
        json.dump(data_for_plot, f)
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(episode_record, B_bid_record, color='red', linestyle='--', label='bid')
    ax1.set_xlabel('episode')
    ax1.set_ylabel('bid')
    ax1.set_title('B_bid')
    ax1.legend()
    ax2.plot(episode_record, Miner_decision_record, color='blue', linestyle='-', label='decision')
    ax2.set_xlabel('episode')
    ax2.set_ylabel('decision')
    ax2.set_title('Miner_decision')
    ax2.legend()
    plt.tight_layout()
    fig.savefig(path_for_plot + 'result.png')