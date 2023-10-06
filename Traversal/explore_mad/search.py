import itertools
import copy

record_for_sequence = []
record_for_all_sequence = []
reward_value = {'dep':30000, 'col':15000, 'fee':1, 'pro_M':1}
length_of_stage = [20, 3]
honest_reward = 14999
record_path = 'C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment_af\\exploration_mad\\result\\sequence.txt'
record_all_path = 'C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment_af\\exploration_mad\\result\\sequence_all.txt'

def generate_sequence(sub_sequence_p, environment_p, back_for_collusion, agent, count_for_blanke_p, stage_p, reward_p, slot_p):

    sub_sequence_back = copy.deepcopy(sub_sequence_p)
    environment_back = copy.deepcopy(environment_p)
    count_for_blanke_back = count_for_blanke_p
    stage_back = stage_p
    reward_back = reward_p
    slot_back = slot_p
    if agent == 0:
        print(agent)
        action_available = calculate_available_action(0, environment_p, stage_p)
        for num in range(len(action_available) + 1):
            for i in itertools.combinations(action_available, num):
                action = list(i)
                sub_sequence = copy.deepcopy(sub_sequence_back)
                environment = copy.deepcopy(environment_back)
                count_for_blanke = count_for_blanke_back
                stage = stage_back
                reward = reward_back
                slot = slot_back
                entry = ''
                if 0 in action:
                    environment[1] = 1
                    entry += 'B publish pre-b  '
                if 1 in action:
                    environment[2] = 1
                    entry += 'B publish tx-col  '
                if entry != '':
                    sub_sequence.append(entry)
                if action != []:
                    count_for_blanke = 0
                else:
                    count_for_blanke += 1
                if count_for_blanke == 3:
                    count_for_blanke = 0
                    max_reward = calculate_possible_max_reward(calculate_available_action(2, environment, stage), environment, stage)
                    reward -= (max_reward + reward_value['pro_M']) * (length_of_stage[stage] - slot)
                    stage += 1
                    if stage == 2:
                        sub_sequence = cut(sub_sequence)
                        record_for_all_sequence.append([sub_sequence, reward])
                        if reward > honest_reward and [sub_sequence, reward] not in record_for_sequence:
                            record_for_sequence.append([sub_sequence, reward])
                    else:
                        sub_sequence.append('after T slots')
                        generate_sequence(sub_sequence, environment, [0, 0, 0, 0, 0], 0, count_for_blanke, stage, reward, slot)
                else:
                    generate_sequence(sub_sequence, environment, [0, 0, 0, 0, 0], 1, count_for_blanke, stage, reward, slot)

    elif agent == 1:
        print(agent)
        action_available = calculate_available_action(1, environment_p, stage_p)
        for num in range(len(action_available) + 1):
            for i in itertools.combinations(action_available, num):
                action = list(i)
                sub_sequence = copy.deepcopy(sub_sequence_back)
                environment = copy.deepcopy(environment_back)
                count_for_blanke = count_for_blanke_back
                stage = stage_back
                reward = reward_back
                slot = slot_back
                if action_check_consistent_with_policy(action, 1, environment, stage):
                    entry = ''
                    if 0 in action:
                        environment[0] = 1
                        entry += 'A publish pre-a  '
                    if entry != '':
                        sub_sequence.append(entry)
                    if action != []:
                        count_for_blanke = 0
                    else:
                        count_for_blanke += 1
                    if count_for_blanke == 3:
                        count_for_blanke = 0
                        max_reward = calculate_possible_max_reward(calculate_available_action(2, environment, stage), environment, stage)
                        reward -= (max_reward + reward_value['pro_M']) * (length_of_stage[stage] - slot)
                        stage += 1
                        if stage == 2:
                            sub_sequence = cut(sub_sequence)
                            record_for_all_sequence.append([sub_sequence, reward])
                            if reward > honest_reward and [sub_sequence, reward] not in record_for_sequence:
                                record_for_sequence.append([sub_sequence, reward])
                        else:
                            sub_sequence.append('after T slots')
                            generate_sequence(sub_sequence, environment, [0, 0, 0, 0, 0], 0, count_for_blanke, stage, reward, slot)
                    else:
                        generate_sequence(sub_sequence, environment, [0, 0, 0, 0, 0], 2, count_for_blanke, stage, reward, slot)

    elif agent == 2:
        print(agent)
        back = environment_p
        action_available = calculate_available_action(0, environment_p, stage_p)
        for num in range(len(action_available) + 1):
            for i in itertools.combinations(action_available, num):
                action = list(i)
                sub_sequence = copy.deepcopy(sub_sequence_back)
                environment = copy.deepcopy(environment_back)
                count_for_blanke = count_for_blanke_back
                stage = stage_back
                reward = reward_back
                slot = slot_back
                entry = ''
                if 0 in action:
                    environment[1] = 1
                    entry += 'B publish pre-b  '
                if 1 in action:
                    environment[2] = 1
                    entry += 'B publish tx-col  '
                if entry != '':
                    sub_sequence.append(entry)
                if action != []:
                    count_for_blanke = 0
                else:
                    count_for_blanke += 1
                if count_for_blanke == 3:
                    count_for_blanke = 0
                    max_reward = calculate_possible_max_reward(calculate_available_action(2, environment, stage), environment, stage)
                    reward -= (max_reward + reward_value['pro_M']) * (length_of_stage[stage] - slot)
                    stage += 1
                    if stage == 2:
                        sub_sequence = cut(sub_sequence)
                        record_for_all_sequence.append([sub_sequence, reward])
                        if reward > honest_reward and [sub_sequence, reward] not in record_for_sequence:
                            record_for_sequence.append([sub_sequence, reward])
                    else:
                        sub_sequence.append('after T slots')
                        generate_sequence(sub_sequence, environment, back, 0, count_for_blanke, stage, reward, slot)
                else:
                    generate_sequence(sub_sequence, environment, back, 3, count_for_blanke, stage, reward, slot)

    else:
        print(agent)
        action_available_before = calculate_available_action(2, back_for_collusion, stage_p)
        action_available = calculate_available_action(2, environment_p, stage_p)
        max_reward = calculate_possible_max_reward(action_available_before, back_for_collusion, stage_p)
        for num in range(len(action_available) + 1):
            for i in itertools.combinations(action_available, num):
                action = list(i)
                sub_sequence = copy.deepcopy(sub_sequence_back)
                environment = copy.deepcopy(environment_back)
                count_for_blanke = count_for_blanke_back
                stage = stage_back
                reward = reward_back
                slot = slot_back
                if action_ckeck_legal(action, 2, environment, stage):
                    reward_M = 0
                    entry = ''
                    if 0 in action:
                        environment[3] = 1
                        reward_M += reward_value['fee']
                        entry += 'M include tx-a-dep  '
                    if 1 in action:
                        environment[3] = 1
                        reward += reward_value['dep']
                        reward -= reward_value['fee']
                        reward_M += reward_value['fee']
                        entry += 'M include tx-b-dep  '
                    if 2 in action:
                        environment[4] = 1
                        reward += reward_value['col']
                        reward -= reward_value['fee']
                        reward_M += reward_value['fee']
                        entry += 'M include tx-b-col  '
                    if 3 in action:
                        environment[3] = 1
                        reward_M += reward_value['dep']
                        entry += 'M redeem v-dep  '
                    if 4 in action:
                        environment[4] = 1
                        reward_M += reward_value['col']
                        entry += 'M redeem v-col  '
                    if entry != '':
                        flag_for_collusion = 0
                        for sub_action in action:
                            if sub_action in action_available and sub_action not in action_available_before:
                                flag_for_collusion = 1
                        if flag_for_collusion == 1:
                            sub_sequence[len(sub_sequence) - 1] += entry
                        else:
                            sub_sequence.append(entry)
                    if action != []:
                        count_for_blanke = 0
                    else:
                        count_for_blanke += 1
                    if count_for_blanke == 3:
                        max_reward = calculate_possible_max_reward(calculate_available_action(2, environment, stage), environment, stage)
                        reward -= (max_reward + reward_value['pro_M']) * (length_of_stage[stage] - slot)
                        stage += 1
                        count_for_blanke = 0
                        slot = 0
                        if stage == 2:
                            sub_sequence = cut(sub_sequence)
                            record_for_all_sequence.append([sub_sequence, reward])
                            if reward > honest_reward and [sub_sequence, reward] not in record_for_sequence:
                                record_for_sequence.append([sub_sequence, reward])
                        else:
                            sub_sequence.append('after T slots')
                            generate_sequence(sub_sequence, environment, [0, 0, 0, 0, 0], 0, count_for_blanke, stage, reward, slot)
                    else:
                        reward += (reward_M - max_reward - reward_value['pro_M'])
                        slot += 1
                        generate_sequence(sub_sequence, environment, [0, 0, 0, 0, 0], 0, count_for_blanke, stage, reward, slot)

def calculate_available_action(agent, environment, stage):
    action_available = []
    if agent == 0:
        if environment[1] == 0:
            action_available.append(0)
        if environment[2] == 0:
            action_available.append(1)
    elif agent == 1:
        if environment[0] == 0:
            action_available.append(0)
    else:
        if environment[3] == 0 and environment[0] == 1:
            action_available.append(0)
        if environment[3] == 0 and environment[1] == 1 and stage == 1:
            action_available.append(1)
        if environment[4] == 0 and environment[2] == 1 and stage == 1:
            action_available.append(2)
        if environment[3] == 0 and environment[0] == 1 and environment[1] == 1:
            action_available.append(3)
        if environment[4] == 0 and environment[0] == 1 and environment[1] == 1 and stage == 1:
            action_available.append(4)
    return action_available

def action_ckeck_legal(action, agent, environment, stage):
    count_du_1 = 0
    count_du_2 = 0
    if 0 in action:
        count_du_1 += 1
    if 1 in action:
        count_du_1 += 1
    if 3 in action:
        count_du_1 += 1
    if 2 in action:
        count_du_2 += 1
    if 4 in action:
        count_du_2 += 1
    if count_du_1 > 1 or count_du_2 > 1:
        flag = 0
    else:
        flag = 1
    return flag

def action_check_consistent_with_policy(action, agent, environment, stage):
    if stage == 0 and environment[0] == 0 and 0 not in action:
        flag = 0
    else:
        flag = 1
    return flag

def calculate_possible_max_reward(action_available, environment, stage):
    max_reward = 0
    for num in range(len(action_available) + 1):
        for i in itertools.combinations(action_available, num):
            action = list(i)
            if action_ckeck_legal(action, 2, environment, stage):
                reward = 0
                if 0 in action:
                    reward += reward_value['fee']
                if 1 in action:
                    reward += reward_value['fee']
                if 2 in action:
                    reward += reward_value['fee']
                if 3 in action:
                    reward += reward_value['dep']
                if 4 in action:
                    reward += reward_value['col']
                if reward > max_reward:
                    max_reward = reward
    return max_reward

def cut(sub_sequence):
    count = 0
    for i in range(len(sub_sequence)):
        if 'M' in sub_sequence[i]:
            count = i
    return sub_sequence[0:count + 1]

if __name__ == '__main__':
    generate_sequence([], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], 0, 0, 0, 0, 0)
    record_file = open(record_path, "w")
    for entry in record_for_sequence:
        print(entry, file = record_file)
    record_file.close()

    record_all_file = open(record_all_path, "w")
    for entry in record_for_all_sequence:
        print(entry, file = record_all_file)
    record_all_file.close()
