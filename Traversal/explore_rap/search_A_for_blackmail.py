import itertools
import copy

record_for_sequence = []
record_for_all_sequence = []
blanke_env = {'pre-a':0, '_P2':0, '_P1.':0, 'pre-a.':0, 'pre-c':0, 'pre-b':0, 'pre-b.':0, '_P2.':0, '_B1':0, '_C1':0, '_A1.':0, '_C1.':0, 'B1':0, 'P1':0, 'P2':0, 'C1':0, 'A1.':0, 'P1.':0, 'P2.':0, 'C1.':0, 'v':0, 'v.':0, 't_P2':0, 'state':0}
reward_value = {'x':10000, 'Ca':20000, 'Cb':20000, 'x.':10000, 'Ca.':20000, 'Cb.':20000, 'fee':10, 'pro_M':10, 'E':8000, 'E.':8000}
#-2:start -1:T0 0:T0. 1:T1 2:T1. 3:end
length_of_stage = {'-2':0, '-1':5, '0':10, '1':20, '2':30, '3':40, 'tau':5, 'tau.':5}
honest_reward = reward_value['Cb'] + reward_value['x.'] + reward_value['Cb.']
honest_reward_A = reward_value['x'] + reward_value['Ca'] + reward_value['Ca.']
record_path = 'C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment_af\\exploration_rap\\result\\A\\sequence.txt'
record_all_path = 'C:\\Users\\Lenovo\\Desktop\\1\\summer_intern\\incentive\\experiment_af\\exploration_rap\\result\\A\\sequence_all.txt'

def generate_sequence(sub_sequence_p, environment_p, back_for_collusion, agent, count_for_blanke_p, stage_p, reward_p, slot_p):

    sub_sequence_back = copy.deepcopy(sub_sequence_p)
    environment_back = copy.deepcopy(environment_p)
    count_for_blanke_back = count_for_blanke_p
    stage_back = stage_p
    reward_back = reward_p
    slot_back = slot_p
    if agent == 0:
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
                entry = ''
                if 'pre-a' in action:
                    environment['pre-a'] = 1
                    entry += 'A publish pre-a  '
                if '_P2' in action:
                    environment['_P2'] = 1
                    entry += 'A publish _P2  '
                if '_P1.' in action:
                    environment['_P1.'] = 1
                    entry += 'A publish _P1.  '
                if 'pre-a.' in action:
                    environment['pre-a.'] = 1
                    entry += 'A publish pre-a.  '
                if '_B1' in action:
                    environment['_B1'] = 1
                    entry += 'A publish _B1  '
                if '_C1' in action:
                    environment['_C1'] = 1
                    entry += 'A publish _C1  '
                if '_A1.' in action:
                    environment['_A1.'] = 1
                    entry += 'A publish _A1.  '
                if '_C1.' in action:
                    environment['_C1.'] = 1
                    entry += 'A publish _C1.  '
                environment = check_state(copy.deepcopy(environment), stage)
                if entry != '':
                    sub_sequence.append(entry)
                if action != []:
                    count_for_blanke = 0
                else:
                    count_for_blanke += 1
                if count_for_blanke == 2:
                    count_for_blanke = 0
                    stage += 1
                    slot = 0
                    if stage == 3:
                        #sub_sequence_all = cut(sub_sequence)
                        #record_file_all = open(record_all_path, "a+")
                        #print([sub_sequence_all, reward], file = record_file_all)
                        #record_file_all.close()
                        #record_for_all_sequence.append([sub_sequence_all, reward])
                        if reward > honest_reward_A and [sub_sequence, reward] not in record_for_sequence:
                            record_file = open(record_path, "a+")
                            print([sub_sequence, reward], file = record_file)
                            record_file.close()
                            record_for_sequence.append([sub_sequence, reward])
                    elif stage == 2:
                        sub_sequence.append("after T1' slots")
                        generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)
                    elif stage == 1:
                        sub_sequence.append('after T1 slots')
                        generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)
                    elif stage == 0:
                        sub_sequence.append("after T0' slots")
                        generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)
                    else:
                        sub_sequence.append('after T0 slots')
                        generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)
                else:
                    generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 1, count_for_blanke, stage, reward, slot)

    elif agent == 1:
        print(agent)
        sub_sequence = copy.deepcopy(sub_sequence_back)
        environment = copy.deepcopy(environment_back)
        count_for_blanke = count_for_blanke_back
        stage = stage_back
        reward = reward_back
        slot = slot_back
        action = generate_action(0, environment, stage, 'rational')
        entry = ''
        if 'pre-c' in action:
            environment['pre-c'] = 1
            entry += 'B publish pre-c  '
        if 'pre-b' in action:
            environment['pre-b'] = 1
            entry += 'B publish pre-b  '
        if 'pre-b.' in action:
            environment['pre-b.'] = 1
            entry += 'B publish pre-b.  '
        if '_P2.' in action:
            environment['_P2.'] = 1
            entry += 'B publish _P2.  '
        if '_B1' in action:
            environment['_B1'] = 1
            entry += 'B publish _B1  '
        if '_C1' in action:
            environment['_C1'] = 1
            entry += 'B publish _C1  '
        if '_A1.' in action:
            environment['_A1.'] = 1
            entry += 'B publish _A1.  '
        if '_C1.' in action:
            environment['_C1.'] = 1
            entry += 'B publish _C1.  '
        environment = check_state(copy.deepcopy(environment), stage)
        if entry != '':
            sub_sequence.append(entry)
        if action != []:
            count_for_blanke = 0
        else:
            count_for_blanke += 1
        if count_for_blanke == 2:
            count_for_blanke = 0
            stage += 1
            slot = 0
            if stage == 3:
                #sub_sequence_all = cut(sub_sequence)
                #record_file_all = open(record_all_path, "a+")
                #print([sub_sequence_all, reward], file = record_file_all)
                #record_file_all.close()
                #record_for_all_sequence.append([sub_sequence_all, reward])
                if reward > honest_reward_A and [sub_sequence, reward] not in record_for_sequence:
                    record_file = open(record_path, "a+")
                    print([sub_sequence, reward], file = record_file)
                    record_file.close()
                    record_for_sequence.append([sub_sequence, reward])
            elif stage == 2:
                sub_sequence.append("after T1' slots")
                generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)
            elif stage == 1:
                sub_sequence.append('after T1 slots')
                generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)
            elif stage == 0:
                sub_sequence.append("after T0' slots")
                generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)
            else:
                sub_sequence.append('after T0 slots')
                generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)
        else:
            generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 2, count_for_blanke, stage, reward, slot)

    else:
        print(agent)
        action_available = calculate_available_action(2, environment_p, stage_p)
        actions = calculate_possible_max_reward(action_available, back_for_collusion, stage_p)
        for action in actions:
            sub_sequence = copy.deepcopy(sub_sequence_back)
            environment = copy.deepcopy(environment_back)
            count_for_blanke = count_for_blanke_back
            stage = stage_back
            reward = reward_back
            slot = slot_back
            if action_ckeck_legal(action, 2, environment, stage):
                reward_M = 0
                reward_A = 0
                reward_B = 0
                entry = ''
                if 'B1' in action:
                    environment['B1'] = 1
                    reward_M += reward_value['fee']
                    entry += 'M include B1  '
                if 'B2' in action:
                    environment['v'] = 1
                    reward_M += (reward_value['E'] - reward_value['fee'])
                    entry += 'M include B2  '
                if 'P1' in action:
                    environment['v'] = 1
                    environment['P1'] = 1
                    reward += (reward_value['x'] + reward_value['Ca'])
                    reward_B += reward_value['Cb']
                    reward_M += reward_value['fee']
                    entry += 'M include P1  '
                if 'P2' in action:
                    environment['t_P2'] = length_of_stage[str(stage)] + slot
                    environment['P2'] = 1
                    reward_M += reward_value['fee']
                    entry += 'M include P2  '
                if 'C1' in action:
                    environment['v'] = 1
                    environment['C1'] = 1
                    reward += reward_value['Ca']
                    reward_B += (reward_value['x'] + reward_value['Cb'])
                    reward_M += reward_value['fee']
                    entry += 'M include C1  '
                if 'C2' in action:
                    environment['v'] = 1
                    reward_M += (reward_value['E'] - reward_value['fee'])
                    entry += 'M include C2  '
                if 'A1.' in action:
                    environment['A1.'] = 1
                    reward_M += reward_value['fee']
                    entry += 'M include A1.  '
                if 'A2.' in action:
                    environment['v.'] = 1
                    reward_M += (reward_value['E.'] - reward_value['fee'])
                    entry += 'M include A2.  '
                if 'P1.' in action:
                    environment['v.'] = 1
                    environment['P1.'] = 1
                    reward += reward_value['Ca.']
                    reward_B += (reward_value['x'] + reward_value['Cb.'])
                    reward_M += reward_value['fee']
                    entry += 'M include P1.  '
                if 'P2.' in action:
                    environment['P2.'] = 1
                    reward_M += reward_value['fee']
                    entry += 'M include P2.  '
                if 'C1.' in action:
                    environment['v.'] = 1
                    environment['C1.'] = 1
                    reward += (reward_value['x.'] + reward_value['Ca.'])
                    reward_B += reward_value['Cb.']
                    reward_M += reward_value['fee']
                    entry += 'M include C1.  '
                if 'C2.' in action:
                    environment['v.'] = 1
                    reward_M += (reward_value['E.'] - reward_value['fee'])
                    entry += 'M include C2.  '
                environment = check_state(copy.deepcopy(environment), stage)
                if entry != '':
                    sub_sequence.append(entry)
                if action != []:
                    count_for_blanke = 0
                else:
                    count_for_blanke += 1
                if count_for_blanke == 2:
                    stage += 1
                    count_for_blanke = 0
                    slot = 0
                    if stage == 3:
                        #sub_sequence_all = cut(sub_sequence)
                        #record_file_all = open(record_all_path, "a+")
                        #print([sub_sequence_all, reward], file = record_file_all)
                        #record_file_all.close()
                        #record_for_all_sequence.append([sub_sequence_all, reward])
                        if reward > honest_reward_A and [sub_sequence, reward] not in record_for_sequence:
                            record_file = open(record_path, "a+")
                            print([sub_sequence, reward], file = record_file)
                            record_file.close()
                            record_for_sequence.append([sub_sequence, reward])
                    elif stage == 2:
                        sub_sequence.append("after T1' slots")
                        generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)
                    elif stage == 1:
                        sub_sequence.append('after T1 slots')
                        generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)
                    elif stage == 0:
                        sub_sequence.append("after T0' slots")
                        generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)
                    else:
                        sub_sequence.append('after T0 slots')
                        generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)
                else:
                    slot += 1
                    generate_sequence(sub_sequence, environment, copy.deepcopy(blanke_env), 0, count_for_blanke, stage, reward, slot)

def check_state(input, stage):
    output = input
    if input['state'] == 0:
        if input['pre-c'] == 1:
            output['state'] = 'ex'
        elif input['pre-c'] == 0 and stage == -1:
            output['state'] = 'ab'
    return output

def calculate_available_action(agent, environment, stage):
    action_available = []
    if agent == 0:
        if environment['pre-c'] == 0:
            action_available.append('pre-c')
        if environment['pre-b'] == 0:
            action_available.append('pre-b')
        if environment['pre-b.'] == 0 and environment['pre-a'] == 1:
            action_available.append('pre-b.')
        if environment['_P2.'] == 0:
            action_available.append('_P2.')
        if environment['_B1'] == 0:
            action_available.append('_B1')
        if environment['_C1'] == 0:
            action_available.append('_C1')
        if environment['_A1.'] == 0:
            action_available.append('_A1.')
        if environment['_C1.'] == 0:
            action_available.append('_C1.')
    elif agent == 1:
        if environment['pre-a'] == 0:
            action_available.append('pre-a')
        if environment['_P2'] == 0:
            action_available.append('_P2')
        if environment['_P1.'] == 0:
            action_available.append('_P1.')
        if environment['pre-a.'] == 0:
            action_available.append('pre-a.')
        if environment['_B1'] == 0:
            action_available.append('_B1')
        if environment['_C1'] == 0:
            action_available.append('_C1')
        if environment['_A1.'] == 0:
            action_available.append('_A1.')
        if environment['_C1.'] == 0:
            action_available.append('_C1.')
    else:
        if environment['_B1'] == 1 and environment['B1'] == 0:
            action_available.append('B1')
        if environment['B1'] == 0 and environment['pre-c'] == 1 and environment['v'] == 0:
            action_available.append('B2')
        if environment['pre-a'] == 0 and environment['pre-c'] == 1 and environment['v'] == 0:
            action_available.append('P1')
        if (stage == 1 or stage == 2) and (environment['pre-b'] == 1 or environment['_P2'] == 1) and environment['P2'] == 0:
            action_available.append('P2')
        if ((environment['t_P2'] + length_of_stage['tau'] > length_of_stage['2'] and stage == 2) or (environment['t_P2'] + length_of_stage['tau'] <= length_of_stage['2'] and (stage == 2 or stage == 1))) and environment['P2'] == 1 and environment['_C1'] == 1 and environment['v'] == 0:
            action_available.append('C1')
        if environment['pre-a'] == 1 and environment['pre-b'] == 1 and environment['pre-c'] == 1 and environment['v'] == 0:
            action_available.append('C2')
        if stage == 2 and environment['_A1.'] == 1 and environment['A1.'] == 0:
            action_available.append('A1.')
        if environment['A1.'] == 0 and (environment['pre-a.'] == 1 or environment['pre-b'] == 1) and environment['v.'] == 0:
            action_available.append('A2.')
        if environment['pre-b.'] == 1 or environment['_P1.'] == 1 and environment['v.'] == 0:
            action_available.append('P1.')
        if stage == 2 and (environment['pre-a.'] == 1 and environment['_P2.'] == 1) and environment['P2.'] == 0:
            action_available.append('P2.')
        if environment['P2.'] == 1 and environment['_C1.'] == 1 and environment['v.'] == 0:
            action_available.append('C1.')
        if (environment['pre-a'] == 1 and environment['pre-a.'] == 1) or (environment['pre-a.'] == 1 and environment['pre-b'] == 1) and environment['v.'] == 0:
            action_available.append('C2.')
    return action_available

def action_ckeck_legal(action, agent, environment, stage):
    count_du_1 = 0
    count_du_2 = 0
    count_for_C1 = 0
    count_for_C1_ = 0
    if 'B2' in action:
        count_du_1 += 1
    if 'P1' in action:
        count_du_1 += 1
    if 'C1' in action:
        count_du_1 += 1
    if 'C2' in action:
        count_du_1 += 1
    if 'A2.' in action:
        count_du_2 += 1
    if 'P1.' in action:
        count_du_2 += 1
    if 'C1.' in action:
        count_du_2 += 1
    if 'C2.' in action:
        count_du_2 += 1
    if 'P2' in action:
        count_for_C1 += 1
    if 'C1' in action:
        count_for_C1 += 1
    if 'P2.' in action:
        count_for_C1_ += 1
    if 'C1.' in action:
        count_for_C1_ += 1
    if count_du_1 > 1 or count_du_2 > 1 or count_for_C1 > 1 or count_for_C1_ > 1:
        flag = 0
    else:
        flag = 1
    return flag

#-2:start -1:T0 0:T0. 1:T1 2:T1. 3:end

def generate_action(agent, environment, stage, policy):
    action = []
    if agent == 0:
        if stage == -2 and environment['_B1'] == 0:
            action.append('_B1')
        if environment['state'] == 0 and environment['B1'] == 1 and environment['pre-c'] == 0:
            action.append('pre-c')
        if environment['state'] == 'ex':
            if stage < 1 and environment['pre-a'] == 1 and environment['pre-b.'] == 0:
                action.append('pre-b.')
            if stage >= 1 and environment['pre-a'] == 0 and environment['_A1.'] == 0:
                action.append('_A1.')
            if stage >= 1 and environment['pre-a'] == 0 and environment['A1.'] == 1 and environment['pre-b'] == 0:
                action.append('pre-b')
            if environment['P2'] == 1 and environment['_C1'] == 0 and ((environment['t_P2'] + length_of_stage['tau'] > length_of_stage['2'] and stage == 2) or (environment['t_P2'] + length_of_stage['tau'] <= length_of_stage['2'] and (stage == 2 or stage == 1))):
                action.append('_C1')
            if environment['C1'] == 1 and environment['_P2.'] == 0:
                action.append('_P2.')
            if environment['P2.'] == 1 and environment['_C1.'] == 0:
                action.append('_C1.')
        if environment['state'] == 'ab':
            if stage >= 0 and environment['_P2.'] == 0:
                action.append('_P2.')
            if stage >= 0 and environment['_A1.'] == 0:
                action.append('_A1.')
            if stage == 2 and environment['_P2'] == 0 and environment['A1.'] == 1 and environment['pre-b'] == 0:
                action.append('pre-b')
            if environment['P2'] == 1 and environment['_C1'] == 0 and ((environment['t_P2'] + length_of_stage['tau'] > length_of_stage['2'] and stage == 2) or (environment['t_P2'] + length_of_stage['tau'] <= length_of_stage['2'] and (stage == 2 or stage == 1))):
                action.append('_C1')
            if environment['P2.'] == 1 and environment['_C1.'] == 0:
                action.append('_C1.')
        if policy == 'rational':
            if environment['v'] == 0:
                if environment['B1'] == 0 and 'pre-c' in action:
                    action.remove('pre-c')
                if environment['pre-a'] == 1 and 'pre-c' in action and 'pre-b' in action:
                    action.remove('pre-b')
                if environment['pre-a'] == 1 and environment['pre-c'] == 1 and 'pre-b' in action:
                    action.remove('pre-b')
                if environment['pre-a'] == 1 and environment['pre-b'] == 1 and 'pre-c' in action:
                    action.remove('pre-c')
            if environment['v.'] == 0:
                if environment['A1.'] == 0 and 'pre-b' in action:
                    action.remove('pre-b')
                if environment['pre-a.'] == 1 and 'pre-b' in action:
                    action.remove('pre-b')
    elif agent == 1:
        if environment['state'] == 'ex':
            if environment['pre-a'] == 0:
                action.append('pre-a')
            if environment['P1'] == 1 and environment['_P1.'] == 0:
                action.append('_P1.')
            if environment['P2'] == 1 and environment['_C1'] == 0 and ((environment['t_P2'] + length_of_stage['tau'] > length_of_stage['2'] and stage == 2) or (environment['t_P2'] + length_of_stage['tau'] <= length_of_stage['2'] and (stage == 2 or stage == 1))):
                action.append('_C1')
            if environment['P2.'] == 1 and environment['_C1.'] == 0:
                action.append('_C1.')
        if environment['state'] == 'ab':
            if stage >= 0 and environment['_P2'] == 0:
                action.append('_P2')
            if stage >= 0 and environment['_A1.'] == 0:
                action.append('_A1.')
            if stage == 2 and environment['_P2.'] == 0 and environment['A1.'] == 1 and environment['pre-b'] == 0:
                action.append('pre-a.')
            if environment['P2'] == 1 and environment['_C1'] == 0 and ((environment['t_P2'] + length_of_stage['tau'] > length_of_stage['2'] and stage == 2) or (environment['t_P2'] + length_of_stage['tau'] <= length_of_stage['2'] and (stage == 2 or stage == 1))):
                action.append('_C1')
            if environment['P2.'] == 1 and environment['_C1.'] == 0:
                action.append('_C1.')
        if policy == 'rational':
            if environment['v'] == 0:
                if environment['pre-b'] == 1 and environment['pre-c'] == 1 and 'pre-a' in action:
                    action.remove('pre-a')
            if environment['v.'] == 0:
                if environment['A1.'] == 0 and 'pre-a.' in action:
                    action.remove('pre-a.')
                if 'pre-a' in action and 'pre-a.' in action:
                    action.remove('pre-a.')
                if environment['pre-a.'] == 1 and 'pre-a' in action:
                    action.remove('pre-a')
                if environment['pre-a'] == 1 and 'pre-a.' in action:
                    action.remove('pre-a.')
                if environment['pre-b'] == 1 and 'pre-a' in action:
                    action.remove('pre-a')
    return action

def calculate_possible_max_reward(action_available, environment, stage):
    max_reward = 0
    actions = [[]]
    for num in range(len(action_available) + 1):
        for i in itertools.combinations(action_available, num):
            action = list(i)
            if action_ckeck_legal(action, 2, environment, stage):
                reward = 0
                if 'B1' in action:
                    reward += reward_value['fee']
                if 'B2' in action:
                    reward += (reward_value['E'] - reward_value['fee'])
                if 'P1' in action:
                    reward += reward_value['fee']
                if 'P2' in action:
                    reward += reward_value['fee']
                if 'C1' in action:
                    reward += reward_value['fee']
                if 'C2' in action:
                    reward += (reward_value['E'] - reward_value['fee'])
                if 'A1.' in action:
                    reward += reward_value['fee']
                if 'A2.' in action:
                    reward += (reward_value['E.'] - reward_value['fee'])
                if 'P1.' in action:
                    reward += reward_value['fee']
                if 'P2.' in action:
                    reward += reward_value['fee']
                if 'C1.' in action:
                    reward += reward_value['fee']
                if 'C2.' in action:
                    reward += (reward_value['E.'] - reward_value['fee'])
                if reward > max_reward:
                    actions = []
                    actions.append(action)
                    max_reward = reward
                if reward == max_reward:
                    actions.append(action)

    return actions

def cut(sub_sequence):
    count = 0
    for i in range(len(sub_sequence)):
        if 'M' in sub_sequence[i]:
            count = i
    return sub_sequence[0:count + 1]

if __name__ == '__main__':

    record_file = open(record_path, "w")
    print('sequence found', file = record_file)
    record_file.close()

    record_file_all = open(record_all_path, "w")
    print('sequence all', file = record_file_all)
    record_file_all.close()

    generate_sequence([], copy.deepcopy(blanke_env), copy.deepcopy(blanke_env), 0, 0, -2, 0, 0)
'''
    record_file = open(record_path, "w")
    for entry in record_for_sequence:
        print(entry, file = record_file)
    record_file.close()

    record_all_file = open(record_all_path, "w")
    for entry in record_for_all_sequence:
        print(entry, file = record_all_file)
    record_all_file.close()

{'pre-a':0, '_P2':0, '_P1.':0, 'pre-a.':0, 'pre-c':0, 'pre-b':0, 'pre-b.':0, '_P2.':0, '_B1':0, '_C1':0, '_A1.':0, '_C1.':0, 'B1':0, 'P2':0, 'A1.':0, 'P2.':0, 'v':0, 'v.':0, 't_P2':0}
'''