import random

import numpy as np
import math

EPS_DECAY = 1000
EPS_START = 0.9
EPS_END = 0.05


def action_selection(q_list, threshold_list, action_set, current_eps, test_list):
    action_seq = []

    E_thresh = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * current_eps / EPS_DECAY)
    # first check the length of the Q_list and threshold_list
    if len(q_list) != len(threshold_list):
        print("the length of the Q-list must same with the length of threshold_list")
        return None
    else:
        for i in range(len(q_list)):
            if len(action_set) == 1:
                # when there only have one action in the action set,
                # the current agent have no choice but only use this action
                action_seq.append(action_set[0])
            else:
                p = np.random.random()
                # for each agent, there has possibility that this agent will choose random action from Ai-1
                if test_list[i]:
                    E_thresh = EPS_END
                if p > E_thresh or test_list[i]:
                    current_q = q_list[i].data.cpu().numpy()[0]
                    threshold = threshold_list[i]
                    # find the max action and max action value
                    max_action = np.argmax(current_q)
                    max_action_value = current_q[max_action]
                    # here we extend the acceptable action value range
                    low_bound = max_action_value - threshold
                    action_value_set = current_q[action_set]
                    new_action_set = action_set[action_value_set >= low_bound]
                    # new_action = random.choice(new_action_set)
                    action_seq.append(max_action)
                    if len(new_action_set) == 0:
                        action_set = np.array([max_action])
                    else:
                        action_set = new_action_set
                else:
                    action = np.random.choice(action_set)
                    action_set = np.array([action])
                    action_seq.append(action)
        return action_seq
# a = np.array([0, 1, 3])
# while True:
#     action = np.random.choice(a)
#     if len(np.array([action])) == 0:
#         break
