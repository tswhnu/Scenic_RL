import random

import numpy as np
import math

EPS_DECAY = 500
EPS_START = 0.9
EPS_END = 0.05


def action_selection(q_list, threshold_list, action_set, current_eps, test_list):
    action_seq = []
    initial_action_set = action_set
    initial_action_table = np.array(initial_action_set).reshape(5, 5)
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
                if p > E_thresh:
                    #the agent will choose action based on the q value
                    current_q = q_list[i].data.cpu().numpy()[0]
                    # the threshold value will decide the range of action set
                    threshold = threshold_list[i]
                    # here we first get the value of actions in the action set
                    action_value_set = current_q[action_set]
                    # find the max action value in the action value set
                    max_action_value = np.max(action_value_set)
                    # here is the max action which have biggest action value
                    max_action_inset = action_set[np.argmax(action_value_set)]
                    # for actions choose by speed control agent
                    # since it will use differnet action selection way
                    if i == 0:
                        # find which class of speed control action that belongs to
                        action_pos = np.where(initial_action_table == max_action_inset)[1][0]
                        q_table = current_q.reshape(5, 5)
                        avg_list = np.average(q_table, axis=0)
                        low_bound = avg_list[action_pos] - threshold
                        fullfilled_actions = np.where(avg_list >= low_bound)[0]
                        new_action_set = initial_action_table[:, fullfilled_actions].reshape(-1)
                        action_seq.append(max_action_inset)
                    else:
                        # the low-bound calculate from the threshold
                        low_bound = max_action_value - threshold
                        # find all the actions that the action value is above this low-bound
                        new_action_set = action_set[action_value_set >= low_bound]
                        # the new action for this objective is random choose from new action set since all of them are considered equally good
                        new_action = random.choice(new_action_set)
                        action_seq.append(new_action)
                    if len(new_action_set) == 0:
                        action_set = np.array([new_action])
                    else:
                        action_set = new_action_set
                else:
                    # the agent will explor on current action set
                    action = np.random.choice(action_set)
                    action_set = np.array([action])
                    action_seq.append(action)
        return action_seq
# a = np.array([0, 1, 3])
# while True:
#     action = np.random.choice(a)
#     if len(np.array([action])) == 0:
#         break
