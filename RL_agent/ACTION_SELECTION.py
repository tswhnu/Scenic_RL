import numpy as np
def action_selection(q_list, threshold_list, action_set):
    action_seq = []
    # first check the length of the Q_list and threshold_list
    if len(q_list) != len(threshold_list):
        print("the length of the Q-list must same with the length of threshold_list")
        return None
    else:
        # if the action num in the current action set is 1, there is no need to select action from it
        if len(action_set) == 1:

            action_seq.append(action_set)
            return action_seq
        else:
            for i in range(len(q_list)):
                current_q = q_list[i].data.cpu().numpy()[0]
                threshold = threshold_list[i]
                # find the max action and max action value
                max_action = np.argmax(current_q)
                max_action_value = current_q[max_action]
                action_seq.append(max_action)
                # here we extend the acceptable action value range
                low_bound = max_action_value - threshold
                action_value_set = current_q[action_set]
                new_action_set = action_set[action_value_set >= low_bound]
                action_set = new_action_set
            return action_seq









