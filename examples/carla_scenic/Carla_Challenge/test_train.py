import time

import cv2
import tqdm
from ENV import *
import matplotlib.pyplot as plt
from bird_view.lidar_birdeye import *
import matplotlib.pyplot as plt
from RL_agent.DDQN import *
from RL_agent.ACTION_SELECTION import *
TRAIN_EPISODES = 5000

n_action = 9
n_state_speed = 2
n_state_path = 10
n_state_collision = 10

# define differnet RL agent for different objectives
RL_speed = DDQN(n_state=n_state_speed, n_action=n_action)
RL_path = DDQN(n_state=n_state_path, n_action=n_action)
agents_list = [RL_path, RL_speed]
threshold_list = [5, 5]
# used to store a values from differnt objectives
Q_list = []

env = CarEnv()

torch.cuda.empty_cache()
destination = [21.0, 28.4]
save_picture = False
reward_list = []
save_model = False
# agent.load_model()

try:
    for episode in range(100):
        epi_reward = 0
        step = 1
        # reset the environment
        state_list = [env.reset(), env.reset()[-2:]]
        # reset the finish flag
        done = False
        # get the start time
        episode_start = time.time()
        episode_dur = 10
        # begin to drive
        while True:
            Q_list = []
            for i in range(len(agents_list)):
                current_state = state_list[i]
                current_agent = agents_list[i]
                current_q = current_agent.action_value(state_list[i])
                Q_list.append(current_q)
            action_seq = action_selection(q_list=Q_list, threshold_list=threshold_list,
                                          action_set=np.array(list(range(9))))
            # get the action based on the current state
            action = action_seq[-1]
            #0 get the result basde on the action
            new_state, reward, done, _ = env.step(action)  # here need to notice that for MORL, reward will be a vector
            new_state_list = [new_state, new_state[-2:]]
            for i in range(len(agents_list)):
                agents_list[i].store_transition(state_list[i], action, reward[i], new_state_list[i], done)
            state_list = new_state_list
            step += 1
            print(step)
            if agents_list[0].memory_counter > MEMORY_CAPACITY:
                for agent in agents_list:
                    print("model optimizing")
                    agent.optimize_model()

            if done:
                print("epi_reward:", epi_reward / step)
                reward_list.append(epi_reward / step)
                break
        if episode % 10 == 0:
            x = np.arange(0, len(v_list))
            plt.plot(x, v_list)
            plt.title('velocity')
            plt.show()
        if episode % 100 == 0 and save_model:
            print('saving data')
            plt.plot(reward_list)
            plt.ylim(-10, 10)
            plt.xlim(0, 900)
            plt.show()
            np.save('average_reward.npy', np.array(reward_list))
            agent.save_model("./policy_net"+str(episode)+'.pt', "./target_net"+str(episode)+'.pt')
        env.terminal()
        # env.client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
finally:
    env.terminal()
