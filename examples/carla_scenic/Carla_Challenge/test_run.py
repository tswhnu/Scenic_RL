import matplotlib.pyplot as plt
import scenic
from scenic.simulators.carla.simulator import CarlaSimulator
import scenic.syntax.veneer as veneer
import types
from collections import OrderedDict
from memory_profiler import profile
from scenic.core.object_types import (enableDynamicProxyFor, setDynamicProxyFor,
                                      disableDynamicProxyFor)
from scenic.core.distributions import RejectionException
import scenic.core.dynamics as dynamics
from scenic.core.errors import RuntimeParseError, InvalidScenarioError
from scenic.core.vectors import Vector
from RL_agent.DDQN import *

from RL_agent.ACTION_SELECTION import *
import time


#####


# the parameters in the threshold list define the range of accpetable action value

# define differnet RL agent for different objectives
def creat_agents(n_action, n_state_list, agent_name_list, load_model=True, current_step=150):
    agent_list = []
    if len(n_state_list) != len(agent_name_list):
        raise Exception('the len of n_state_list and agent_name_list must be same')
    else:
        for i in range(len(n_state_list)):
            agent = DDQN(n_state=n_state_list[i], n_action=n_action, agent_name=agent_name_list[i])
            if load_model:
                agent.load_model(current_step)
            agent_list.append(agent)
    return agent_list


def train(episodes=None, maxSteps=None, RL_agents_list=None, current_episodes=150, n_state_list=None):
    scenario = scenic.scenarioFromFile('carlaChallenge10_original.scenic',
                                       model='scenic.simulators.carla.model')
    simulator = CarlaSimulator(carla_map='Town05', map_path='../../../tests/formats/opendrive/maps/CARLA/Town05.xodr')

    threshold_list = np.array([3, 2])
    TH_start = 0.8
    TH_end = 0.15
    TH_decay = 200
    reward_list = []

    for episode in range(current_episodes, episodes):
        scene, _ = scenario.generate()
        simulation = simulator.createSimulation(scene)

        last_position = None
        actionSequence = []
        # Initialize dynamic scenario
        veneer.beginSimulation(simulation)
        dynamicScenario = simulation.scene.dynamicScenario
        ##########################################################
        start_time = time.time()
        epi_reward = np.zeros(2)
        route = []
        #########################################################

        try:
            # Initialize dynamic scenario
            dynamicScenario._start()

            # Update all objects in case the simulator has adjusted any dynamic
            # properties during setup
            simulation.updateObjects()
            ###################################################################################
            # here get the initial state for the RL agent
            initial_state= simulation.get_state()
            initial_ego_position = np.array([simulation.ego.carlaActor.get_location().x,
                                             simulation.ego.carlaActor.get_location().y])
            state_list = [initial_state[0:n_state_list[0]], initial_state[-n_state_list[1]:]]
            last_position = initial_ego_position
            ###################################################################################
            # Run simulation
            assert simulation.currentTime == 0
            # terminationReason = None
            while maxSteps is None or simulation.currentTime < maxSteps:
                if simulation.verbosity >= 3:
                    print(f'    Time step {simulation.currentTime}:')

                # Run compose blocks of compositional scenarios
                # terminationReason = dynamicScenario._step()

                # Check if any requirements fail
                # dynamicScenario._checkAlwaysRequirements()

                # # Run monitors
                # newReason = dynamicScenario._runMonitors()
                # if newReason is not None:
                #     terminationReason = newReason

                # "Always" and scenario-level requirements have been checked;
                # now safe to terminate if the top-level scenario has finished
                # or a monitor requested termination
                # if terminationReason is not None:
                #     done = True
                #     print(terminationReason)
                # terminationReason = dynamicScenario._checkSimulationTerminationConditions()
                # if terminationReason is not None:
                #     done = True
                #     print(terminationReason)

                # Compute the actions of the agents in this time step
                allActions = OrderedDict()
                schedule = simulation.scheduleForAgents()
                for agent in schedule:
                    behavior = agent.behavior
                    if not behavior._runningIterator:  # TODO remove hack
                        behavior.start(agent)
                    actions = behavior.step()

                    assert isinstance(actions, tuple)
                    if len(actions) == 1 and isinstance(actions[0], (list, tuple)):
                        actions = tuple(actions[0])
                    if not simulation.actionsAreCompatible(agent, actions):
                        raise InvalidScenarioError(f'agent {agent} tried incompatible '
                                                   f' action(s) {actions}')
                    allActions[agent] = actions

                # Execute the actions
                if simulation.verbosity >= 3:
                    for agent, actions in allActions.items():
                        print(f'      Agent {agent} takes action(s) {actions}')
                simulation.executeActions(allActions)

                ####################################################################
                Q_list = []
                # this part is used to obtain the q vlaue of different RL agent
                TH_q = TH_end + (TH_start - TH_end) * math.exp(-1. * episode / TH_decay)
                threshold_list = TH_q * threshold_list
                for i in range(len(RL_agents_list)):
                    q = RL_agents_list[i].action_value(state_list[i])
                    Q_list.append(q)

                action_seq = action_selection(q_list=Q_list, threshold_list=threshold_list,
                                              action_set=np.array(list(range(n_action))), current_eps=episode)
                # if simulation.currentTime % (maxSteps / 2) == 0:
                #     print("action_seq: ", action_seq)
                # the final action will be decided by last action in the list
                action = action_seq[-1]
                # Run the simulation for a single step and read its state back into Scenic
                new_state, reward, done, _ = simulation.step(
                    action=action,
                    last_position=last_position)  # here need to notice that the reward value here will be a list
                new_state_list = [new_state[0:n_state_list[0]], new_state[-n_state_list[1]:]]
                # here we got tge cumulative reward of the current episode
                epi_reward += reward
                for i in range(len(RL_agents_list)):
                    RL_agents_list[i].store_transition(state_list[i], action_seq[i], reward[i], new_state_list[i], done)
                # RL_path.store_transition(state_list[0], action_seq[0], reward_list[0], new_state, done)
                # RL_speed.store_transition(state_list[1], action_seq[1], reward_list[1], new_state[-2:], done)
                # update the state velue
                state_list = new_state_list
                if RL_agents_list[0].memory_counter > MEMORY_CAPACITY:
                    for RL_agent in RL_agents_list:
                        RL_agent.optimize_model()
                if done:
                    print("reward_info: ", epi_reward / simulation.currentTime)
                    reward_list.append(epi_reward / simulation.currentTime)
                    reward_array = np.array(reward_list)
                    np.save("./log/reward_list" + str(episode) + ".npy", reward_array)
                    break

                simulation.updateObjects()
                simulation.currentTime += 1

                # Save the new state
                ##########################################################
                last_position = np.array([simulation.ego.carlaActor.get_location().x,
                                             simulation.ego.carlaActor.get_location().y])
                # trajectory.append(new_state[1])
                # route.append([new_state[0], new_state[1]])
                actionSequence.append(allActions)

        finally:
            # route = np.array(route)
            # trajectory = np.array(trajectory)
            np.save("./log/vehicle_trajectory" + str(episode) + ".npy", simulation.driving_trajectory)
            np.save("./log/reference_route" + str(episode) + ".npy", simulation.driving_route)
            np.save("./log/vehicle_speed" + str(episode) + ".npy", simulation.speed_list)
            # driving_trajectory = simulation.driving_trajectory
            # plt.plot(trajectory[:, 0], trajectory[:, 1])
            # plt.scatter(simulation.ego_spawn_point[0], simulation.ego_spawn_point[1])
            # plt.plot(route[:, 0], route[:, 1])
            # plt.legend(['trajectory','point', 'route'])
            # plt.show()
            ##############################################################################
            epi_end = time.time()
            epi_dur = epi_end - start_time
            if episode % 20 == 0:
                print("current_epi:", episode, "last_epi_duration:", epi_dur)
                print(epi_reward / (simulation.currentTime + 1))
            if episode % 100 == 0:
                print("saving model")
                for RL_agent in RL_agents_list:
                    RL_agent.save_model(episode)
            ##############################################################################
            simulation.destroy()
            for obj in simulation.scene.objects:
                disableDynamicProxyFor(obj)
            for agent in simulation.agents:
                agent.behavior.stop()
            for monitor in simulation.scene.monitors:
                monitor.stop()
            veneer.endSimulation(simulation)


n_action = 9
n_state_list = [7, 2]
agent_name_list = ['path', 'speed']
RL_agents_list = creat_agents(n_action=n_action, n_state_list=n_state_list, agent_name_list=agent_name_list,
                              load_model=True, current_step=500)
train(episodes=5000, RL_agents_list=RL_agents_list, current_episodes=500, maxSteps=1000, n_state_list=n_state_list)

# simulation.run(maxSteps=None)
#         result = simulation.trajectory
#         for i, state in enumerate(result):
#                 egoPos = state
