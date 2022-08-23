import matplotlib.pyplot as plt
import scenic
from scenic.simulators.carla.simulator import CarlaSimulator
import scenic.syntax.veneer as veneer
import types
from collections import OrderedDict
from scenic.core.object_types import (enableDynamicProxyFor, setDynamicProxyFor,
                                      disableDynamicProxyFor)
from scenic.core.distributions import RejectionException
import scenic.core.dynamics as dynamics
from scenic.core.errors import RuntimeParseError, InvalidScenarioError
from scenic.core.vectors import Vector
from RL_agent.DDQN import *

from RL_agent.ACTION_SELECTION import *
import time
from scenic.simulators.carla.utils.generate_traffic import *
import numpy as np
import pygame
from scenic.simulators.carla.utils.HUD_render import *
#####


# the parameters in the threshold list define the range of accpetable action value

# define differnet RL agent for different objectives
def creat_agents(n_action, n_state_list, agent_name_list, load_model=True, current_step=150, test_mode=False):
    agent_list = []
    if len(n_state_list) != len(agent_name_list):
        raise Exception('the len of n_state_list and agent_name_list must be same')
    else:
        for i in range(len(n_state_list)):
            agent = DDQN(n_state=n_state_list[i], n_action=n_action, agent_name=agent_name_list[i], test=test_mode[i])
            if load_model[i]:
                agent.load_model(current_step[i])
                print("model_" + str(i) + "lodaing")
            agent_list.append(agent)
    return agent_list


def train(episodes=None, maxSteps=None, RL_agents_list=None,
          current_episodes=150, n_state_list=None,
          npc_vehicle_num = 100, npc_ped_num = 100,
          traffic_generation = False, save_model=False, test_list=[True, True, True], render_hud = True, save_log=False):
    scenario = scenic.scenarioFromFile('carlaChallenge10.scenic',
                                       model='scenic.simulators.carla.model')
    simulator = CarlaSimulator(carla_map='Town05', map_path='../../../tests/formats/opendrive/maps/CARLA/Town05.xodr')

    threshold_list = np.array([3, 3])
    TH_start = 0.8
    TH_end = 0.15
    TH_decay = 200
    reward_list = []
    simulation = None
    initial_action_set = np.arange(n_action)

    try:
        if render_hud:
            # Init Pygame
            pygame.init()
            display = pygame.display.set_mode(
                (1920, 1080),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

            # Place a title to game window
            pygame.display.set_caption('test')

            # Show loading screen
            font = pygame.font.Font(pygame.font.get_default_font(), 20)
            text_surface = font.render('Rendering map...', True, COLOR_WHITE)
            display.blit(text_surface, text_surface.get_rect(center=(1920 / 2, 1080 / 2)))
            pygame.display.flip()
            clock = pygame.time.Clock()
        if traffic_generation:
            vehicle_list, all_id, all_actors = generate_traffic(vehicle_num=npc_vehicle_num,
                                                                ped_num=npc_ped_num,
                                                                carla_client=simulator.client)

        for episode in range(current_episodes, episodes):

            scene, _ = scenario.generate()
            simulation = simulator.createSimulation(scene, render_hud=render_hud)
            simulation.episode = episode

            last_position = None
            actionSequence = []
            # Initialize dynamic scenario
            veneer.beginSimulation(simulation)
            dynamicScenario = simulation.scene.dynamicScenario
            ##########################################################
            start_time = time.time()
            epi_reward = np.array([0.0,0.0])
            totoal_reward = np.array([0.0,0.0])
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
                state_list = [initial_state[0][-2:], initial_state[0],  initial_state[1]]
                last_position = initial_ego_position
                ###################################################################################
                # Run simulation
                assert simulation.currentTime == 0
                # terminationReason = None
                while maxSteps is None or simulation.currentTime < maxSteps:
                    if render_hud:
                        simulation.clock.tick(60)
                        display.fill(COLOR_ALUMINIUM_4)
                        simulation.rendering(display)
                        simulation.hud.render(display)
                        pygame.display.flip()
                    if simulation.verbosity >= 3:
                        print(f'    Time step {simulation.currentTime}:')

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
                    for i in range(len(RL_agents_list)):
                        # collsion can path following agent can seclect action from what they want
                        action_value = RL_agents_list[i].action_value(state_list[i])
                        Q_list.append(action_value)
                    action_seq = action_selection(Q_list, threshold_list, initial_action_set, episode, test_list=test_list)
                    final_action = action_seq[-1]
                    # Run the simulation for a single step and read its state back into Scenic
                    new_state, reward, done, _ = simulation.step(
                        action=final_action)  # here need to notice that the reward value here will be a list
                    new_state_list = [new_state[0][-2:], new_state[0], new_state[1]]
                    # here we got tge cumulative reward of the current episode
                    epi_reward += reward

                    for i in range(len(RL_agents_list)):
                        # there need to notice that all the objectives need to store same action
                        RL_agents_list[i].store_transition(state_list[i], final_action, reward[i], new_state_list[i], done)
                    # RL_path.store_transition(state_list[0], action_seq[0], reward_list[0], new_state, done)
                    # RL_speed.store_transition(state_list[1], action_seq[1], reward_list[1], new_state[-2:], done)
                    # update the state velue
                    state_list = new_state_list
                    if RL_agents_list[0].memory_counter > MEMORY_CAPACITY:
                        for i, RL_agent in enumerate(RL_agents_list):
                            if test_list[i] is not True:
                                RL_agent.optimize_model()
                    if done:
                        print("reward_info: ", epi_reward / simulation.currentTime)
                        totoal_reward += epi_reward /simulation.currentTime
                        reward_list.append(epi_reward / simulation.currentTime)
                        reward_array = np.array(reward_list)
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
                # simulation.draw_trace(simulation.driving_trajectory)
                if save_log:
                    np.save("./log_01/reward_list" + str(episode) + ".npy", reward_array)
                    np.save("./log_01/vehicle_trajectory" + str(episode) + ".npy", simulation.reference_route)
                    np.save("./log_01/reference_route" + str(episode) + ".npy", simulation.driving_trajectory)
                    np.save("./log_01/vehicle_speed" + str(episode) + ".npy", simulation.speed_list)
                # reference_route = simulation.reference_route
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
                    for i, RL_agent in enumerate(RL_agents_list):
                        if save_model[i] is True:
                            RL_agent.save_model(episode)
                    if episode != 0:
                        print('total reward:', totoal_reward / episode)
                ##############################################################################
                simulation.destroy()
                for obj in simulation.scene.objects:
                    disableDynamicProxyFor(obj)
                for agent in simulation.agents:
                    agent.behavior.stop()
                for monitor in simulation.scene.monitors:
                    monitor.stop()
                veneer.endSimulation(simulation)
    finally:
        if traffic_generation and simulation is not None:
            print('\ndestroying %d vehicles' % len(vehicle_list))
            simulator.client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
            # stop walker controllers (list is [controller, actor, controller, actor ...])
            for i in range(0, len(all_id), 2):
                all_actors[i].stop()
            print('\ndestroying ' +  str(len(all_actors) // 2) + ' walkers')
            simulator.client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
            time.sleep(0.5)
        else:
            pass

n_action = 25
# n_state_list = [7, 2]
# agent_name_list = ['path', 'speed']
# n_state_list = [2, 8]
# test_list = [True, False]
# load_model = [True, False]
# save_model = [False, True]
# step_list = [600, 0]
# agent_name_list = ['speed', 'path_following']
n_state_list = [2, 8]
test_list = [False, False]
load_model = [False, False]
save_model = [False, False]
step_list = [0, 0]
agent_name_list = ['speed', 'path']
RL_agents_list = creat_agents(n_action=n_action, n_state_list=n_state_list, agent_name_list=agent_name_list,
                              load_model=load_model, current_step=step_list, test_mode=test_list)
train(episodes=1000, RL_agents_list=RL_agents_list, current_episodes=0,
      maxSteps=1000, n_state_list=n_state_list, traffic_generation=False, save_model=save_model, test_list=test_list,
      render_hud=False, save_log=False)

# simulation.run(maxSteps=None)
#         result = simulation.trajectory
#         for i, state in enumerate(result):
#                 egoPos = state
