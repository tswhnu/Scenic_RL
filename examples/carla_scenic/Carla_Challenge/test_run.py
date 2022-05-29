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

n_action = 9
n_state_speed = 2
n_state_path = 10
n_state_collision = 10

scenario = scenic.scenarioFromFile('carlaChallenge10_original.scenic',
                                   model='scenic.simulators.carla.model')
simulator = CarlaSimulator(carla_map='Town05', map_path='../../../tests/formats/opendrive/maps/CARLA/Town05.xodr')

# define differnet RL agent for different objectives
RL_speed = DDQN(n_state=n_state_speed, n_action=n_action)
RL_path = DDQN(n_state=n_state_path, n_action=n_action)
RL_collision = DDQN(n_state=n_state_collision, n_action=n_action)
agents_list = [RL_collision, RL_path, RL_speed]
threshold_list = [2, 2, 2]
# used to store a values from differnt objectives
Q_list = []
maxSteps = 800


def train(episodes=100):
    for episode in range(episodes):
        scene, _ = scenario.generate()
        simulation = simulator.createSimulation(scene)

        trajectory = simulation.trajectory
        actionSequence = []
        # Initialize dynamic scenario
        veneer.beginSimulation(simulation)
        dynamicScenario = simulation.scene.dynamicScenario
        ##########################################################
        start_time = time.time()
        epi_reward = np.zeros(len(agents_list))
        #########################################################


        try:
            # Initialize dynamic scenario
            dynamicScenario._start()

            # Update all objects in case the simulator has adjusted any dynamic
            # properties during setup
            simulation.updateObjects()
            ###################################################################################
            # here get the initial state for the RL agent
            state_list = [simulation.get_state(), simulation.get_state(), simulation.get_state()[-2:]]
            ###################################################################################
            # Run simulation
            assert simulation.currentTime == 0
            terminationReason = None
            while maxSteps is None or simulation.currentTime < maxSteps:
                if simulation.verbosity >= 3:
                    print(f'    Time step {simulation.currentTime}:')

                # Run compose blocks of compositional scenarios
                terminationReason = dynamicScenario._step()

                # Check if any requirements fail
                dynamicScenario._checkAlwaysRequirements()

                # Run monitors
                newReason = dynamicScenario._runMonitors()
                if newReason is not None:
                    terminationReason = newReason

                # "Always" and scenario-level requirements have been checked;
                # now safe to terminate if the top-level scenario has finished
                # or a monitor requested termination
                if terminationReason is not None:
                    break
                terminationReason = dynamicScenario._checkSimulationTerminationConditions()
                if terminationReason is not None:
                    break

                # Compute the actions of the agents in this time step
                allActions = OrderedDict()
                schedule = simulation.scheduleForAgents()
                for agent in schedule:
                    behavior = agent.behavior
                    if not behavior._runningIterator:  # TODO remove hack
                        behavior.start(agent)
                    actions = behavior.step()
                    # if isinstance(actions, EndSimulationAction):
                    #     terminationReason = str(actions)
                    #     break
                    assert isinstance(actions, tuple)
                    if len(actions) == 1 and isinstance(actions[0], (list, tuple)):
                        actions = tuple(actions[0])
                    if not simulation.actionsAreCompatible(agent, actions):
                        raise InvalidScenarioError(f'agent {agent} tried incompatible '
                                                   f' action(s) {actions}')
                    allActions[agent] = actions
                if terminationReason is not None:
                    break

                # Execute the actions
                if simulation.verbosity >= 3:
                    for agent, actions in allActions.items():
                        print(f'      Agent {agent} takes action(s) {actions}')
                simulation.executeActions(allActions)

                ####################################################################
                Q_list = []
                # this part is used to obtain the q vlaue of different RL agent
                for i in range(len(agents_list)):
                    # state for current RL agent
                    agent_state = state_list[i]
                    current_agent = agents_list[i]
                    current_q = current_agent.action_value(agent_state)
                    Q_list.append(current_q)

                action_seq = action_selection(q_list=Q_list, threshold_list=threshold_list,
                                              action_set=np.array(list(range(9))), current_eps=episode)
                if simulation.currentTime % (maxSteps / 2) == 0:
                    print("action_seq: ", action_seq)
                # the final action will be decided by last action in the list
                action = action_seq[-1]
                # Run the simulation for a single step and read its state back into Scenic
                new_state, reward_list, done, _ = simulation.step(
                    action=action)  # here need to notice that the reward value here will be a list
                print("len: new state:", len(new_state))
                new_state_list = [new_state, new_state, new_state[-2:]]
                # here we got tge cumulative reward of the current episode
                epi_reward += reward_list
                for i in range(len(agents_list)):
                    agents_list[i].store_transition(state_list[i], action_seq[i], reward_list[i], new_state_list[i],
                                                    done)
                    # RL_path.store_transition(state_list[0], action_seq[0], reward_list[0], new_state, done)
                    # RL_speed.store_transition(state_list[1], action_seq[1], reward_list[1], new_state[-2:], done)
                # update the state velue
                state_list = new_state_list
                if agents_list[0].memory_counter > MEMORY_CAPACITY:
                    for agent in agents_list:
                        agent.optimize_model()
                if done:
                    print(reward_list)
                    reward_list.append(epi_reward / simulation.currentTime)
                    break

                simulation.updateObjects()
                simulation.currentTime += 1

                # Save the new state
                trajectory.append(simulation.currentState())
                actionSequence.append(allActions)

            if terminationReason is None:
                terminationReason = f'reached time limit ({maxSteps} steps)'

        finally:
            ##############################################################################
            epi_end = time.time()
            epi_dur = epi_end - start_time
            print("current_epi:", episode)
            if episode % 10 == 0:
                print("current_epi:", episode, "last_epi_duration:", epi_dur)
                print(epi_reward / simulation.currentTime)
            if episode % 30 == 0:
                for j in range(len(agents_list)):
                     print("saving model")
                     agents_list[j].save_model("./policy_agent_" + str(j) + "_episode_" + str(episode) + ".pt",
                                               "./target_agent_" + str(j) + "_episode_" + str(episode) + ".pt")
            ##############################################################################
            simulation.destroy()
            for obj in simulation.scene.objects:
                disableDynamicProxyFor(obj)
            for agent in simulation.agents:
                agent.behavior.stop()
            for monitor in simulation.scene.monitors:
                monitor.stop()
            veneer.endSimulation(simulation)


train(2000)

# simulation.run(maxSteps=None)
#         result = simulation.trajectory
#         for i, state in enumerate(result):
#                 egoPos = state
