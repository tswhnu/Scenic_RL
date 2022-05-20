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

n_action = 5
n_state_speed = 2
n_state_path = 10
n_state_collision = 10

scenario = scenic.scenarioFromFile('carlaChallenge10_original.scenic',
                                   model='scenic.simulators.carla.model')
simulator = CarlaSimulator(carla_map='Town05', map_path='../../../tests/formats/opendrive/maps/CARLA/Town05.xodr')

# define differnet RL agent for different objectives
RL_speed = DDQN(n_state=n_state_speed, n_action=n_action)
RL_path = DDQN(n_state=n_state_path, n_action=n_action)
RL_collsion = DDQN(n_state=n_state_collision, n_action=n_action)
maxSteps = None

def train(episodes = 100):
    for i in range(episodes):
        scene, _ = scenario.generate()
        simulation = simulator.createSimulation(scene)

        trajectory = simulation.trajectory
        actionSequence = []
        # Initialize dynamic scenario
        veneer.beginSimulation(simulation)
        dynamicScenario = simulation.scene.dynamicScenario
        try:
            # Initialize dynamic scenario
            dynamicScenario._start()

            # Update all objects in case the simulator has adjusted any dynamic
            # properties during setup
            simulation.updateObjects()
            ###################################################################################
            current_state = simulation.get_state()
            speed_state = current_state[-2:]
            done = False
            epi_reward = 0
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
                action = RL_collsion.select_action(current_state)
                action2 = RL_path.select_action(current_state)
                action3 = RL_speed.select_action(current_state[-2:])
                # Run the simulation for a single step and read its state back into Scenic
                new_state, reward, done, _ = simulation.step(action=action)
                RL_collsion.store_transition(current_state, action, reward[0], new_state, done)
                RL_path.store_transition(current_state, action, reward[1], new_state, done)
                RL_speed.store_transition(current_state[-2:], action, reward[2], new_state[-2:], done)

                current_state = new_state
                if RL_collsion.memory_counter > MEMORY_CAPACITY:
                    RL_collsion.optimize_model()
                    RL_path.optimize_model()
                    RL_speed.optimize_model()
                if done:
                    print(reward)

                simulation.updateObjects()
                simulation.currentTime += 1

                # Save the new state
                trajectory.append(simulation.currentState())
                actionSequence.append(allActions)

            if terminationReason is None:
                terminationReason = f'reached time limit ({maxSteps} steps)'
        finally:
            simulation.destroy()
            for obj in simulation.scene.objects:
                disableDynamicProxyFor(obj)
            for agent in simulation.agents:
                agent.behavior.stop()
            for monitor in simulation.scene.monitors:
                monitor.stop()
            veneer.endSimulation(simulation)
train(100)
        # simulation.run(maxSteps=None)
    #         result = simulation.trajectory
    #         for i, state in enumerate(result):
    #                 egoPos = state
