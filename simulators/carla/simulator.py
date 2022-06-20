"""Simulator interface for CARLA."""
import numpy
import numpy as np

try:
    import carla
except ImportError as e:
    raise ModuleNotFoundError('CARLA scenarios require the "carla" Python package') from e

import math
import os
import time
from scenic.syntax.translator import verbosity

if verbosity == 0:  # suppress pygame advertisement at zero verbosity
    import os

    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame
##########
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from bird_view.birdview_semantic import *
from matplotlib import pyplot as plt
import numpy as np
from scenic.simulators.carla.utils.Reward_functions import *
from agents.navigation.controller import *
#########
from scenic.domains.driving.simulators import DrivingSimulator, DrivingSimulation
from scenic.core.simulators import SimulationCreationError
from scenic.syntax.veneer import verbosePrint
import scenic.simulators.carla.utils.utils as utils
import scenic.simulators.carla.utils.visuals as visuals


class CarlaSimulator(DrivingSimulator):
    """Implementation of `Simulator` for CARLA."""

    def __init__(self, carla_map, map_path, address='127.0.0.1', port=2000, timeout=20,
                 render=False, record='', timestep=0.1):
        super().__init__()
        verbosePrint('Connecting to CARLA...')
        self.client = carla.Client(address, port)
        self.client.set_timeout(timeout)  # limits networking operations (seconds)
        if carla_map is not None:
            self.world = self.client.load_world(carla_map)
        else:
            if map_path.endswith('.xodr'):
                with open(map_path) as odr_file:
                    self.world = self.client.generate_opendrive_world(odr_file.read())
            else:
                raise RuntimeError(f'CARLA only supports OpenDrive maps')
        self.timestep = timestep

        self.tm = self.client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)

        # Set to synchronous with fixed timestep
        settings = self.world.get_settings()
        settings.no_rendering_mode = False
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = timestep  # NOTE: Should not exceed 0.1
        self.world.apply_settings(settings)
        verbosePrint('Map loaded in simulator.')

        self.render = render  # visualization mode ON/OFF
        self.record = record  # whether to use the carla recorder
        self.scenario_number = 0  # Number of the scenario executed

    def createSimulation(self, scene, verbosity=0):
        self.scenario_number += 1
        return CarlaSimulation(scene, self.client, self.tm, self.timestep,
                               render=self.render, record=self.record,
                               scenario_number=self.scenario_number, verbosity=verbosity)

    def destroy(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(False)

        super().destroy()


class CarlaSimulation(DrivingSimulation):
    def __init__(self, scene, client, tm, timestep, render, record, scenario_number, verbosity=0, speed_limit=25):
        super().__init__(scene, timestep=timestep, verbosity=verbosity)
        self.client = client
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprintLib = self.world.get_blueprint_library()
        self.tm = tm

        ##################
        self.spectator = self.world.get_spectator()
        self.collision_history = []
        self.speed_limit = speed_limit  # km/h
        self.ego_spawn_point = None
        self.driving_trajectory = []
        self.driving_route = []
        self.speed_list = []
        self.tra_point_index = 0
        self.ego_throttle = 0
        self.ego_brake = 0
        self.ego_steer = 0
        ##################

        weather = scene.params.get("weather")
        if weather is not None:
            if isinstance(weather, str):
                self.world.set_weather(getattr(carla.WeatherParameters, weather))
            elif isinstance(weather, dict):
                self.world.set_weather(carla.WeatherParameters(**weather))

        # Reloads current world: destroys all actors, except traffic manager instances
        # self.client.reload_world()

        # Setup HUD
        self.render = render
        self.record = record
        self.scenario_number = scenario_number
        if self.render:
            self.displayDim = (1280, 720)
            self.displayClock = pygame.time.Clock()
            self.camTransform = 0
            pygame.init()
            pygame.font.init()
            self.hud = visuals.HUD(*self.displayDim)
            self.display = pygame.display.set_mode(
                self.displayDim,
                pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            self.cameraManager = None

        if self.record:
            if not os.path.exists(self.record):
                os.mkdir(self.record)
            name = "{}/scenario{}.log_01".format(self.record, self.scenario_number)
            self.client.start_recorder(name)

        # Create Carla actors corresponding to Scenic objects
        self.ego = None
        for obj in self.objects:
            carlaActor = self.createObjectInSimulator(obj)

            # Check if ego (from carla_scenic_taks.py)
            if obj is self.objects[0]:
                self.ego = obj
                ###########################################################################
                for i in range(len(self.ego.trajectory)):
                    if i == 2:
                        self.driving_trajectory += list(self.ego.trajectory[i].centerline.points)
                    else:
                        self.driving_trajectory += list(self.ego.trajectory[i].centerline.points[0:-1])
                self.driving_trajectory = np.array(self.driving_trajectory)
                # there need to have a transfer between the scenic position and carla position
                self.driving_trajectory[:, 0] = self.driving_trajectory[:, 0]
                self.driving_trajectory[:, 1] = -self.driving_trajectory[:, 1]
                ###################################################################################
                # the starting and end point of the trajectory
                self.ego_spawn_point = [obj.position[0], -obj.position[1]]
                # find the cloest point in the trajectory list
                self.tra_point_index = self.find_closest_point(self.driving_trajectory, self.ego_spawn_point)
                if self.tra_point_index == 0:
                    self.tra_point_index += 1
                ###########################################################
                # add routeplanner for vehicles in carla
                self.route_planner = GlobalRoutePlanner(self.map, sampling_resolution=2.0)

                # add collision sensor to the ego vehicle
                # osition of the collision sensor
                transform = carla.Transform(carla.Location(x=2.5, z=0.7))
                collision_sensor = self.blueprintLib.find("sensor.other.collision")
                self.collision_sensor = self.world.spawn_actor(collision_sensor, transform, attach_to=carlaActor)
                self.collision_sensor.listen(lambda event: self.collision_data(event))

                # Set up camera manager and collision sensor for ego
                if self.render:
                    camIndex = 0
                    camPosIndex = 0
                    self.cameraManager = visuals.CameraManager(self.world, carlaActor, self.hud)
                    self.cameraManager._transform_index = camPosIndex
                    self.cameraManager.set_sensor(camIndex)
                    self.cameraManager.set_transform(self.camTransform)

        self.world.tick()  ## allowing manualgearshift to take effect 	# TODO still need this?

        for obj in self.objects:
            if isinstance(obj.carlaActor, carla.Vehicle):
                obj.carlaActor.apply_control(carla.VehicleControl(manual_gear_shift=False))

        self.world.tick()

        # Set Carla actor's initial speed (if specified)
        for obj in self.objects:
            if obj.speed is not None:
                equivVel = utils.scenicSpeedToCarlaVelocity(obj.speed, obj.heading)
                if hasattr(obj.carlaActor, 'set_target_velocity'):
                    obj.carlaActor.set_target_velocity(equivVel)
                else:
                    obj.carlaActor.set_velocity(equivVel)

    ########################################################################################################################
    # def trace_route(self, waypoint_mode=False, points_num=3):
    #     # here the destination is the last point of the endline
    #     destination = self.ego.trajectory[2].centerline.points[-1]
    #     destination = carla.Location(x=destination[0], y=destination[1], z=0.5)
    #     current_location = utils.scenicToCarlaLocation(self.ego.position, world=self.world,
    #                                                    blueprint=self.ego.blueprint)
    #     trace = self.route_planner.trace_route(current_location, destination)
    #     if len(trace) < points_num:
    #         for i in range(points_num - len(trace)):
    #             trace.append(trace[-1])
    #     else:
    #         # here we pickup first n points in the waypoints list
    #         trace = trace[:points_num]
    #     if not waypoint_mode:
    #         return [[i[0].transform.location.x, i[0].transform.location.y] for i in trace]
    #     else:
    #         return trace
    def trace_route(self, point_num=3):
        # here giving the future three points based on the current trajectory and the vehicle position
        route = self.driving_trajectory[(self.tra_point_index - 1):self.tra_point_index + point_num - 1]
        route = list(route)
        if len(route) < point_num:
            for i in range(point_num - len(route)):
                route.append(route[-1])
        return route

    def angle(self, point1, point2):
        x_diff = point2[0] - point1[0]
        y_diff = point2[1] - point1[1]
        if x_diff == 0:
            if point2[1] > point1[1]:
                return 90
            else:
                return -90
        result = math.atan(y_diff / x_diff) * (180 / math.pi)
        if x_diff < 0 and result > 0:
            return result - 180
        elif x_diff < 0 and result < 0:
            return result + 180
        else:
            return result

    def draw_trace(self, ego_vehicle_location, trace):
        ego_vehicle_location = np.array(ego_vehicle_location)
        trace = np.array(trace)
        new_trace = trace - ego_vehicle_location

    def destination_reward_test(self, destination, ego_location):
        return math.sqrt(
            (ego_location[0] - destination[0]) ** 2 + (ego_location[1] - destination[1]) ** 2)

    def angle_diff(self, route_angle):
        vehicle_yaw = self.ego.carlaActor.get_transform().rotation.yaw

        angle_diff = route_angle - vehicle_yaw
        if angle_diff <= -180:
            angle_diff += 360
        elif angle_diff >= 180:
            angle_diff -= 360
        return angle_diff

    def get_state(self):
        # route = np.array(self.trace_route())
        # trace = np.array(self.trace_route()).reshape(-1)
        # route_angle = math.atan((route[1][1] - route[0][1]) / (route[1][0] - route[0][0])) * 180 / math.pi
        # vehicle_yaw = self.ego.carlaActor.get_transform().rotation.yaw
        # ego_location = np.array([self.ego.carlaActor.get_location().x, self.ego.carlaActor.get_location().y])

        # route information
        route = np.array(self.trace_route())
        ego_location = np.array([self.ego.carlaActor.get_location().x, self.ego.carlaActor.get_location().y])

        # the angle of vector between current vehicle location and following tracking waypoint
        angle1 = self.angle(ego_location, route[2])
        # the angle of vector between current vehicle location and current tracking waypoint
        angle2 = self.angle(ego_location, route[1])

        # vehicle states

        self.driving_route.append(list(ego_location))
        ego_car_speed = np.array([self.ego.carlaActor.get_velocity().x, self.ego.carlaActor.get_velocity().y])
        vehicle_vel = math.sqrt(ego_car_speed[0] ** 2 + ego_car_speed[1] ** 2)
        current_speed = 3.6 * vehicle_vel  # km/h
        self.speed_list.append(current_speed)
        angular_velocity = self.ego.carlaActor.get_angular_velocity().z

        # calculate difference
        # the distance between the vehicle and current route
        route_distance = self.route_distance(route[0], route[1], ego_location)
        # the distance between the vechile and precious waypoint
        distance1 = self.distance(ego_location, route[0])
        # dustance between the previous waypoint and current waypoint
        distance2 = self.distance(route[0], route[1])
        angle_diff1 = self.angle_diff(angle1)
        angle_diff2 = self.angle_diff(angle2)

        state = np.array([route_distance, distance1, distance2, angle_diff1, angle_diff2,
                          self.ego_steer, current_speed, self.speed_limit]).astype(np.single)

        return state

        # return np.append(trace, [ego_location, ego_speed]).astype(np.single)

    def distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def route_distance(self, point1, point2, ego_car_location):
        A = point2[1] - point1[1]
        B = point1[0] - point2[0]
        C = (point2[0] * point1[1]) - (point1[0] * point2[1])

        distance = (A * ego_car_location[0] + B * ego_car_location[1] + C) / (np.sqrt(A ** 2 + B ** 2) + 0.1)

        return distance

    def find_closest_point(self, route_list, vehicle_position = None):
        if vehicle_position is None:
            ego_location = [self.ego.carlaActor.get_location().x, self.ego.carlaActor.get_location().y]
        else:
            ego_location = vehicle_position
        min_distance = float('inf')
        closest_index = -1
        for i, route_point in enumerate(route_list):
            distance = self.distance(ego_location, route_point)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        return closest_index

    def trace_waypoint(self, route_list):
        ego_location = [self.ego.carlaActor.get_location().x, self.ego.carlaActor.get_location().y]
        if self.tra_point_index == 0:
            self.tra_point_index += 1
        else:
            for i in range(self.tra_point_index, len(route_list) - 1):
                # the vector between the current waypoint and ego vehicle
                vector1 = [route_list[i][0] - ego_location[0], route_list[i][1] - ego_location[1]]
                # the vector between the following waypoint and cirrent waypoint
                vector2 = [route_list[i+1][0] - route_list[i][0], route_list[i+1][1] - route_list[i][1]]

                value = (vector1[0] * vector2[0] + vector1[1] * vector2[1]) / \
                        (math.sqrt(vector1[0] ** 2 + vector1[1] ** 2) *
                         math.sqrt(vector2[0] ** 2 + vector2[1] ** 2 ))
                sign = abs(value) / value

                # the angle bwteen the vector1 and vector2
                angle = math.acos(min(1.0, abs(value)) * sign)

                if angle < (math.pi/2):
                    # means this point is the next point that the vehicle need to go to
                    break
                else:
                    # this point already passed
                    self.tra_point_index += 1

    def collision_data(self, event):
        self.collision_history.append(event)

    ####################################################################################################################
    def createObjectInSimulator(self, obj):
        # Extract blueprint
        blueprint = self.blueprintLib.find(obj.blueprint)
        if obj.rolename is not None:
            blueprint.set_attribute('role_name', obj.rolename)

        # set walker as not invincible
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'False')

        # Set up transform

        loc = utils.scenicToCarlaLocation(obj.position, world=self.world, blueprint=obj.blueprint)
        rot = utils.scenicToCarlaRotation(obj.heading)
        transform = carla.Transform(loc, rot)

        # Create Carla actor
        carlaActor = self.world.try_spawn_actor(blueprint, transform)
        if carlaActor is None:
            self.destroy()
            raise SimulationCreationError(f'Unable to spawn object {obj}')
        obj.carlaActor = carlaActor

        carlaActor.set_simulate_physics(obj.physics)

        if isinstance(carlaActor, carla.Vehicle):
            carlaActor.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))
        elif isinstance(carlaActor, carla.Walker):
            carlaActor.apply_control(carla.WalkerControl())
            # spawn walker controller
            controller_bp = self.blueprintLib.find('controller.ai.walker')
            controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), carlaActor)
            if controller is None:
                self.destroy()
                raise SimulationCreationError(f'Unable to spawn carla controller for object {obj}')
            obj.carlaController = controller
        return carlaActor

    def executeActions(self, allActions):
        super().executeActions(allActions)

        # Apply control updates which were accumulated while executing the actions
        for obj in self.agents:
            if obj is self.agents[0]:
                continue
            else:
                ctrl = obj._control
                if ctrl is not None:
                    obj.carlaActor.apply_control(ctrl)
                    obj._control = None

    def step(self, action, last_position):
        # defination of different actions
        # if action == 0:
        #     self.ego.carlaActor.apply_control(carla.VehicleControl(throttle=1.0))
        # elif action == 1:
        #     self.ego.carlaActor.apply_control(carla.VehicleControl(throttle=0.5))
        # elif action == 2:
        #     self.ego.carlaActor.apply_control(carla.VehicleControl(throttle=0, brake=0))
        # elif action == 3:
        #     self.ego.carlaActor.apply_control(carla.VehicleControl(brake=0.5))
        # elif action == 4:
        #     self.ego.carlaActor.apply_control(carla.VehicleControl(brake=1.0))
        if action == 0:
            self.ego_steer -= 0.3
        elif action == 1:
            self.ego_steer -= 0.1
        elif action == 2:
            pass
        elif action == 3:
            self.ego_steer += 0.1
        elif action == 4:
           self.ego_steer += 0.3
        ################################################################################################################
        speed_controller = PIDLongitudinalController(vehicle=self.ego.carlaActor)
        acceleration = speed_controller.run_step(self.speed_limit)
        if acceleration >= 0.0:
            throttle = min(acceleration, 0.8)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(acceleration), 0.3)
        self.ego.carlaActor.apply_control(carla.VehicleControl(steer=self.ego_steer, throttle=throttle, brake=brake))
        # the env information
        ego_location = [self.ego.carlaActor.get_location().x, self.ego.carlaActor.get_location().y]
        ego_speed = [self.ego.carlaActor.get_velocity().x, self.ego.carlaActor.get_velocity().y]
        final_destination = self.driving_trajectory[-1]

        # here check the end situation
        if len(self.collision_history) != 0:
            print("collision")
            done = True
        # here check whether the vehicle reach the destination
        elif math.sqrt(
                (ego_location[0] - final_destination[0]) ** 2 + (ego_location[1] - final_destination[1]) ** 2) < 2:
            print("reach the destination")
            done = True
        # if the vehicle travel exceed a range
        elif math.sqrt((ego_location[0] - self.ego_spawn_point[0]) ** 2 + (
                ego_location[1] - self.ego_spawn_point[1]) ** 2) > 200:
            print(math.sqrt(
                (ego_location[0] - self.ego_spawn_point[0]) ** 2 + (ego_location[1] - self.ego_spawn_point[1]) ** 2))
            print("travel exceed range")
            done = True
        else:
            done = False
        rv = speed_reward(ego_speed, self.speed_limit, 5)
        rp = pathfollowing_reward(current_state=self.get_state(), current_route=self.trace_route(), ego_car_location=ego_location)
        reward = np.array([rp, rv])
        ############################################################
        # Run simulation for one timestep
        self.world.tick()
        self.trace_waypoint(self.driving_trajectory)

        new_state = self.get_state()
        # Render simulation
        spectator_transform = self.ego.carlaActor.get_transform()
        spectator_transform.location += carla.Location(x=-2, y=0, z=2.0)
        self.spectator.set_transform(spectator_transform)
        if self.render:
            # self.hud.tick(self.world, self.ego, self.displayClock)
            self.cameraManager.render(self.display)
            # self.hud.render(self.display)
            pygame.display.flip()
        return new_state, reward, done, len(self.collision_history)

    def getProperties(self, obj, properties):
        # Extract Carla properties
        carlaActor = obj.carlaActor
        currTransform = carlaActor.get_transform()
        currLoc = currTransform.location
        currRot = currTransform.rotation
        currVel = carlaActor.get_velocity()
        currAngVel = carlaActor.get_angular_velocity()

        # Prepare Scenic object properties
        velocity = utils.carlaToScenicPosition(currVel)
        speed = math.hypot(*velocity)

        values = dict(
            position=utils.carlaToScenicPosition(currLoc),
            elevation=utils.carlaToScenicElevation(currLoc),
            heading=utils.carlaToScenicHeading(currRot),
            velocity=velocity,
            speed=speed,
            angularSpeed=utils.carlaToScenicAngularSpeed(currAngVel),
        )
        return values

    def destroy(self):
        for obj in self.objects:
            if obj.carlaActor is not None:
                if isinstance(obj.carlaActor, carla.Vehicle):
                    obj.carlaActor.set_autopilot(False, self.tm.get_port())
                if isinstance(obj.carlaActor, carla.Walker):
                    obj.carlaController.stop()
                    obj.carlaController.destroy()
                obj.carlaActor.destroy()
        if self.render and self.cameraManager:
            self.cameraManager.destroy_sensor()
        # self.collision_sensor.destroy()

        self.client.stop_recorder()

        self.world.tick()
        super().destroy()
