"""Simulator interface for CARLA."""
import numpy as np

try:
	import carla
except ImportError as e:
	raise ModuleNotFoundError('CARLA scenarios require the "carla" Python package') from e

import math
import os

from scenic.syntax.translator import verbosity
if verbosity == 0:	# suppress pygame advertisement at zero verbosity
	import os
	os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame
##########
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from bird_view.birdview_semantic import *
from matplotlib import pyplot as plt
import numpy as np
#########
from scenic.domains.driving.simulators import DrivingSimulator, DrivingSimulation
from scenic.core.simulators import SimulationCreationError
from scenic.syntax.veneer import verbosePrint
import scenic.simulators.carla.utils.utils as utils
import scenic.simulators.carla.utils.visuals as visuals


class CarlaSimulator(DrivingSimulator):
	"""Implementation of `Simulator` for CARLA."""
	def __init__(self, carla_map, map_path, address='127.0.0.1', port=2000, timeout=10,
				 render=True, record='', timestep=0.1):
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
	def __init__(self, scene, client, tm, timestep, render, record, scenario_number, verbosity=0):
		super().__init__(scene, timestep=timestep, verbosity=verbosity)
		self.client = client
		self.world = self.client.get_world()
		self.map = self.world.get_map()
		self.blueprintLib = self.world.get_blueprint_library()
		self.tm = tm

		##################
		self.collision_history = []
		self.speed_limit = 60 # km/h
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
			name = "{}/scenario{}.log".format(self.record, self.scenario_number)
			self.client.start_recorder(name)

		# Create Carla actors corresponding to Scenic objects
		self.ego = None
		for obj in self.objects:
			carlaActor = self.createObjectInSimulator(obj)

			# Check if ego (from carla_scenic_taks.py)
			if obj is self.objects[0]:
				self.ego = obj
###########################################################
				# add routeplanner for vehicles in carla
				self.route_planner = GlobalRoutePlanner(self.map, sampling_resolution=2.0)

				# add collision sensor to the ego vehicle
				# osition of the collision sensor
				# transform = carla.Transform(carla.Location(x=2.5, z=0.7))
				# collision_sensor = self.blueprintLib.find("sensor.other.collision")
				# self.collision_sensor = self.world.spawn_actor(collision_sensor, transform, attach_to=carlaActor)
				# self.collision_sensor.listen(lambda event: self.collision_data(event))

				# Set up camera manager and collision sensor for ego
				if self.render:
					camIndex = 0
					camPosIndex = 0
					self.cameraManager = visuals.CameraManager(self.world, carlaActor, self.hud)
					self.cameraManager._transform_index = camPosIndex
					self.cameraManager.set_sensor(camIndex)
					self.cameraManager.set_transform(self.camTransform)

		self.world.tick() ## allowing manualgearshift to take effect 	# TODO still need this?

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
	def trace_route(self, waypoint_mode = False, points_num=3):
		# here the destination is the last point of the endline
		destination = self.ego.trajectory[2].centerline.points[-1]
		destination = carla.Location(x=destination[0], y=destination[1], z=0.5)
		current_location = utils.scenicToCarlaLocation(self.ego.position, world=self.world, blueprint=self.ego.blueprint)
		trace = self.route_planner.trace_route(current_location, destination)
		# here we pickup first n points in the waypoints list
		trace = trace[:points_num]
		if not waypoint_mode:
			return [[i[0].transform.location.x, i[0].transform.location.y] for i in trace]
		else:
			return trace

	def speed_reward_test(self,ego_speed, speed_limit):
		vehicle_vel = math.sqrt(ego_speed[0] ** 2 + ego_speed[1] ** 2)
		current_speed = 3.6 * vehicle_vel
		# if have collision risk, the vehicle speed need to be smaller than speed limit
		if speed_limit == 0:
			if current_speed == 0.0:
				reward = 0
			else:
				reward = -100
		# elif speed_limit == 60:
		#     reward = - ((current_speed - speed_limit) ** 2 - 25) * 0.05

		elif current_speed > speed_limit:
			reward = - (current_speed - speed_limit) ** 2 * 0.05
		else:
			reward = current_speed / (speed_limit + 0.1)
		# reward = 0

		return reward

	def destination_reward_test(self, destination, ego_location):
		return math.sqrt(
            (ego_location[0] - destination[0]) ** 2 + (ego_location[1] - destination[1]) ** 2)

	def get_state(self):
		trace = np.array(self.trace_route(waypoint_mode=False)).reshape(-1)
		ego_location = np.array([self.ego.carlaActor.get_location().x, self.ego.carlaActor.get_location().y])
		ego_speed = np.array([self.ego.carlaActor.get_velocity().x, self.ego.carlaActor.get_velocity().y])
		return np.append(trace, [ego_location, ego_speed]).astype(np.single)

	def collision_data(self, event):
		self.collision_history.append(event)
########################################################################################################################
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
	def step(self, action):
		# defination of different actions
		if action == 0:
			self.ego.carlaActor.apply_control(carla.VehicleControl(throttle=0.0, steer=-1.0))
		elif action == 1:
			self.ego.carlaActor.apply_control(carla.VehicleControl(throttle=0.0, steer=1.0))
		elif action == 2:
			self.ego.carlaActor.apply_control(carla.VehicleControl(throttle=1.0))
		elif action == 3:
			self.ego.carlaActor.apply_control(carla.VehicleControl(throttle=0.0))
		elif action == 4:
			self.ego.carlaActor.apply_control(carla.VehicleControl(brake=1.0))

		# the env information
		trace = self.trace_route(waypoint_mode=False)
		ego_location = [self.ego.carlaActor.get_location().x, self.ego.carlaActor.get_location().y]
		ego_speed = [self.ego.carlaActor.get_velocity().x, self.ego.carlaActor.get_velocity().y]
		destination = trace[-1]
		if len(self.collision_history) != 0:
			print("collision")
			print(self.collision_history[0])
			done = True
			print("last_action:", action)
			rc = -1000
		else:
			done = False
			rc = 0
			rv = self.speed_reward_test(ego_speed, self.speed_limit)
			rp = self.destination_reward_test(destination, ego_location)
		reward = [rc, rv, rp]
############################################################
		# Run simulation for one timestep
		self.world.tick()
		new_state = self.get_state()
		# Render simulation
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
