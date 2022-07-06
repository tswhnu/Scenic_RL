"""Simulator interface for CARLA."""
import cv2
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
from bird_view.draw_map import *
from matplotlib import pyplot as plt
import numpy as np
from scenic.simulators.carla.utils.Reward_functions import *
from agents.navigation.controller import *
from bird_view.draw_routes import *
from scenic.simulators.carla.utils.generate_traffic import *
from scenic.simulators.carla.utils.HUD_render import *
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
        self.original_settings = self.world.get_settings()
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
    def __init__(self, scene, client, tm,
                 timestep, render, record,
                 scenario_number, verbosity=0,
                 speed_limit=25):
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
        self.reference_route = []
        self.driving_trajectory = []
        self.speed_list = []
        self.tra_point_index = 0
        self.ego_throttle = 0
        self.ego_brake = 0
        self.ego_steer = 0
        self.waypoint_path = []
        self.other_vehicles = []
        self.peds = []
        self.peds_ids = []
        self.hero_actor = None
        self.hero_transform = None
        self.traffic_light_surfaces = TrafficLightSurfaces()
        self.actors_with_transforms = None
        ##################
        # preparation of the HUD surface
        self.map_image = MapImage(carla_world=self.world,
                                  carla_map=self.map,
                                  pixels_per_meter=PIXELS_PER_METER,
                                  show_triggers=False,
                                  show_connections=False,
                                  show_spawn_points=False)
        self._hud = HUD('test', 1920, 1080)
        self.input_control = InputControl('test')

        self.original_surface_size = min(self._hud.dim[0], self._hud.dim[1])
        self.surface_size = self.map_image.big_map_surface.get_width()

        self.scaled_size = int(self.surface_size)
        self.prev_scaled_size = int(self.surface_size)

        # Render Actors
        self.actors_surface = pygame.Surface((self.map_image.surface.get_width(), self.map_image.surface.get_height()))
        self.actors_surface.set_colorkey(COLOR_BLACK)

        self.vehicle_id_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.vehicle_id_surface.set_colorkey(COLOR_BLACK)

        self.border_round_surface = pygame.Surface(self._hud.dim, pygame.SRCALPHA).convert()
        self.border_round_surface.set_colorkey(COLOR_WHITE)
        self.border_round_surface.fill(COLOR_BLACK)

        # Used for Hero Mode, draws the map contained in a circle with white border
        center_offset = (int(self._hud.dim[0] / 2), int(self._hud.dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, COLOR_ALUMINIUM_1, center_offset, int(self._hud.dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, COLOR_WHITE, center_offset, int((self._hud.dim[1] - 8) / 2))

        scaled_original_size = self.original_surface_size * (1.0 / 0.9)
        self.hero_surface = pygame.Surface((scaled_original_size, scaled_original_size)).convert()

        self.result_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.result_surface.set_colorkey(COLOR_BLACK)


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
            if obj is self.objects[0]:
                point1 = obj.trajectory[0].centerline.points[0]
                point2 = obj.trajectory[0].centerline.points[1]
                point1 = [point1[0], -point1[1]]
                point2 = [point2[0], -point2[1]]
                yaw = self.angle(point1=point1, point2=point2)
                spawn_point = point1
                carlaActor = self.createObjectInSimulator(obj, yaw, spawn_point)
            else:
                carlaActor = self.createObjectInSimulator(obj)

            # Check if ego (from carla_scenic_taks.py)
            if obj is self.objects[0]:
                self.ego = obj
                self.hero_actor = self.ego.carlaActor
                ##################################################################################
                route_planner = GlobalRoutePlanner(wmap=self.map, sampling_resolution=2.0)
                start_point = self.ego.trajectory[0].centerline.points[0]
                end_point = self.ego.trajectory[-1].centerline.points[-1]
                start_point = carla.Location(x=start_point[0], y=-start_point[1], z=0.5)
                end_point = carla.Location(x=end_point[0], y=-end_point[1], z=0.5)
                path = route_planner.trace_route(origin=start_point, destination=end_point)
                temp_path = []
                for i in range(len(path)):
                    self.waypoint_path.append(path[i][0])
                    location = [path[i][0].transform.location.x, path[i][0].transform.location.y]
                    temp_path.append(location)
                self.reference_route = np.array(temp_path)
                ###################################################################################
                # the starting and end point of the trajectory
                self.ego_spawn_point = [self.reference_route[0, 0], self.reference_route[0, 1]]
                self.spawn_yaw = self.angle(self.reference_route[0], self.reference_route[1])
                # find the cloest point in the trajectory list
                self.tra_point_index = self.find_closest_point(self.reference_route, self.ego_spawn_point)
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

        self.tick()  ## allowing manualgearshift to take effect 	# TODO still need this?

        for obj in self.objects:
            if isinstance(obj.carlaActor, carla.Vehicle):
                obj.carlaActor.apply_control(carla.VehicleControl(manual_gear_shift=False))

        self.tick()

        # Set Carla actor's initial speed (if specified)
        for obj in self.objects:
            if obj.speed is not None:
                equivVel = utils.scenicSpeedToCarlaVelocity(obj.speed, obj.heading)
                if hasattr(obj.carlaActor, 'set_target_velocity'):
                    obj.carlaActor.set_target_velocity(equivVel)
                else:
                    obj.carlaActor.set_velocity(equivVel)

    ########################################################################################################################
    def trace_route(self, point_num=3):
        # here giving the future three points based on the current trajectory and the vehicle position
        route = self.reference_route[(self.tra_point_index - 1):self.tra_point_index + point_num - 1]
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


    def tick(self):
        actors = self.world.get_actors()
        self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()
        self.world.tick()

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

        self.driving_trajectory.append(list(ego_location))
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
    # HUD rendering part
    def _split_actors(self):
        """Splits the retrieved actors by type id"""
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []

        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0]
            if 'vehicle' in actor.type_id:
                vehicles.append(actor_with_transform)
            elif 'traffic_light' in actor.type_id:
                traffic_lights.append(actor_with_transform)
            elif 'speed_limit' in actor.type_id:
                speed_limits.append(actor_with_transform)
            elif 'walker.pedestrian' in actor.type_id:
                walkers.append(actor_with_transform)

        return (vehicles, traffic_lights, speed_limits, walkers)

    def _render_traffic_lights(self, surface, list_tl, world_to_pixel):
        """Renders the traffic lights and shows its triggers and bounding boxes if flags are enabled"""
        self.affected_traffic_light = None

        for tl in list_tl:
            world_pos = tl.get_location()
            pos = world_to_pixel(world_pos)


            if self.hero_actor is not None:
                corners = Util.get_bounding_box(tl)
                corners = [world_to_pixel(p) for p in corners]
                tl_t = tl.get_transform()

                transformed_tv = tl_t.transform(tl.trigger_volume.location)
                hero_location = self.hero_actor.get_location()
                d = hero_location.distance(transformed_tv)
                s = Util.length(tl.trigger_volume.extent) + Util.length(self.hero_actor.bounding_box.extent)
                if (d <= s):
                    # Highlight traffic light
                    self.affected_traffic_light = tl
                    srf = self.traffic_light_surfaces.surfaces['h']
                    surface.blit(srf, srf.get_rect(center=pos))

            srf = self.traffic_light_surfaces.surfaces[tl.state]
            surface.blit(srf, srf.get_rect(center=pos))

    def _render_speed_limits(self, surface, list_sl, world_to_pixel, world_to_pixel_width):
        """Renders the speed limits by drawing two concentric circles (outer is red and inner white) and a speed limit text"""

        font_size = world_to_pixel_width(2)
        radius = world_to_pixel_width(2)
        font = pygame.font.SysFont('Arial', font_size)

        for sl in list_sl:

            x, y = world_to_pixel(sl.get_location())

            # Render speed limit concentric circles
            white_circle_radius = int(radius * 0.75)

            pygame.draw.circle(surface, COLOR_SCARLET_RED_1, (x, y), radius)
            pygame.draw.circle(surface, COLOR_ALUMINIUM_0, (x, y), white_circle_radius)

            limit = sl.type_id.split('.')[2]
            font_surface = font.render(limit, True, COLOR_ALUMINIUM_5)

            # Blit
            if self.hero_actor is not None:
                # In hero mode, Rotate font surface with respect to hero vehicle front
                angle = -self.hero_transform.rotation.yaw - 90.0
                font_surface = pygame.transform.rotate(font_surface, angle)
                offset = font_surface.get_rect(center=(x, y))
                surface.blit(font_surface, offset)

            else:
                # In map mode, there is no need to rotate the text of the speed limit
                surface.blit(font_surface, (x - radius / 2, y - radius / 2))

    def _render_walkers(self, surface, list_w, world_to_pixel):
        """Renders the walkers' bounding boxes"""
        for w in list_w:
            color = COLOR_PLUM_0

            # Compute bounding box points
            bb = w[0].bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y)]

            w[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)

    def _render_vehicles(self, surface, list_v, world_to_pixel):
        """Renders the vehicles' bounding boxes"""
        for v in list_v:
            color = COLOR_SKY_BLUE_0
            if int(v[0].attributes['number_of_wheels']) == 2:
                color = COLOR_CHOCOLATE_1
            if v[0].attributes['role_name'] == 'hero':
                color = COLOR_CHAMELEON_0
            # Compute bounding box points
            bb = v[0].bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y),
                       carla.Location(x=bb.x - 0.8, y=-bb.y),
                       carla.Location(x=bb.x, y=0),
                       carla.Location(x=bb.x - 0.8, y=bb.y),
                       carla.Location(x=-bb.x, y=bb.y),
                       carla.Location(x=-bb.x, y=-bb.y)
                       ]
            v[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.lines(surface, color, False, corners, int(math.ceil(4.0 * self.map_image.scale)))
    def _render_routes(self, surface, world_to_pixel):
        future_route = self.reference_route[self.tra_point_index:(self.tra_point_index+5)]
        route_carla = [carla.Location(x=point[0], y=point[1]) for point in future_route]
        route_carla = [world_to_pixel(p) for p in route_carla]
        whole_route = self.reference_route
        whole_route = [carla.Location(x=p[0], y=p[1]) for p in whole_route]
        whole_route = [world_to_pixel(p) for p in whole_route]
        try:
            pygame.draw.lines(surface, (0, 0, 255), False, whole_route, int(math.ceil(4.0 * self.map_image.scale)))
            pygame.draw.lines(surface, (255, 0, 0), False, route_carla, int(math.ceil(4.0 * self.map_image.scale)))
        except:
            pass
        finally:
            pass
    def render_actors(self, surface, vehicles, traffic_lights, speed_limits, walkers):
        """Renders all the actors"""
        # Static actors
        self._render_traffic_lights(surface, [tl[0] for tl in traffic_lights], self.map_image.world_to_pixel)
        self._render_speed_limits(surface, [sl[0] for sl in speed_limits], self.map_image.world_to_pixel,
                                  self.map_image.world_to_pixel_width)

        # Dynamic actors
        self._render_vehicles(surface, vehicles, self.map_image.world_to_pixel)
        self._render_walkers(surface, walkers, self.map_image.world_to_pixel)
        self._render_routes(surface, self.map_image.world_to_pixel)

    def clip_surfaces(self, clipping_rect):
        """Used to improve perfomance. Clips the surfaces in order to render only the part of the surfaces that are going to be visible"""
        self.actors_surface.set_clip(clipping_rect)
        self.vehicle_id_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)

    def _compute_scale(self, scale_factor):
        """Based on the mouse wheel and mouse position, it will compute the scale and move the map so that it is zoomed in or out based on mouse position"""
        m = self._input.mouse_pos

        # Percentage of surface where mouse position is actually
        px = (m[0] - self.scale_offset[0]) / float(self.prev_scaled_size)
        py = (m[1] - self.scale_offset[1]) / float(self.prev_scaled_size)

        # Offset will be the previously accumulated offset added with the
        # difference of mouse positions in the old and new scales
        diff_between_scales = ((float(self.prev_scaled_size) * px) - (float(self.scaled_size) * px),
                               (float(self.prev_scaled_size) * py) - (float(self.scaled_size) * py))

        self.scale_offset = (self.scale_offset[0] + diff_between_scales[0],
                             self.scale_offset[1] + diff_between_scales[1])

        # Update previous scale
        self.prev_scaled_size = self.scaled_size

        # Scale performed
        self.map_image.scale_map(scale_factor)

    def rendering(self, display):
        """Renders the map and all the actors in hero and map mode"""
        if self.actors_with_transforms is None:
            return
        self.result_surface.fill(COLOR_BLACK)

        # Split the actors by vehicle type id
        vehicles, traffic_lights, speed_limits, walkers = self._split_actors()

        # Zoom in and out
        scale_factor = 1.0

        # Render Actors
        self.actors_surface.fill(COLOR_BLACK)
        self.render_actors(
            self.actors_surface,
            vehicles,
            traffic_lights,
            speed_limits,
            walkers)

        # Render Ids
        self._hud.render_vehicles_ids(self.vehicle_id_surface, vehicles,
                                      self.map_image.world_to_pixel, self.hero_actor, self.hero_transform)

        # Blit surfaces
        surfaces = ((self.map_image.surface, (0, 0)),
                    (self.actors_surface, (0, 0)),
                    (self.vehicle_id_surface, (0, 0)),
                    )

        angle = 0.0 if self.hero_actor is None else self.hero_transform.rotation.yaw + 90.0
        self.traffic_light_surfaces.rotozoom(-angle, self.map_image.scale)

        center_offset = (0, 0)
        if self.hero_actor is not None:
            # Hero Mode
            hero_location_screen = self.map_image.world_to_pixel(self.hero_transform.location)
            hero_front = self.hero_transform.get_forward_vector()
            translation_offset = (hero_location_screen[0] - self.hero_surface.get_width() / 2 + hero_front.x * PIXELS_AHEAD_VEHICLE,
                                  (hero_location_screen[1] - self.hero_surface.get_height() / 2 + hero_front.y * PIXELS_AHEAD_VEHICLE))

            # Apply clipping rect
            clipping_rect = pygame.Rect(translation_offset[0],
                                        translation_offset[1],
                                        self.hero_surface.get_width(),
                                        self.hero_surface.get_height())
            self.clip_surfaces(clipping_rect)

            Util.blits(self.result_surface, surfaces)

            self.border_round_surface.set_clip(clipping_rect)

            self.hero_surface.fill(COLOR_ALUMINIUM_4)
            self.hero_surface.blit(self.result_surface, (-translation_offset[0],
                                                         -translation_offset[1]))

            rotated_result_surface = pygame.transform.rotozoom(self.hero_surface, angle, 0.9).convert()

            center = (display.get_width() / 2, display.get_height() / 2)
            rotation_pivot = rotated_result_surface.get_rect(center=center)
            display.blit(rotated_result_surface, rotation_pivot)

            display.blit(self.border_round_surface, (0, 0))
        else:
            # Map Mode
            # Translation offset
            translation_offset = (self._input.mouse_offset[0] * scale_factor + self.scale_offset[0],
                                  self._input.mouse_offset[1] * scale_factor + self.scale_offset[1])
            center_offset = (abs(display.get_width() - self.surface_size) / 2 * scale_factor, 0)

            # Apply clipping rect
            clipping_rect = pygame.Rect(-translation_offset[0] - center_offset[0], -translation_offset[1],
                                        self._hud.dim[0], self._hud.dim[1])
            self.clip_surfaces(clipping_rect)
            Util.blits(self.result_surface, surfaces)

            display.blit(self.result_surface, (translation_offset[0] + center_offset[0],
                                               translation_offset[1]))



    ####################################################################################################################
    def createObjectInSimulator(self, obj, yaw=None, spawn_point=None):
        # Extract blueprint
        blueprint = self.blueprintLib.find(obj.blueprint)
        if obj.rolename is not None:
            blueprint.set_attribute('role_name', obj.rolename)

        # set walker as not invincible
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'False')

        # Set up transform
        if spawn_point is not None:
            loc = carla.Location(x=spawn_point[0], y=spawn_point[1], z=0.5)
        else:
            loc = utils.scenicToCarlaLocation(obj.position, world=self.world, blueprint=obj.blueprint)
        if yaw is not None:
            rot = carla.Rotation(yaw=yaw)
        else:
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

    def step(self, actions, last_position):
        # defination of different actions
        # steer action
        if actions[0] == 0:
            self.ego_steer -= 0.1
        elif actions[0] == 1:
            self.ego_steer -= 0.05
        elif actions[0] == 2:
            pass
        elif actions[0] == 3:
            self.ego_steer += 0.05
        elif actions[0] == 4:
           self.ego_steer += 0.1

        if actions[1] == 0:
            throttle = 1.0
            brake = 0.0
        elif actions[1] == 1:
            throttle = 0.5
            brake = 0.0
        elif actions[1] == 2:
            throttle = 0.0
            brake = 0.0
        elif actions[1] == 3:
            throttle = 0.0
            brake = 0.5
        elif actions[1] == 4:
            throttle = 0.0
            brake = 1.0
        # limit the range of steer angle
        if self.ego_steer < 0:
            self.ego_steer = max(-0.8, self.ego_steer)
        else:
            self.ego_steer = min(0.8, self.ego_steer)
        ################################################################################################################
        # speed_controller = PIDLongitudinalController(vehicle=self.ego.carlaActor)
        # acceleration = speed_controller.run_step(self.speed_limit)
        # if acceleration >= 0.0:
        #     throttle = min(acceleration, 0.8)
        #     brake = 0.0
        # else:
        #     throttle = 0.0
        #     brake = min(abs(acceleration), 0.3)
        self.ego.carlaActor.apply_control(carla.VehicleControl(steer=self.ego_steer, throttle=throttle, brake=brake))
        # the env information
        ego_location = [self.ego.carlaActor.get_location().x, self.ego.carlaActor.get_location().y]
        ego_speed = [self.ego.carlaActor.get_velocity().x, self.ego.carlaActor.get_velocity().y]
        final_destination = self.reference_route[-1]

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
        self.tick()
        self.trace_waypoint(self.reference_route)

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
