#!/usr/bin/env python
import glob
import logging
import os
import sys
import cv2
from ideal_sensor import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np
import math
from queue import Queue
from queue import Empty

from DQN import *
from bird_view.lidar_birdeye import *
from bird_view.birdview_semantic import *


def sensor_callback(data, queue):
    queue.put(data)


class CarEnv(object):
    # attributes shared by the object created from same class
    show_cam = False
    im_width = 400
    im_height = 400
    steer_control = 1.0
    front_camera = None

    """docstring for CarEnv"""

    def __init__(self, sync_mode=True):

        self.depth_camera = None
        self.depth_queue = None
        self.manual_mode = False
        self.sync_mode = True
        self.synchronous_master = False
        self.spawn_npc = False

        self.seconds_per_epi = 10
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        self.traffic_manager = self.client.get_trafficmanager(8000)

        self.number_of_vehicles = 100
        self.num_of_pedestrian = 100

        # set the destination of the vehicle
        self.destination = [20.0, 28.4]
        self.blueprint_library = self.world.get_blueprint_library()
        self.ego_bp = self.blueprint_library.filter("model3")[0]
        self.ego_transform = carla.Transform(carla.Location(x=8.3, y=-49.6, z=0.5), carla.Rotation(yaw=270))
        self.ego_vehicle = None
        self.collision_sensor = None
        self.lidar_sensor = None
        self.depth_camera_bp = None
        self.episode_start = None

        # blueprint of lidar
        self.lidar_data = None
        self.lidar_height = 2.1
        self.lidar_trans = carla.Transform(carla.Location(x=-0.5, z=1.8))
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', str(64))
        self.lidar_bp.set_attribute('range', str(100))
        self.lidar_bp.set_attribute('points_per_second', str(100000))
        self.lidar_bp.set_attribute('dropoff_general_rate', str(0))

        # rgb_camera
        self.rgb_cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam_bp.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam_bp.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam_bp.set_attribute("fov", f"110")
        self.rgb_cam = None

        self.depth_camera_bp = self.blueprint_library.find("sensor.camera.depth")
        self.depth_camera_bp.set_attribute("image_size_x", f"{self.im_width}")
        self.depth_camera_bp.set_attribute("image_size_y", f"{self.im_height}")
        self.depth_camera_bp.set_attribute("fov", f"110")

        # the list that store the information
        self.collision_history = None
        self.vehicle_list = None
        self.sensor_list = None
        self.walkers_list = None
        self.all_id = None
        self.all_ped = None

        # the queue used to save the data from lidar
        self.lidar_queue = None
        self.image_queue = None
        self.position_hist = None

    def reset(self):
        self.synchronous_master = False

        # initialize the world
        self.collision_history = []
        self.all_id = []
        self.vehicle_list = []
        self.sensor_list = []
        self.walkers_list = []
        self.lidar_queue = Queue()
        self.image_queue = Queue()
        self.depth_queue = Queue()
        self.position_hist = []
        return np.random.random(5)

    def collision_data(self, event):
        self.collision_history.append(event)

    def test_env(self):

        return np.random.random(5)

    def step(self, action):

        reward = [1, 1]
        state = np.random.random()
        done = False
        return state, reward, done, None
