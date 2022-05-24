#!/usr/bin/env python
import glob
import logging
import os
import sys
import cv2

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
        return np.random.random(10).astype(np.single)

    def collision_data(self, event):
        self.collision_history.append(event)

    def test_env(self):

        return np.random.random(10)

    def step(self, action):

        reward = [1, 1]
        state = np.random.random(10).astype(np.single)
        done = False
        return state, reward, done, None

    def terminal(self):
        a = 1+1
        return a
