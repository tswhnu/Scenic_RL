import math
import random

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


# def world_to_pixel(self, location, offset=(0, 0)):
#     """Converts the world coordinates to pixel coordinates"""
#     x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
#     y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
#     return [int(x - offset[0]), int(y - offset[1])]
# a = Image.open('cache/no_rendering_mode/Town05_2e1d4af27b41d85ff578b77d9ea0fd91f45abd81.tga')


def world_to_pixel(location, world_offset=(-326.0445251464844, -257.8750915527344), offset=(0,0)):
    """Converts the world coordinates to pixel coordinates"""
    x = 10 * (location[0] - world_offset[0])
    y = 10 * (location[1] - world_offset[1])
    return [int(x - offset[0]), int(y - offset[1])]

def draw_route(route_list, image = None, color=(255, 0, 0)):
    pts_list = []
    for i in range(len(route_list)):
        pixel_position = world_to_pixel(route_list[i])
        pts_list.append(pixel_position)
    pts_list = np.array(pts_list, np.int32)
    pts = pts_list.reshape((-1, 1, 2))
    image = cv2.polylines(image, [pts], False, color, 3)
    return image

def draw_trace(reference_route, driving_trajectory, original_map, bound_width = 5):

    max_x = np.max(reference_route[:,0]) + bound_width
    min_x = np.min(reference_route[:,0]) - bound_width
    max_y = np.max(reference_route[:,1]) + bound_width
    min_y = np.min(reference_route[:, 1]) - bound_width
    pts_min = world_to_pixel([min_x, min_y])
    pts_max = world_to_pixel([max_x, max_y])
    original_map = np.array(original_map) / 255.0
    map_with_route = draw_route(reference_route, original_map, color=(0, 255, 0))
    map_with_trajectory = draw_route(driving_trajectory, map_with_route, color= (255, 165, 0))
    map_with_trajectory = map_with_trajectory[pts_min[1]:pts_max[1], pts_min[0]:pts_max[0], :]
    plt.imshow(map_with_trajectory)
    plt.show()

# num = random.randint(2000, 2100)
# map = np.load('Town05.npy')
# refer_route = np.load('../examples/carla_scenic/Carla_Challenge/log_01/reference_route'+ str(num) + '.npy')
# driving_route = np.load('../examples/carla_scenic/Carla_Challenge/log_01/vehicle_trajectory'+ str(num) + '.npy')
# draw_trace(refer_route, driving_route, map)
# def point_rotate(original_point, yaw):
#     new_point = [original_point[0] * math.cos(yaw * math.pi / 180) - original_point[1] * math.sin(yaw * math.pi / 180),
#                  original_point[0] * math.sin(yaw * math.pi / 180) + original_point[1] * math.cos(yaw * math.pi / 180)]
#     return new_point
# def coordinate_rotate(original_point, yaw):
#     new_location = [original_point[0] * math.cos(yaw * math.pi / 180) + original_point[1] * math.sin(yaw * math.pi / 180),
#                     -original_point[0] * math.sin(yaw * math.pi / 180) + original_point[1] * math.cos(yaw * math.pi / 180)]
#
#     return new_location
# def draw_route(image, route_list=None, agent_vehicle=None, image_range=None, resolution=0.1, way_point = False):
#     vehicle_position = [agent_vehicle.get_location().x, agent_vehicle.get_location().y]
#     # yaw angle of the ego vehicle
#     angle = agent_vehicle.get_transform().rotation.yaw
#     # the upper left corner when there is no rotation
#     # relative to the position of the ego vehicle
#     upper_left_corner = [image_range[0] / 2, image_range[1] / 2]
#     # after the rotation
#     new_upper_left_corner = coordinate_rotate(point_rotate(upper_left_corner, angle), angle)
#     if route_list is not None:
#         pts_list = []
#         for i in range(len(route_list)):
#             if way_point:
#                 wp = route_list[i][0]
#                 wp_locati
#                 on = [wp.transform.location.x, wp.transform.location.y]
#             else:
#                 wp_location = route_list[i]
#             # the new location of waypoint after rotate the coordnate
#             relative_wp_location = [wp_location[0] - vehicle_position[0], wp_location[1] - vehicle_position[1]]
#             new_wp_location = coordinate_rotate(relative_wp_location, angle)
#             image_location = [new_wp_location[0] - new_upper_left_corner[0], new_wp_location[1] - new_upper_left_corner[1]]
#             pixel_position = [abs(int(image_location[1] / resolution)), abs(int(image_location[0] / resolution))]
#             pts_list.append(pixel_position)
#             # rgb[pixel_position[0], pixel_position[1], :] = [0, 255, 255]
#         pts_list = np.array(pts_list, np.int32)
#         pts = pts_list.reshape((-1, 1, 2))
#         image = cv2.polylines(image, [pts], False, (255, 0, 0), 3)
#     return image


