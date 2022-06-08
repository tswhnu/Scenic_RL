import math
import numpy as np
import cv2
def point_rotate(original_point, yaw):
    new_point = [original_point[0] * math.cos(yaw * math.pi / 180) - original_point[1] * math.sin(yaw * math.pi / 180),
                 original_point[0] * math.sin(yaw * math.pi / 180) + original_point[1] * math.cos(yaw * math.pi / 180)]
    return new_point
def coordinate_rotate(original_point, yaw):
    new_location = [original_point[0] * math.cos(yaw * math.pi / 180) + original_point[1] * math.sin(yaw * math.pi / 180),
                    -original_point[0] * math.sin(yaw * math.pi / 180) + original_point[1] * math.cos(yaw * math.pi / 180)]

    return new_location
def draw_route(image, route_list=None, agent_vehicle=None, image_range=None, resolution=0.1, way_point = False):
    vehicle_position = [agent_vehicle.get_location().x, agent_vehicle.get_location().y]
    # yaw angle of the ego vehicle
    yaw = agent_vehicle.get_transform().rotation.yaw
    # the upper left corner when there is no rotation
    # relative to the position of the ego vehicle
    upper_left_corner = [image_range[0] / 2, image_range[1] / 2]
    # after the rotation
    new_upper_left_corner = coordinate_rotate(point_rotate(upper_left_corner, yaw), yaw)
    if route_list is not None:
        pts_list = []
        for i in range(len(route_list)):
            if way_point:
                wp = route_list[i][0]
                wp_location = [wp.transform.location.x, wp.transform.location.y]
            else:
                wp_location = route_list[i]
            # the new location of waypoint after rotate the coordnate
            relative_wp_location = [wp_location[0] - vehicle_position[0], wp_location[1] - vehicle_position[1]]
            new_wp_location = coordinate_rotate(relative_wp_location, yaw)
            image_location = [new_wp_location[0] - new_upper_left_corner[0], new_wp_location[1] - new_upper_left_corner[1]]
            pixel_position = [abs(int(image_location[1] / resolution)), abs(int(image_location[0] / resolution))]
            pts_list.append(pixel_position)
            # rgb[pixel_position[0], pixel_position[1], :] = [0, 255, 255]
        pts_list = np.array(pts_list, np.int32)
        pts = pts_list.reshape((-1, 1, 2))
        image = cv2.polylines(image, [pts], False, (255, 0, 0), 3)
    return image
