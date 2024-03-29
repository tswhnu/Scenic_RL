""" Scenario Description
Traffic Scenario 10.
Crossing negotiation at an unsignalized intersection.
The ego-vehicle needs to negotiate with other vehicles to cross an unsignalized intersection. In
this situation it is assumed that the first to enter the intersection has priority.
"""

## SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
param map = localPath('../../../tests/formats/opendrive/maps/CARLA/Town05.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town05'
model scenic.simulators.carla.model

## CONSTANTS
EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 10
SAFETY_DISTANCE = 20
BRAKE_INTENSITY = 1.0

## DEFINING SPATIAL RELATIONS
# Please refer to scenic/domains/driving/roads.py how to access detailed road infrastructure
# 'network' is the 'class Network' object in roads.py

fourWayIntersection = filter(lambda i: i.is4Way and not i.isSignalized, network.intersections)

# make sure to put '*' to uniformly randomly select from all elements of the list
intersec = Uniform(*fourWayIntersection)
ego_start_lane = Uniform(*intersec.incomingLanes)

ego_maneuver = Uniform(*ego_start_lane.maneuvers)
ego_trajectory = [ego_maneuver.startLane, ego_maneuver.connectingLane, ego_maneuver.endLane]


## OBJECT PLACEMENT
ego_spawn_pt = OrientedPoint in ego_maneuver.startLane.centerline

ego = Car at ego_spawn_pt,
    with blueprint EGO_MODEL,
    with trajectory ego_trajectory,
    with rolename "ego_car"
