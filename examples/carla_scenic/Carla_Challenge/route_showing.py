import numpy as np
import matplotlib.pyplot as plt
import random
episodes = random.randint(3000, 3300)

reference_route = np.load('./log/reference_route' + str(episodes) + '.npy')
driving_trajectory = np.load('./log/vehicle_trajectory'+str(episodes) +'.npy')
vehicle_speed = np.load('./log/vehicle_speed' + str(episodes) + '.npy')
reward_curve = np.load('./log/reward_list561.npy')
plt.plot(driving_trajectory[:,0], driving_trajectory[:,1])
plt.plot(reference_route[:,0], reference_route[:,1])
# plt.plot(np.array([i for i in range(len(vehicle_speed))]), vehicle_speed)
# plt.plot(np.array([i for i in range(len(reward_curve[:,0]))]), reward_curve[:,0])
plt.xlabel('globalx [m]')
plt.ylabel('globaly [m]')
plt.legend(['ref', 'driving'])
plt.show()