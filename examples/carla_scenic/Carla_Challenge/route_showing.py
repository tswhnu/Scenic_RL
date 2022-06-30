import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
episodes = random.randint(2000, 2420)

# def func(x, a, b, c):
#     return a + b * x + c * x ** 2
#
reference_route = np.load('./log_01/reference_route' + str(episodes) + '.npy')
print(reference_route.shape)
driving_trajectory = np.load('./log_01/vehicle_trajectory'+str(episodes) +'.npy')
# vehicle_speed = np.load('./log_01/vehicle_speed' + str(episodes) + '.npy')
# para2 = np.polyfit(current_route[:, 0], current_route[:, 1], 3 )
# p = np.poly1d(para2)
# x = np.linspace(start=current_route[0, 0], stop=current_route[-1, 0], num=100)
# y = np.array([p(i) for i in x])

# plt.plot(reference_route[:,0], reference_route[:,1])
# plt.plot(driving_trajectory[:,0], driving_trajectory[:,1])
plt.plot(reference_route[:,0], reference_route[:,1])
plt.title(r'$\alpha_i > \beta_i$', fontsize=20)
plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
plt.show()
# plt.plot(np.array([i for i in range(len(vehicle_speed))]), vehicle_speed)
plt.show()
# for i in range(2):
#     current_route = reference_route[i : i + 4]
#     plt.scatter(current_route[:, 0], current_route[:, 1])
#     if len(current_route) == 2:
#         break
#     para2 = np.polyfit(current_route[:, 0], current_route[:, 1], 2 )
#
#     x = np.linspace(start=current_route[0, 0], stop=current_route[-1, 0], num=100)
#     y = np.array([p(i) for i in x])
#     plt.plot(x, y)
#     plt.show()


# vehicle_speed = np.load('./log_01/vehicle_speed' + str(episodes) + '.npy')
# reward_curve = np.load('./log_01/reward_list561.npy')
#

# plt.plot(np.array([i for i in range(len(vehicle_speed))]), vehicle_speed)
# plt.plot(np.array([i for i in range(len(reward_curve[:,0]))]), reward_curve[:,0])