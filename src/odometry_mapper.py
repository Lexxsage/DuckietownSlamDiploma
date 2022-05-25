import matplotlib.pyplot as plt
import numpy as np
from odometry_adapter import OdometryAdapter as adapter

class OdometryMapper():

    def get_theta_value(self, theta):
        if theta > np.pi:
            theta = theta - 2*np.pi
        elif theta < theta - 2*np.pi:
            theta = theta + 2*np.pi
        else:
            theta = theta
        return theta

    def differential_drive(self, x_prev, y_prev, theta_prev,dt,vel_left,vel_right):
        l = 0.1
        translational_velocity = (vel_left+vel_right)/2
        rotational_velocity = (1/(2*l))*(vel_right-vel_left)

        x_k = x_prev + dt*np.cos(theta_prev)*translational_velocity
        y_k = y_prev + dt*np.sin(theta_prev)*translational_velocity
        theta_k = theta_prev + dt*rotational_velocity
        #theta_k = self.get_theta_value(theta_k)

        return x_k, y_k, theta_k


    def create_map_odometry(self):
        vel_left, vel_right, time_diff = adapter().load_odometry()
        x_new = [0]
        y_new = [0]
        theta_new = [0]
        sum_time = []

        for i in range(len(vel_left)-1):
            #throw in past state and current controls
            x_k, y_k, theta_k = self.differential_drive(x_new[i],y_new[i], theta_new[i],time_diff[i],vel_left[i],vel_right[i])
            x_new.append(x_k)
            y_new.append(y_k)
            theta_new.append(theta_k)

            sum_time.append(time_diff[i])

        print(sum_time)
        plt.plot(x_new, y_new)
        plt.show()

mapper = OdometryMapper()
mapper.create_map_odometry()