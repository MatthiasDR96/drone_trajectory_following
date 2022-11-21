from math import sin, cos

import matplotlib.pyplot as plt
import numpy as np


class Drone:

    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, size=1):

        # Params
        self.kf = 300000
        self.km = 3000
        self.m = 0.2
        self.I = 1
        self.l = size / 2
        self.g = 9.81
        self.g_vec = np.mat([0, 0, -self.g]).T

        # Positions of rotors in base frame (homogeneous coordinates)
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([0, size / 2, 0, 1]).T
        self.p3 = np.array([-size / 2, 0, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T

        # Init state
        self.x = x
        self.y = y
        self.z = z
        self.x_dot = 0
        self.y_dot = 0
        self.z_dot = 0
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.roll_dot = 0
        self.pitch_dot = 0
        self.yaw_dot = 0

        # Simulation params
        self.dt = 0.1

        # Init trajectory arrays
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.x_data.append(self.x)
        self.y_data.append(self.y)
        self.z_data.append(self.z)

        # Init canvas and plot first state
        plt.ion()
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        self.plot()

    def dynamics(self, u):

        # Calculate rotor thrusts
        f = self.kf * np.power(u, 2)

        # Calculate rotor resistance torques
        mu = self.km * np.power(u, 2)

        # Calculate force vector in body frame
        f_b = np.mat([0, 0, u[0]]).T
        # f_b = np.mat([0, 0, np.sum(f)]).T

        # Calculate rotation matrix from inertial to body frame
        rot_i_b = self.rotation_matrix()

        # Calculate translational acceleration in body frame
        acc_t = (self.m * self.g_vec + rot_i_b * f_b) / self.m

        # Calculate torque vector in body frame
        # tau_b = np.mat([self.l * (f[1, 0] - f[3, 0]), self.l * (f[2, 0] - f[0, 0]), mu[0, 0] - mu[1, 0] + mu[2, 0] - mu[3, 0]]).T

        # Calculate rotational acceleration in B-frame
        acc_r = np.mat([u[1], u[2], u[3]]).T / self.I
        # acc_r = tau_b / self.I

        # Pack total state derivative vector
        x_dot = np.vstack((acc_t, acc_r))

        return x_dot

    def rotation_matrix(self):

        # Get state
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw

        # Calculate ZYX rotation matrix
        zxy_matrix = np.mat(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll),
              sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll)],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) *
              sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll)],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw)]
             ])

        return zxy_matrix

    def step(self, u):
        # Dynamics
        acc = self.dynamics(u)

        # Get angular accelerations in inertial frame
        roll_acc = acc[3, 0]
        pitch_acc = acc[4, 0]
        yaw_acc = acc[5, 0]

        # Update angular velocities via Euler iteration
        self.roll_dot += roll_acc * self.dt
        self.pitch_dot += pitch_acc * self.dt
        self.yaw_dot += yaw_acc * self.dt

        # Update angular positions via Euler iteration
        self.roll += self.roll_dot * self.dt
        self.pitch += self.pitch_dot * self.dt
        self.yaw += self.yaw_dot * self.dt

        # Get linear accelerations in inertial frame
        x_acc = acc[0, 0]
        y_acc = acc[1, 0]
        z_acc = acc[2, 0]

        # Update linear velocities via Euler iteration
        self.x_dot += x_acc * self.dt
        self.y_dot += y_acc * self.dt
        self.z_dot += z_acc * self.dt

        # Update linear positions via Euler iteration
        self.x += self.x_dot * self.dt
        self.y += self.y_dot * self.dt
        self.z += self.z_dot * self.dt

        # Add previous states
        self.x_data.append(self.x)
        self.y_data.append(self.y)
        self.z_data.append(self.z)

    def plot(self):
        # Calculate transformation matrix from inertial frame to base frame
        T = self.transformation_matrix()

        # Calculate positions of rotors in inertial frame
        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        # Clear current axis
        plt.cla()

        # Plot rotors
        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.')

        # Plot skeleton
        self.ax.plot([p1_t[0], p3_t[0]], [p1_t[1], p3_t[1]], [p1_t[2], p3_t[2]], 'r-')
        self.ax.plot([p2_t[0], p4_t[0]], [p2_t[1], p4_t[1]], [p2_t[2], p4_t[2]], 'r-')

        # Plot traveled trajectory
        self.ax.plot(self.x_data, self.y_data, self.z_data, 'b:')

        # Plot params
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        self.ax.set_zlim(0, 10)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        plt.pause(0.001)

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll),
              sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw), z]])
