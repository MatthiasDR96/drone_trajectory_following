import numpy as np


class Controller:

    def __init__(self, model):

        # Model
        self.model = model

        # Proportional coefficients
        self.Kp_x = 1
        self.Kp_y = 1
        self.Kp_z = 1
        self.Kp_roll = 25
        self.Kp_pitch = 25
        self.Kp_yaw = 25

        # Derivative coefficients
        self.Kd_x = 10
        self.Kd_y = 10
        self.Kd_z = 10

        # Inverse kinematic matrix
        a = self.model.km / self.model.kf
        self.A = np.mat([[1.0, 1.0, 1.0, 1.0],
                           [0.0, self.model.l, 0.0, -self.model.l],
                           [-self.model.l, 0.0, self.model.l, 0.0],
                           [a, -a, a, -a]])

    def control(self, des_x_acc, des_y_acc, des_z_pos, des_z_vel, des_z_acc):

        # Feedforward plus feedback linearizing control
        acc_ff = des_z_acc
        acc_fb = self.Kp_z * (des_z_pos - self.model.z) + self.Kd_z * (des_z_vel - self.model.z_dot)
        thrust = float(max(0, self.model.m * (self.model.g + acc_ff + acc_fb)))

        # Proportional angular control
        roll_torque = float(self.Kp_roll * ((-des_y_acc / self.model.g) - self.model.roll))
        pitch_torque = float(self.Kp_pitch * ((des_x_acc / self.model.g) - self.model.pitch))
        yaw_torque = float(self.Kp_yaw * self.model.yaw)

        # Coefficients matrix
        B = np.mat([thrust, roll_torque, pitch_torque, yaw_torque]).T

        # Calculate needed rotor thrusts
        f = np.linalg.solve(self.A, B)

        # Calculate needed rotor velocities
        u = [thrust, roll_torque, pitch_torque, yaw_torque]
        # u = np.sqrt(f / self.model.kf)

        return u
