from Controller import Controller
from Drone import Drone
from TrajectoryGenerator import TrajectoryGenerator
from mpl_toolkits.mplot3d import Axes3D


def calculate_position(c, t):
    return c[0] * t ** 5 + c[1] * t ** 4 + c[2] * t ** 3 + c[3] * t ** 2 + c[4] * t + c[5]


def calculate_velocity(c, t):
    return 5 * c[0] * t ** 4 + 4 * c[1] * t ** 3 + 3 * c[2] * t ** 2 + 2 * c[3] * t + c[4]


def calculate_acceleration(c, t):
    return 20 * c[0] * t ** 3 + 12 * c[1] * t ** 2 + 6 * c[2] * t + 2 * c[3]


if __name__ == "__main__":

    # Define init state in inertial frame
    x = -5
    y = -5
    z = 5
    roll = 0
    pitch = 0
    yaw = 0

    # Define waypoints
    waypoints = [[-5, -5, 5], [-5, -5, 10], [-5, -5, 5], [-5, -5, 10]]

    # Create drone
    drone = Drone(x, y, z, roll, pitch, yaw)

    # Create controller
    controller = Controller(drone)

    # Init coefficients array
    x_coeffs = [[], [], [], []]
    y_coeffs = [[], [], [], []]
    z_coeffs = [[], [], [], []]

    # Interpolate trajectories between all waypoints
    T = 5
    for i in range(len(waypoints)):
        # Create traj object
        traj = TrajectoryGenerator(waypoints[i], waypoints[(i + 1) % len(waypoints)], T)

        # Solve traj coefficients
        traj.solve()

        # Save coefficients
        x_coeffs[i] = traj.x_c
        y_coeffs[i] = traj.y_c
        z_coeffs[i] = traj.z_c

    # Simulate
    i = 0
    t = 0
    while True:

        # Loop until waypoint reached
        while t <= T:

            # Get desired positions from traj coefficients
            des_x_pos = calculate_position(x_coeffs[i], t)
            des_y_pos = calculate_position(y_coeffs[i], t)
            des_z_pos = calculate_position(z_coeffs[i], t)

            # Get desired velocities from traj coefficients
            des_x_vel = calculate_velocity(x_coeffs[i], t)
            des_y_vel = calculate_velocity(y_coeffs[i], t)
            des_z_vel = calculate_velocity(z_coeffs[i], t)

            # Get desired acceleration from traj coefficients
            des_x_acc = calculate_acceleration(x_coeffs[i], t)
            des_y_acc = calculate_acceleration(y_coeffs[i], t)
            des_z_acc = calculate_acceleration(z_coeffs[i], t)

            # Compute control input
            u = controller.control(des_x_acc, des_y_acc, des_z_pos, des_z_vel, des_z_acc)

            # Step model
            drone.step(u)

            # Plot model
            drone.plot()

            # Proceed time
            t += drone.dt

        # Next waypoint
        t = 0
        i = (i + 1) % 4
