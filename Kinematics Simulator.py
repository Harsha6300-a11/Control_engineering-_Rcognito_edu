from controllers import N_CTRL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# Create the controller
controller = N_CTRL(
    k_rho = 9.0,
    k_alpha = 7.5,
    k_beta = -8.0,
    target_x=2.0,
    target_y=2.0,
    target_theta=0.0,
    sampling_time=0.1,
    dim_input=2,
    dim_output=3
)

# Simple simulation (start at origin)
x, y, th = 0.0, 0.0, 0.0

# Simulation loop
for t in np.arange(0, 10, 0.1):
    obs = np.array([x, y, th])
    action = controller.compute_action(t, obs)

    # Unpack velocities
    v, w = action

    # Simulate robot forward (Euler integration)
    th += w * 0.1
    x += v * np.cos(th) * 0.1
    y += v * np.sin(th) * 0.1

# Extract logs
obs = np.array(controller.obs_log)
act = np.array(controller.act_log)
time = np.array(controller.time_log)

# Plot trajectory
plt.figure()
plt.plot(obs[:, 0], obs[:, 1], label="Robot Path")
plt.plot(controller.target_x, controller.target_y, 'ro', label="Target")
plt.xlabel("X"); plt.ylabel("Y")
plt.title("Robot Trajectory")
plt.legend(); plt.grid()

# Plot tracking error
rho_err = np.linalg.norm(obs[:, :2] - np.array([controller.target_x, controller.target_y]), axis=1)
plt.figure()
plt.plot(time, rho_err)
plt.xlabel("Time"); plt.ylabel("Distance to Target")
plt.title("Tracking Error Over Time")
plt.grid()

# Plot control inputs
plt.figure()
plt.plot(time, act[:, 0], label="v (Linear Velocity)")
plt.plot(time, act[:, 1], label="w (Angular Velocity)")
plt.xlabel("Time"); plt.ylabel("Control Input")
plt.title("Control Inputs")
plt.legend(); plt.grid()

plt.show()
