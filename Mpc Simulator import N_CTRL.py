import numpy as np
import matplotlib.pyplot as plt
from controllers import MPC

dt = 0.1
A = np.eye(3)
B = np.array([
    [dt, 0],
    [0, dt],
    [0, dt]
])

Q = np.diag([10.0, 10.0, 9])
R = np.diag([1.0, 1.0])

target = [2.0, 2.0, 0.0]
controller = MPC(A, B, Q, R, N=15, obs_target=target, sampling_time=dt)

x, y, th = 0.0, 0.0, 0.0
trajectory = []

for t in np.arange(0, 10, dt):
    obs = np.array([x, y, th])
    action = controller.compute_action(t, obs)
    v, w = action

    th += w * dt
    x += v * np.cos(th) * dt
    y += v * np.sin(th) * dt
    trajectory.append([x, y, th])

trajectory = np.array(trajectory)
obs_log = np.array(controller.obs_log)
act_log = np.array(controller.act_log)
time_log = np.array(controller.time_log)

# Plotting
plt.figure()
plt.plot(obs_log[:, 0], obs_log[:, 1], label="MPC Path")
plt.plot(target[0], target[1], 'ro', label="Target")
plt.xlabel("X"); plt.ylabel("Y")
plt.title("MPC Trajectory")
plt.legend(); plt.grid()

plt.figure()
error = np.linalg.norm(obs_log[:, :2] - target[:2], axis=1)
plt.plot(time_log, error)
plt.title("Tracking Error Over Time")
plt.xlabel("Time"); plt.ylabel("Distance to Target")
plt.grid()

plt.figure()
plt.plot(time_log, act_log[:, 0], label="v (Linear)")
plt.plot(time_log, act_log[:, 1], label="w (Angular)")
plt.title("Control Inputs")
plt.xlabel("Time"); plt.ylabel("Control Input")
plt.legend(); plt.grid()

plt.show()