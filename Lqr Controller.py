import numpy as np
from scipy.linalg import solve_discrete_are

class LQR:
    def __init__(self, A, B, Q, R, obs_target=[0.0, 0.0, 0.0], sampling_time=0.1):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.observation_target = np.array(obs_target)
        self.sampling_time = sampling_time
        self.ctrl_clock = 0.0
        self.action_curr = np.zeros(B.shape[1])

        self.obs_log = []
        self.act_log = []
        self.time_log = []

    def compute_gain(self):
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
        return K

    def compute_action(self, t, observation):
        if t >= self.ctrl_clock + self.sampling_time - 1e-6:
            self.ctrl_clock = t
            err = observation - self.observation_target
            K = self.compute_gain()
            u = -K @ err
            self.action_curr = np.clip(u, -1, 1)

            self.obs_log.append(observation.copy())
            self.act_log.append(self.action_curr.copy())
            self.time_log.append(t)

        return self.action_curr
