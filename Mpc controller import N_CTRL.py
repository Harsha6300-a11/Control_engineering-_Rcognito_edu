import numpy as np
import cvxpy as cp

class MPC:
    def __init__(self, A, B, Q, R, N=10, obs_target=[0, 0, 0], sampling_time=0.1):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = N
        self.obs_target = np.array(obs_target)
        self.sampling_time = sampling_time

        self.ctrl_clock = 0.0
        self.action_curr = np.zeros(B.shape[1])
        self.obs_log = []
        self.act_log = []
        self.time_log = []

    def compute_action(self, t, observation):
        if t < self.ctrl_clock + self.sampling_time - 1e-6:
            return self.action_curr

        self.ctrl_clock = t
        x0 = observation - self.obs_target

        # Define optimization variables
        x = cp.Variable((self.A.shape[0], self.N + 1))
        u = cp.Variable((self.B.shape[1], self.N))

        # Define cost
        cost = 0
        constr = [x[:, 0] == x0]
        for k in range(self.N):
            cost += cp.quad_form(x[:, k], self.Q) + cp.quad_form(u[:, k], self.R)
            constr += [x[:, k+1] == self.A @ x[:, k] + self.B @ u[:, k]]
            constr += [cp.norm(u[:, k], 'inf') <= 1]  # bounded inputs

        cost += cp.quad_form(x[:, self.N], self.Q)  # terminal cost

        # Solve
        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve()

        if u.value is not None:
            self.action_curr = np.clip(u.value[:, 0], -1, 1)
        else:
            self.action_curr = np.zeros(self.B.shape[1])  # fallback

        self.obs_log.append(observation.copy())
        self.act_log.append(self.action_curr.copy())
        self.time_log.append(t)

        return self.action_curr
