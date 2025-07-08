import numpy as np
import math

class N_CTRL:
    def __init__(self, k_rho=1.0, k_alpha=2.0, k_beta=-1.5,
                 target_x=0.0, target_y=0.0, target_theta=0.0,
                 sampling_time=0.01, dim_input=2, dim_output=3):
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.k_beta = k_beta
        self.dt = sampling_time

        self.target_x = target_x
        self.target_y = target_y
        self.target_theta = target_theta

        self.ctrl_clock = 0.0
        self.sampling_time = sampling_time
        self.action_curr = np.array([0.0, 0.0])
        self.accum_obj_val = 0.0
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.state_sys = np.zeros(dim_output)

        # New logs for plotting
        self.obs_log = []
        self.act_log = []
        self.time_log = []

    def wrap_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle <= -math.pi:
            angle += 2 * math.pi
        return angle

    def compute_action(self, t, observation):
        time_in_sample = t - self.ctrl_clock

        if t >= self.ctrl_clock + self.sampling_time - 1e-6:
            self.ctrl_clock = t

            x, y, th = observation
            x_f = self.target_x
            y_f = self.target_y
            th_f = self.target_theta

            dx = x_f - x
            dy = y_f - y
            rho = math.sqrt(dx*2 + dy*2)

            alpha = self.wrap_angle(-th + math.atan2(dy, dx))
            beta = self.wrap_angle((th_f - th) - alpha)

            pos_tolerance = 0.05
            angle_tolerance = 0.05

            if rho < pos_tolerance and abs(self.wrap_angle(th_f - th)) < angle_tolerance:
                v = 0.0
                w = 0.0
            else:
                v = self.k_rho * rho
                w = self.k_alpha * alpha + self.k_beta * beta

            v = np.clip(v, -1, 1)
            w = np.clip(w, -1.0, 1.0)

            self.action_curr = np.array([v, w])

            # Log data
            self.obs_log.append(observation)
            self.act_log.append(self.action_curr)
            self.time_log.append(t)

            return self.action_curr
        else:
            return self.action_curr

    def reset(self, t0):
        self.ctrl_clock = t0
        self.action_curr = np.array([0.0, 0.0])
        self.accum_obj_val = 0.0

        self.obs_log = []
        self.act_log = []
        self.time_log = []

    def receive_sys_state(self, state):
        self.state_sys = state

    def upd_accum_obj(self, observation, action):
        self.accum_obj_val += self.run_obj(observation, action) * self.sampling_time

    def run_obj(self, observation, action):
        x, y, th = observation
        x_f = self.target_x
        y_f = self.target_y
        th_f = self.target_theta

        pos_cost = (x - x_f)*2 + (y - y_f)*2
        orientation_cost = self.wrap_angle(th - th_f)**2

        v, w = action
        control_effort_cost = 0.1 * (v*2 + w*2)

        total_cost = pos_cost * 10 + orientation_cost * 5 + control_effort_cost
        return total_cost
