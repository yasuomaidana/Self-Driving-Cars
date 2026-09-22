"""
Session 01 Solution: Kinematic Bicycle Model & Constraint Verification
"""

import math
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class KinematicBicycleModel:
    def __init__(self, wheelbase: float = 2.8, max_steer_rad: float = 0.6, max_accel: float = 3.0, max_lat_accel: float = 3.5):
        self.L = wheelbase
        self.max_steer = max_steer_rad
        self.max_accel = max_accel
        self.max_lat_accel = max_lat_accel

    def step(self, state: dict, delta: float, a: float, dt: float) -> dict:
        delta_clamped = max(-self.max_steer, min(self.max_steer, delta))
        a_clamped = max(-self.max_accel, min(self.max_accel, a))
        
        # Continuous differential model integrated with Forward Euler
        x = state['x']
        y = state['y']
        theta = state['theta']
        v = state['v']
        
        x_next = x + v * math.cos(theta) * dt
        y_next = y + v * math.sin(theta) * dt
        theta_next = theta + (v / self.L) * math.tan(delta_clamped) * dt
        v_next = max(0.0, v + a_clamped * dt)
        
        kappa = math.tan(delta_clamped) / self.L
        a_lat = (v ** 2) * kappa
        
        return {
            'x': float(x_next),
            'y': float(y_next),
            'theta': float(theta_next),
            'v': float(v_next),
            'delta': float(delta_clamped),
            'a': float(a_clamped),
            'kappa': float(kappa),
            'a_lat': float(a_lat)
        }


def simulate_maneuver(model: KinematicBicycleModel, control_profile: list, dt: float = 0.05) -> pl.DataFrame:
    history = []
    current_state = {'x': 0.0, 'y': 0.0, 'theta': 0.0, 'v': 5.0}
    time = 0.0
    for delta, a in control_profile:
        step_result = model.step(current_state, delta, a, dt)
        step_result['time'] = round(time, 3)
        history.append(step_result)
        current_state = {
            'x': step_result['x'],
            'y': step_result['y'],
            'theta': step_result['theta'],
            'v': step_result['v']
        }
        time += dt
        
    return pl.DataFrame(history)


if __name__ == "__main__":
    model = KinematicBicycleModel(wheelbase=2.8, max_steer_rad=0.5, max_lat_accel=3.0)
    steps = 200
    controls = []
    for i in range(steps):
        t = i * 0.05
        if t < 2.0:
            delta = 0.0
            a = 1.0
        elif t < 6.0:
            delta = 0.3 * math.sin(math.pi * (t - 2.0) / 2.0)
            a = 0.2
        else:
            delta = 0.0
            a = -0.5
        controls.append((delta, a))
        
    df = simulate_maneuver(model, controls, dt=0.05)
    print("Simulation completed successfully. DataFrame shape:", df.shape)
    print(df.head(5))
