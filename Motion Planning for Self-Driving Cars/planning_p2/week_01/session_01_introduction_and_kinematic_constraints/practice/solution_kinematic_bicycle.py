"""
Session 01 Solution: Kinematic Bicycle Model & Constraint Verification
----------------------------------------------------------------------
Full instructor reference implementation.
"""

import math
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class KinematicBicycleModel:
    def __init__(self, wheelbase: float = 2.8, max_steer_rad: float = 0.5, max_accel: float = 3.0, max_lat_accel: float = 3.0):
        """
        Parameters:
            wheelbase: Distance between front and rear axle (L) in meters
            max_steer_rad: Mechanical limit on steering angle (delta_max) in radians
            max_accel: Max longitudinal acceleration in m/s^2
            max_lat_accel: Max lateral acceleration (comfort / friction limit) in m/s^2
        """
        self.L = wheelbase
        self.max_steer = max_steer_rad
        self.max_accel = max_accel
        self.max_lat_accel = max_lat_accel

    def step(self, state: dict, delta: float, a: float, dt: float) -> dict:
        """
        Perform one discrete-time forward integration step.
        state = {'x': float, 'y': float, 'theta': float, 'v': float}
        delta = steering angle command (radians)
        a = acceleration command (m/s^2)
        dt = time delta (seconds)
        """
        # 1. Clamp steering and acceleration inputs to physical actuation limits
        delta_clamped = max(-self.max_steer, min(self.max_steer, delta))
        a_clamped = max(-self.max_accel, min(self.max_accel, a))
        
        # 2. Discrete forward integration
        x_next = state['x'] + state['v'] * math.cos(state['theta']) * dt
        y_next = state['y'] + state['v'] * math.sin(state['theta']) * dt
        theta_next = state['theta'] + (state['v'] / self.L) * math.tan(delta_clamped) * dt
        v_next = max(0.0, state['v'] + a_clamped * dt)
        
        # 3. Dynamic constraint metrics
        kappa = math.tan(delta_clamped) / self.L
        a_lat = (state['v'] ** 2) * kappa
        
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
    """
    Simulate a trajectory given control commands [(delta_0, a_0), (delta_1, a_1), ...]
    Returns a Polars DataFrame with full state telemetry.
    """
    history = []
    current_state = {'x': 0.0, 'y': 0.0, 'theta': 0.0, 'v': 5.0}  # Initial 5 m/s (~18 km/h)
    
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
        
    df = pl.DataFrame(history)
    return df


def plot_simulation_dashboard(df: pl.DataFrame, model: KinematicBicycleModel, output_html: str = "kinematic_bicycle_plotly.html"):
    """
    Creates an interactive 4-panel dashboard in Plotly.
    """
    violations = df.filter(pl.col("a_lat").abs() > model.max_lat_accel)
    print(f"Simulation completed: {len(df)} steps.")
    print(f"Comfort lateral acceleration violations (|a_lat| > {model.max_lat_accel} m/s^2): {len(violations)}")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "2D Vehicle Trajectory (Global X vs Y)",
            "Longitudinal Velocity Profile v(t)",
            "Front Steering Angle δ(t)",
            "Lateral Acceleration a_lat(t) vs Comfort Limit"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # 1. 2D Path
    fig.add_trace(
        go.Scatter(x=df["x"], y=df["y"], mode="lines+markers", name="Path (x,y)",
                   line=dict(color="#1f77b4", width=3),
                   marker=dict(size=4, color="#1f77b4")),
        row=1, col=1
    )
    
    # 2. Velocity
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["v"], mode="lines", name="Velocity (m/s)",
                   line=dict(color="#2ca02c", width=2.5)),
        row=1, col=2
    )
    
    # 3. Steering Angle
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["delta"], mode="lines", name="Steer δ (rad)",
                   line=dict(color="#ff7f0e", width=2.5)),
        row=2, col=1
    )
    fig.add_hline(y=model.max_steer, line_dash="dot", line_color="red",
                  annotation_text="+δ_max", row=2, col=1)
    fig.add_hline(y=-model.max_steer, line_dash="dot", line_color="red",
                  annotation_text="-δ_max", row=2, col=1)
    
    # 4. Lateral Acceleration
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["a_lat"], mode="lines", name="a_lat (m/s²)",
                   line=dict(color="#9467bd", width=2.5)),
        row=2, col=2
    )
    fig.add_hline(y=model.max_lat_accel, line_dash="dash", line_color="crimson",
                  annotation_text="+Max a_lat", row=2, col=2)
    fig.add_hline(y=-model.max_lat_accel, line_dash="dash", line_color="crimson",
                  annotation_text="-Max a_lat", row=2, col=2)
    
    # Highlight violations if any
    if len(violations) > 0:
        fig.add_trace(
            go.Scatter(x=violations["time"], y=violations["a_lat"], mode="markers",
                       marker=dict(color="red", size=8, symbol="circle-open"),
                       name="Limit Violations"),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Kinematic Bicycle Model: Multi-Channel Telemetry & Constraint Dashboard",
        height=720,
        showlegend=False
    )
    fig.write_html(output_html)
    print(f"Interactive plot successfully saved to {output_html}")


if __name__ == "__main__":
    model = KinematicBicycleModel(wheelbase=2.8, max_steer_rad=0.5, max_lat_accel=3.0)
    
    # Generate test control sequence: 10 seconds total (dt = 0.05s -> 200 steps)
    steps = 200
    controls = []
    for i in range(steps):
        t = i * 0.05
        if t < 2.0:
            delta = 0.0
            a = 1.0   # Accelerate on straight
        elif t < 6.0:
            delta = 0.3 * math.sin(math.pi * (t - 2.0) / 2.0) # S-turn steering
            a = 0.2
        else:
            delta = 0.0
            a = -0.5  # Decelerate
        controls.append((delta, a))
        
    df_telemetry = simulate_maneuver(model, controls, dt=0.05)
    plot_simulation_dashboard(df_telemetry, model, output_html="kinematic_bicycle_solution.html")
