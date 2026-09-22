"""
Session 01 Practice: Kinematic Bicycle Model & Constraint Verification
----------------------------------------------------------------------
In this lab, you will implement a discrete-time non-holonomic Kinematic Bicycle Model
and evaluate trajectory validity against physical and comfort constraints.

Tools used:
- Polars: For high-performance tabular state telemetry
- Plotly: For interactive spatial and dynamic trajectory plotting
"""

import math
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class KinematicBicycleModel:
    def __init__(self, wheelbase: float = 2.8, max_steer_rad: float = 0.6, max_accel: float = 3.0, max_lat_accel: float = 3.5):
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

        TODO (Student):
        1. Clamp delta to [-self.max_steer, self.max_steer]
        2. Clamp a to [-self.max_accel, self.max_accel]
        3. Compute next state (x_next, y_next, theta_next, v_next) using Kinematic Bicycle Model
        4. Calculate curvature kappa and lateral acceleration a_lat = v^2 * kappa
        5. Return updated state dict
        """
        # --- YOUR CODE HERE ---
        # Replace the dummy pass-through below:
        delta_clamped = max(-self.max_steer, min(self.max_steer, delta))
        a_clamped = max(-self.max_accel, min(self.max_accel, a))
        
        # TODO: Compute state derivatives & integrate
        x_next = state['x'] + state['v'] * math.cos(state['theta']) * dt
        y_next = state['y'] + state['v'] * math.sin(state['theta']) * dt
        theta_next = state['theta'] + (state['v'] / self.L) * math.tan(delta_clamped) * dt
        v_next = max(0.0, state['v'] + a_clamped * dt)
        
        kappa = math.tan(delta_clamped) / self.L
        a_lat = (state['v'] ** 2) * kappa
        
        return {
            'x': x_next,
            'y': y_next,
            'theta': theta_next,
            'v': v_next,
            'delta': delta_clamped,
            'a': a_clamped,
            'kappa': kappa,
            'a_lat': a_lat
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
        
    # Convert simulation history into Polars DataFrame
    df = pl.DataFrame(history)
    return df


def verify_and_plot(df: pl.DataFrame, model: KinematicBicycleModel, output_html: str = "trajectory_analysis.html"):
    """
    Validates dynamic constraints and creates an interactive Plotly dashboard.
    """
    # Polars constraint evaluation: Identify instances exceeding comfort limits
    violations = df.filter(pl.col("a_lat").abs() > model.max_lat_accel)
    print(f"Total simulation steps: {len(df)}")
    print(f"Comfort lateral acceleration violations (|a_lat| > {model.max_lat_accel} m/s^2): {len(violations)}")
    
    # Create interactive multi-panel Plotly figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("2D Vehicle Trajectory (X vs Y)", "Velocity Profile v(t)",
                        "Steering Angle δ(t)", "Lateral Acceleration a_lat(t) vs Limit")
    )
    
    # 1. 2D Trajectory
    fig.add_trace(
        go.Scatter(x=df["x"], y=df["y"], mode="lines+markers", name="Path (x,y)",
                   line=dict(color="blue", width=3)),
        row=1, col=1
    )
    
    # 2. Velocity
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["v"], mode="lines", name="Velocity (m/s)",
                   line=dict(color="green", width=2)),
        row=1, col=2
    )
    
    # 3. Steering Angle
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["delta"], mode="lines", name="Steer δ (rad)",
                   line=dict(color="orange", width=2)),
        row=2, col=1
    )
    
    # 4. Lateral Acceleration
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["a_lat"], mode="lines", name="Lateral Accel",
                   line=dict(color="purple", width=2)),
        row=2, col=2
    )
    # Add lateral acceleration limit lines
    fig.add_hline(y=model.max_lat_accel, line_dash="dash", line_color="red",
                  annotation_text="+Max Lat Accel", row=2, col=2)
    fig.add_hline(y=-model.max_lat_accel, line_dash="dash", line_color="red",
                  annotation_text="-Max Lat Accel", row=2, col=2)
    
    fig.update_layout(title="Kinematic Bicycle Model Simulation & Constraint Analysis", height=700)
    fig.write_html(output_html)
    print(f"Interactive plot successfully saved to {output_html}")


if __name__ == "__main__":
    model = KinematicBicycleModel(wheelbase=2.8, max_steer_rad=0.5, max_lat_accel=3.0)
    
    # Generate test control sequence: 10 seconds total (dt = 0.05s -> 200 steps)
    # Maneuver: Accelerate while performing a smooth S-turn (lane change)
    steps = 200
    controls = []
    for i in range(steps):
        t = i * 0.05
        # Sinusoidal steering maneuver
        if t < 2.0:
            delta = 0.0
            a = 1.0  # accelerate
        elif t < 6.0:
            delta = 0.3 * math.sin(math.pi * (t - 2.0) / 2.0)
            a = 0.2
        else:
            delta = 0.0
            a = -0.5 # decelerate
        controls.append((delta, a))
        
    df_telemetry = simulate_maneuver(model, controls, dt=0.05)
    verify_and_plot(df_telemetry, model, output_html="kinematic_bicycle_plotly.html")
