"""
Session 06 Practice: Dynamic Motion Prediction & Time-to-Collision (TTC) Evaluator
---------------------------------------------------------------------------------
In this lab, you will implement dynamic motion prediction models (CV and CTRV)
and compute spatial-temporal TTC using oriented bounding box collision checking.

Tools used:
- Shapely: Oriented bounding box polygon geometric intersection
- Polars: Dynamic obstacle state tracking and risk metrics logging
- Plotly: 2D spatial-temporal trajectory collision visualization
"""

import math
import numpy as np
import polars as pl
import plotly.graph_objects as go
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


def create_vehicle_polygon(center_x: float, center_y: float, yaw_rad: float,
                            length: float = 4.5, width: float = 2.0) -> Polygon:
    """Creates a Shapely Polygon representing a rotated vehicle bounding box."""
    half_l = length / 2.0
    half_w = width / 2.0
    # Axis-aligned box centered at (0,0)
    base_box = Polygon([
        (-half_l, -half_w),
        (half_l, -half_w),
        (half_l, half_w),
        (-half_l, half_w)
    ])
    # Rotate by yaw (in degrees for shapely)
    rotated = rotate(base_box, math.degrees(yaw_rad), origin=(0, 0))
    # Translate to (center_x, center_y)
    final_poly = translate(rotated, xoff=center_x, yoff=center_y)
    return final_poly


def predict_constant_velocity(x0: float, y0: float, yaw0: float, v0: float,
                              dt: float = 0.1, horizon: float = 4.0) -> list[dict]:
    """Propagates obstacle future states under Constant Velocity (CV) model."""
    states = []
    num_steps = int(horizon / dt)
    for i in range(num_steps):
        t = i * dt
        x = x0 + v0 * math.cos(yaw0) * t
        y = y0 + v0 * math.sin(yaw0) * t
        states.append({"time": t, "x": x, "y": y, "yaw": yaw0, "v": v0})
    return states


def evaluate_ttc(ego_states: list[dict], obs_states: list[dict],
                 ego_dim: tuple = (4.5, 2.0), obs_dim: tuple = (4.5, 2.0)) -> tuple[float, int]:
    """
    Evaluates dynamic Time-to-Collision (TTC) between ego and dynamic obstacle trajectories.
    
    Returns:
        (ttc_seconds, collision_step_index)
        If no collision occurs, returns (float('inf'), -1).
        
    TODO (Student):
    1. Iterate through time-synchronized ego and obstacle states
    2. Construct Shapely bounding box polygons for each at time t
    3. Check polygon intersection: ego_poly.intersects(obs_poly)
    4. Return timestamp of first collision
    """
    for i, (ego_st, obs_st) in enumerate(zip(ego_states, obs_states)):
        ego_poly = create_vehicle_polygon(ego_st["x"], ego_st["y"], ego_st["yaw"], ego_dim[0], ego_dim[1])
        obs_poly = create_vehicle_polygon(obs_st["x"], obs_st["y"], obs_st["yaw"], obs_dim[0], obs_dim[1])
        
        if ego_poly.intersects(obs_poly):
            return ego_st["time"], i
            
    return float('inf'), -1


def run_intersection_risk_assessment() -> tuple[pl.DataFrame, float]:
    """
    Simulates Ego crossing an intersection while Dynamic Obstacle approaches perpendicularly.
    """
    dt = 0.1
    horizon = 5.0
    
    # Ego: Starts at (0, -25), moving North (yaw = pi/2) at 10 m/s
    ego_states = []
    num_steps = int(horizon / dt)
    for i in range(num_steps):
        t = i * dt
        ego_states.append({
            "time": round(t, 2),
            "x": 0.0,
            "y": -25.0 + 10.0 * t,
            "yaw": math.pi / 2,
            "v": 10.0
        })
        
    # Obstacle: Starts at (-30, 0), moving East (yaw = 0) at 12 m/s
    obs_states = predict_constant_velocity(x0=-30.0, y0=0.0, yaw0=0.0, v0=12.0, dt=dt, horizon=horizon)
    
    ttc, col_idx = evaluate_ttc(ego_states, obs_states)
    
    # Log synchronized telemetry into Polars DataFrame
    records = []
    for i in range(num_steps):
        ego = ego_states[i]
        obs = obs_states[i]
        dist = math.hypot(ego["x"] - obs["x"], ego["y"] - obs["y"])
        records.append({
            "time_s": ego["time"],
            "ego_x": round(ego["x"], 2),
            "ego_y": round(ego["y"], 2),
            "obs_x": round(obs["x"], 2),
            "obs_y": round(obs["y"], 2),
            "center_dist_m": round(dist, 2),
            "is_colliding": (i == col_idx)
        })
        
    return pl.DataFrame(records), ttc


def plot_dynamic_collision(df: pl.DataFrame, ttc: float, output_html: str = "dynamic_ttc.html"):
    """Visualizes 2D intersecting trajectories and collision zone with Plotly."""
    fig = go.Figure()
    
    # Ego Path
    fig.add_trace(go.Scatter(
        x=df["ego_x"], y=df["ego_y"], mode="lines+markers",
        name="Ego Trajectory (Northbound)", line=dict(color="blue", width=3)
    ))
    
    # Obstacle Path
    fig.add_trace(go.Scatter(
        x=df["obs_x"], y=df["obs_y"], mode="lines+markers",
        name="Obstacle Trajectory (Eastbound)", line=dict(color="orange", width=3)
    ))
    
    # Highlight collision point if any
    col_rows = df.filter(pl.col("is_colliding"))
    if len(col_rows) > 0:
        fig.add_trace(go.Scatter(
            x=[col_rows["ego_x"][0], col_rows["obs_x"][0]],
            y=[col_rows["ego_y"][0], col_rows["obs_y"][0]],
            mode="markers",
            marker=dict(size=18, color="red", symbol="x"),
            name=f"Collision Point (TTC = {ttc:.2f}s)"
        ))
        
    fig.update_layout(
        title=f"Dynamic Intersection Prediction & Collision Risk (TTC: {ttc:.2f}s)",
        xaxis_title="Global X (m)",
        yaxis_title="Global Y (m)",
        width=800,
        height=800
    )
    fig.write_html(output_html)
    print(f"Interactive TTC plot saved to {output_html}")


if __name__ == "__main__":
    df_risk, ttc = run_intersection_risk_assessment()
    print("--- Dynamic Collision Analysis (Polars DataFrame) ---")
    print(df_risk.filter(pl.col("center_dist_m") < 10.0))
    print(f"\nCalculated Time-To-Collision (TTC): {ttc:.2f} seconds")
    
    plot_dynamic_collision(df_risk, ttc, "dynamic_ttc_plotly.html")
