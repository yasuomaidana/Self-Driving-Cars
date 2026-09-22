"""
Session 09 Practice: Path Optimization & Curvature-Constrained Velocity Profiling
---------------------------------------------------------------------------------
In this lab, you will generate a smooth trajectory from coarse waypoints and compute
a dynamically feasible velocity profile obeying speed limits, curvature limits,
and acceleration/braking bounds.

Tools used:
- SciPy: Numerical optimization and spline interpolation
- Polars: Telemetry data logging and constraint violation analysis
- Plotly: Interactive spatial trajectory and velocity/acceleration dashboard
"""

import math
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import splprep, splev


def generate_curved_path_waypoints() -> tuple[np.ndarray, np.ndarray]:
    """Generates coarse path waypoints including a straight road, a 90-degree turn, and a final straight."""
    x = [0.0, 30.0, 60.0, 80.0, 90.0, 95.0, 95.0, 95.0]
    y = [0.0, 0.0, 0.0, 10.0, 30.0, 50.0, 80.0, 110.0]
    return np.array(x), np.array(y)


def fit_smooth_spline_path(x_pts: np.ndarray, y_pts: np.ndarray, num_samples: int = 150):
    """
    Fits a B-spline to coarse waypoints and evaluates continuous positions and derivatives.
    """
    tck, u = splprep([x_pts, y_pts], s=0, k=3)
    u_new = np.linspace(0, 1, num_samples)
    x_smooth, y_smooth = splev(u_new, tck)
    
    # First derivatives dx/du, dy/du
    dx, dy = splev(u_new, tck, der=1)
    # Second derivatives d2x/du2, d2y/du2
    ddx, ddy = splev(u_new, tck, der=2)
    
    # Path curvature: kappa = (dx*ddy - dy*ddx) / (dx^2 + dy^2)^(1.5)
    curvature = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5)
    
    # Cumulative arc length s
    ds = np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    
    return s, x_smooth, y_smooth, curvature


def compute_velocity_profile(s: np.ndarray, curvature: np.ndarray,
                             v_cruise: float = 12.0,
                             a_lat_max: float = 2.5,
                             a_accel_max: float = 1.5,
                             a_decel_max: float = 2.0,
                             v_init: float = 0.0) -> np.ndarray:
    """
    Computes a dynamically feasible velocity profile using forward and backward integration passes.
    
    TODO (Student):
    1. Initialize v_max based on v_cruise and curvature limit: v_max = sqrt(a_lat_max / abs(kappa))
    2. Set final target velocity v_max[-1] = 0.0 (stop at end of path)
    3. Perform Backward pass to enforce braking deceleration limit (a_decel_max)
    4. Perform Forward pass from v_init to enforce acceleration limit (a_accel_max)
    5. Return final velocity array
    """
    N = len(s)
    v_max = np.zeros(N)
    
    for i in range(N):
        k = abs(curvature[i])
        if k > 1e-4:
            v_curv = math.sqrt(a_lat_max / k)
        else:
            v_curv = v_cruise
        v_max[i] = min(v_cruise, v_curv)
        
    v_max[-1] = 0.0 # Complete stop at goal
    
    # --- Backward Pass (Braking) ---
    v_profile = np.copy(v_max)
    for i in range(N - 2, -1, -1):
        ds = s[i+1] - s[i]
        # v_i^2 <= v_{i+1}^2 + 2 * a_decel * ds
        v_allowed = math.sqrt(v_profile[i+1]**2 + 2 * a_decel_max * ds)
        v_profile[i] = min(v_profile[i], v_allowed)
        
    # --- Forward Pass (Acceleration) ---
    v_profile[0] = v_init
    for i in range(0, N - 1):
        ds = s[i+1] - s[i]
        # v_{i+1}^2 <= v_i^2 + 2 * a_accel * ds
        v_allowed = math.sqrt(v_profile[i]**2 + 2 * a_accel_max * ds)
        v_profile[i+1] = min(v_profile[i+1], v_allowed)
        
    return v_profile


def build_telemetry_and_plot(s: np.ndarray, x: np.ndarray, y: np.ndarray,
                             curvature: np.ndarray, v_profile: np.ndarray,
                             a_lat_max: float = 2.5,
                             output_html: str = "velocity_profile.html") -> pl.DataFrame:
    """Builds Polars telemetry DataFrame and renders multi-panel Plotly visualization."""
    a_lat = (v_profile ** 2) * np.abs(curvature)
    
    df_telemetry = pl.DataFrame({
        "s_m": np.round(s, 2),
        "x_m": np.round(x, 2),
        "y_m": np.round(y, 2),
        "curvature_inv_m": np.round(curvature, 4),
        "velocity_mps": np.round(v_profile, 2),
        "lat_accel_mps2": np.round(a_lat, 2)
    })
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Spatial Path (X vs Y)", "Path Curvature κ(s)",
                        "Velocity Profile v(s)", "Lateral Acceleration a_lat(s)")
    )
    
    # 1. Path
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Path", line=dict(color="blue", width=3)), row=1, col=1)
    # 2. Curvature
    fig.add_trace(go.Scatter(x=s, y=curvature, mode="lines", name="Curvature", line=dict(color="orange")), row=1, col=2)
    # 3. Velocity Profile
    fig.add_trace(go.Scatter(x=s, y=v_profile, mode="lines", name="Velocity (m/s)", line=dict(color="green", width=3)), row=2, col=1)
    # 4. Lateral Acceleration
    fig.add_trace(go.Scatter(x=s, y=a_lat, mode="lines", name="a_lat (m/s^2)", line=dict(color="purple")), row=2, col=2)
    fig.add_hline(y=a_lat_max, line_dash="dash", line_color="red", annotation_text="Max Lat Accel", row=2, col=2)
    
    fig.update_layout(title="Trajectory Optimization & Dynamic Velocity Profiling", height=700)
    fig.write_html(output_html)
    print(f"Interactive velocity profile saved to {output_html}")
    
    return df_telemetry


if __name__ == "__main__":
    x_coarse, y_coarse = generate_curved_path_waypoints()
    s, x_smooth, y_smooth, curvature = fit_smooth_spline_path(x_coarse, y_coarse)
    
    v_prof = compute_velocity_profile(s, curvature, v_cruise=12.0, a_lat_max=2.5, a_accel_max=1.5, a_decel_max=2.0)
    df_res = build_telemetry_and_plot(s, x_smooth, y_smooth, curvature, v_prof, a_lat_max=2.5, output_html="velocity_profiler_plotly.html")
    
    print("--- Velocity Profile Summary (Polars DataFrame) ---")
    print(df_res.head(10))
