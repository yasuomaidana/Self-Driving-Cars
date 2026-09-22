"""
Session 01 Solution: Classical vs. Modern Velocity Profile Optimization
-----------------------------------------------------------------------
Instructor Reference Implementation

This script compares:
1. Classical Decoupled Spatial Velocity Profile (heuristic/bang-bang & lateral limit clamping)
2. Modern Multi-Objective Trajectory Optimization (smooth time-domain QP/NLP with hard constraints & jerk minimization)
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize


# ==========================================
# 1. Road & Problem Definition
# ==========================================

def road_curvature(s: np.ndarray) -> np.ndarray:
    """
    Gaussian road curve centered at s = 100m.
    kappa(s) = kappa_max * exp( - (s - s_curve)^2 / (2 * sigma^2) )
    """
    kappa_max = 0.04  # 1/m (Radius = 25m at apex)
    s_curve = 100.0   # curve center (meters)
    sigma = 15.0      # curve width (meters)
    return kappa_max * np.exp(-((s - s_curve) ** 2) / (2.0 * (sigma ** 2)))


# ==========================================
# 2. Classical Spatial Approach
# ==========================================

def compute_classical_velocity_profile(
    s_points: np.ndarray,
    v_start: float = 8.0,
    v_limit: float = 13.0,
    a_lat_max: float = 2.5,
    a_max: float = 2.0,
    a_min: float = -3.0
) -> dict:
    """
    Classical approach:
    1. Compute static geometric velocity limit: v_bound(s) = min(v_limit, sqrt(a_lat_max / kappa(s)))
    2. Apply forward and backward longitudinal acceleration integration passes.
    """
    kappa = road_curvature(s_points)
    v_bound = np.minimum(v_limit, np.sqrt(a_lat_max / np.maximum(kappa, 1e-6)))
    
    N = len(s_points)
    v_prof = np.zeros(N)
    v_prof[0] = v_start
    
    # Forward pass (acceleration limit)
    for i in range(1, N):
        ds = s_points[i] - s_points[i - 1]
        v_forward = np.sqrt(max(0.0, v_prof[i - 1]**2 + 2.0 * a_max * ds))
        v_prof[i] = min(v_bound[i], v_forward)
        
    # Backward pass (braking limit)
    for i in range(N - 2, -1, -1):
        ds = s_points[i + 1] - s_points[i]
        v_backward = np.sqrt(max(0.0, v_prof[i + 1]**2 - 2.0 * a_min * ds))
        v_prof[i] = min(v_prof[i], v_backward)
        
    # Time domain conversion
    dt_arr = np.zeros(N)
    time_arr = np.zeros(N)
    for i in range(1, N):
        ds = s_points[i] - s_points[i - 1]
        v_avg = max(0.5 * (v_prof[i] + v_prof[i - 1]), 0.1)
        dt_arr[i] = ds / v_avg
        time_arr[i] = time_arr[i - 1] + dt_arr[i]
        
    accel = np.gradient(v_prof, time_arr)
    jerk = np.gradient(accel, time_arr)
    a_lat = (v_prof ** 2) * kappa

    return {
        "s": s_points,
        "time": time_arr,
        "v": v_prof,
        "accel": accel,
        "jerk": jerk,
        "a_lat": a_lat,
        "kappa": kappa,
        "v_bound": v_bound
    }


# ==========================================
# 3. Modern Multi-Objective Trajectory Optimization
# ==========================================

def solve_modern_trajectory_optimization(
    N: int = 160,
    dt: float = 0.1,
    v_start: float = 8.0,
    v_target: float = 15.0,
    v_limit: float = 13.0,
    a_lat_max: float = 2.5,
    w_track: float = 1.0,
    w_accel: float = 2.0,
    w_jerk: float = 10.0
) -> dict:
    """
    Modern unified trajectory optimization:
    Solves for v = [v_0, ..., v_N] minimizing tracking + accel + jerk costs,
    subject to hard constraints:
      - Initial speed: v_0 = v_start
      - Speed limit: v_k <= v_limit
      - Lateral acceleration comfort: v_k^2 * kappa(s_k) <= a_lat_max
    """
    time = np.linspace(0, N * dt, N + 1)
    
    def get_s(v):
        s = np.zeros(N + 1)
        for k in range(N):
            s[k + 1] = s[k] + v[k] * dt
        return s

    def objective(v):
        # 1. Target speed tracking cost
        cost_track = np.sum((v - v_target) ** 2) * dt
        
        # 2. Longitudinal acceleration cost
        accel = np.diff(v) / dt
        cost_accel = np.sum(accel ** 2) * dt
        
        # 3. Longitudinal jerk cost (smoothness)
        jerk = np.diff(accel) / dt
        cost_jerk = np.sum(jerk ** 2) * dt
        
        return w_track * cost_track + w_accel * cost_accel + w_jerk * cost_jerk

    # Hard constraints:
    # 1. Lateral acceleration: a_lat_max - v^2 * kappa(s) >= 0
    def lat_accel_constraint(v):
        s = get_s(v)
        kappa = road_curvature(s)
        return a_lat_max - (v ** 2) * kappa

    # 2. Speed limit: v_limit - v >= 0
    def speed_limit_constraint(v):
        return v_limit - v

    bounds = [(0.0, v_limit) for _ in range(N + 1)]
    constraints = [
        {'type': 'eq', 'fun': lambda v: v[0] - v_start},
        {'type': 'ineq', 'fun': lat_accel_constraint},
        {'type': 'ineq', 'fun': speed_limit_constraint}
    ]
    
    v_init = np.full(N + 1, v_start)
    res = minimize(
        objective,
        v_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-6}
    )
    
    v_opt = res.x
    s_opt = get_s(v_opt)
    kappa_opt = road_curvature(s_opt)
    accel_opt = np.gradient(v_opt, time)
    jerk_opt = np.gradient(accel_opt, time)
    a_lat_opt = (v_opt ** 2) * kappa_opt
    
    return {
        "s": s_opt,
        "time": time,
        "v": v_opt,
        "accel": accel_opt,
        "jerk": jerk_opt,
        "a_lat": a_lat_opt,
        "kappa": kappa_opt
    }


# ==========================================
# 4. Comparative Telemetry & Visualization
# ==========================================

def create_comparison_telemetry(classical_res: dict, modern_res: dict) -> pl.DataFrame:
    df_classical = pl.DataFrame({
        "time": classical_res["time"],
        "s": classical_res["s"],
        "velocity": classical_res["v"],
        "accel": classical_res["accel"],
        "jerk": classical_res["jerk"],
        "a_lat": classical_res["a_lat"],
        "method": ["Classical Decoupled"] * len(classical_res["time"])
    })
    
    df_modern = pl.DataFrame({
        "time": modern_res["time"],
        "s": modern_res["s"],
        "velocity": modern_res["v"],
        "accel": modern_res["accel"],
        "jerk": modern_res["jerk"],
        "a_lat": modern_res["a_lat"],
        "method": ["Modern Trajectory Opt"] * len(modern_res["time"])
    })
    
    return pl.concat([df_classical, df_modern])


def plot_optimization_comparison(classical_res: dict, modern_res: dict, v_limit: float = 13.0, a_lat_max: float = 2.5):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Velocity Profile v(s)",
            "Longitudinal Acceleration a(s)",
            "Longitudinal Jerk j(s) [Passenger Comfort]",
            "Lateral Acceleration a_lat(s)"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # 1. Velocity Profile
    fig.add_trace(go.Scatter(x=classical_res["s"], y=classical_res["v_bound"], mode='lines', name='Geometric Bound v_bound(s)', line=dict(dash='dash', color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=classical_res["s"], y=classical_res["v"], mode='lines', name='Classical Profile', line=dict(color='crimson', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=modern_res["s"], y=modern_res["v"], mode='lines', name='Modern Smooth Opt', line=dict(color='royalblue', width=2.5)), row=1, col=1)
    
    # 2. Acceleration Profile
    fig.add_trace(go.Scatter(x=classical_res["s"], y=classical_res["accel"], mode='lines', name='Classical Accel', line=dict(color='crimson', width=1.5), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=modern_res["s"], y=modern_res["accel"], mode='lines', name='Modern Accel', line=dict(color='royalblue', width=2), showlegend=False), row=1, col=2)
    
    # 3. Jerk Profile
    fig.add_trace(go.Scatter(x=classical_res["s"], y=classical_res["jerk"], mode='lines', name='Classical Jerk', line=dict(color='crimson', width=1.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=modern_res["s"], y=modern_res["jerk"], mode='lines', name='Modern Jerk', line=dict(color='royalblue', width=2), showlegend=False), row=2, col=1)
    
    # 4. Lateral Acceleration Profile
    fig.add_trace(go.Scatter(x=classical_res["s"], y=[a_lat_max]*len(classical_res["s"]), mode='lines', name='a_lat Max Comfort Limit', line=dict(dash='dash', color='orange')), row=2, col=2)
    fig.add_trace(go.Scatter(x=classical_res["s"], y=classical_res["a_lat"], mode='lines', name='Classical a_lat', line=dict(color='crimson', width=1.5), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=modern_res["s"], y=modern_res["a_lat"], mode='lines', name='Modern a_lat', line=dict(color='royalblue', width=2), showlegend=False), row=2, col=2)
    
    # Layout adjustments
    fig.update_xaxes(title_text="Arc-Length s (m)", row=1, col=1)
    fig.update_xaxes(title_text="Arc-Length s (m)", row=1, col=2)
    fig.update_xaxes(title_text="Arc-Length s (m)", row=2, col=1)
    fig.update_xaxes(title_text="Arc-Length s (m)", row=2, col=2)
    
    fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Accel (m/s²)", row=1, col=2)
    fig.update_yaxes(title_text="Jerk (m/s³)", row=2, col=1)
    fig.update_yaxes(title_text="Lat Accel (m/s²)", row=2, col=2)
    
    fig.update_layout(
        title_text="Comparison: Classical Decoupled vs. Modern Multi-Objective Trajectory Optimization",
        template="plotly_white",
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
    )
    return fig


if __name__ == "__main__":
    s_points = np.linspace(0, 180, 200)
    print("Computing classical decoupled velocity profile...")
    classical_res = compute_classical_velocity_profile(s_points)
    
    print("Solving modern multi-objective trajectory optimization...")
    modern_res = solve_modern_trajectory_optimization()
    
    df_telemetry = create_comparison_telemetry(classical_res, modern_res)
    print(f"\nTelemetry DataFrame generated successfully with shape: {df_telemetry.shape}")
    print(df_telemetry.head(5))
    
    fig = plot_optimization_comparison(classical_res, modern_res)
    print(f"\nFinal Distance Covered - Classical: {classical_res['s'][-1]:.1f}m | Modern: {modern_res['s'][-1]:.1f}m")
    print(f"Max Lateral Accel - Classical: {np.max(classical_res['a_lat']):.2f} m/s² | Modern: {np.max(modern_res['a_lat']):.2f} m/s² (Limit: 2.50 m/s²)")
