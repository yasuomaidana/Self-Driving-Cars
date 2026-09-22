"""
Session 01 Practice: Classical vs. Modern Velocity Profile Optimization
-----------------------------------------------------------------------
Student Lab Exercise

In this lab, you will implement and compare:
1. Classical Decoupled Velocity Profile Generation (using geometric lateral limits and forward-backward integration).
2. Modern Multi-Objective Trajectory Optimization (smooth time-domain optimization using hard constraints and jerk minimization).

Follow the TODO comments in the mathematical and definition functions below.
All plotting, problem setup, and telemetry logging logic are already provided.
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.integrate as integrate
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


def compute_2d_spatial_path(s: np.ndarray, kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstructs 2D Cartesian path (x, y, theta) by integrating road curvature along arc-length s:
      theta(s) = integral(kappa(s) ds)
      x(s) = integral(cos(theta(s)) ds)
      y(s) = integral(sin(theta(s)) ds)
    """
    theta = integrate.cumulative_trapezoid(kappa, s, initial=0.0)
    x = integrate.cumulative_trapezoid(np.cos(theta), s, initial=0.0)
    y = integrate.cumulative_trapezoid(np.sin(theta), s, initial=0.0)
    return x, y, theta


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
    Computes a classical decoupled velocity profile along arc-length coordinates s.
    
    TODO (Student):
    1. Compute static geometric speed bound 'v_bound' based on lateral acceleration limit:
       v_bound(s) = min(v_limit, sqrt(a_lat_max / kappa(s)))
    2. Forward pass: enforce maximum forward acceleration 'a_max'.
       v_forward = sqrt(v_{i-1}^2 + 2 * a_max * ds)
       v_prof[i] = min(v_bound[i], v_forward)
    3. Backward pass: enforce maximum braking deceleration 'a_min'.
       v_backward = sqrt(v_{i+1}^2 - 2 * a_min * ds)
       v_prof[i] = min(v_prof[i], v_backward)
    """
    kappa = road_curvature(s_points)
    N = len(s_points)
    v_prof = np.zeros(N)
    v_prof[0] = v_start
    
    # --- TODO: Compute v_bound ---
    v_bound = None  # TODO: compute static speed limit using v_limit, a_lat_max, and kappa
    
    # --- TODO: Forward integration pass ---
    # for i in range(1, N):
    #     ds = s_points[i] - s_points[i - 1]
    #     ...
    
    # --- TODO: Backward integration pass ---
    # for i in range(N - 2, -1, -1):
    #     ds = s_points[i + 1] - s_points[i]
    #     ...
    
    # [Pre-implemented] Telemetry conversion logic
    if v_bound is None:
        v_bound = np.full(N, v_limit)
        v_prof = np.full(N, v_start)

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
    x, y, theta = compute_2d_spatial_path(s_points, kappa)

    return {
        "s": s_points,
        "time": time_arr,
        "v": v_prof,
        "accel": accel,
        "jerk": jerk,
        "a_lat": a_lat,
        "kappa": kappa,
        "v_bound": v_bound,
        "x": x,
        "y": y,
        "theta": theta
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
    Formulates and solves a modern multi-objective trajectory optimization problem:
      min_v J(v) = w_track * J_track + w_accel * J_accel + w_jerk * J_jerk

    Subject to:
      - Initial velocity constraint: v(0) = v_start
      - Speed limit constraint: v_k <= v_limit
      - Hard lateral acceleration constraint: v_k^2 * kappa(s_k) <= a_lat_max

    TODO (Student):
    1. Integrate arc positions s_k over time horizon: s_{k+1} = s_k + v_k * dt
    2. Compute tracking cost: (v - v_target)^2
    3. Compute longitudinal acceleration cost: a_k = (v_{k+1} - v_k) / dt
    4. Compute longitudinal jerk cost: j_k = (a_{k+1} - a_k) / dt
    5. Implement the inequality constraint function lat_accel_constraint(v):
       return a_lat_max - (v ** 2) * kappa(s)  (must be >= 0)
    """
    time = np.linspace(0, N * dt, N + 1)
    
    def get_s(v):
        # --- TODO: Compute s position trajectory ---
        s = np.zeros(N + 1)
        # TODO: compute s[k+1] = s[k] + v[k] * dt
        return s

    def objective(v):
        # --- TODO: 1. Tracking cost (cost_track) ---
        cost_track = 0.0
        
        # --- TODO: 2. Longitudinal acceleration cost (cost_accel) ---
        cost_accel = 0.0
        
        # --- TODO: 3. Longitudinal jerk cost (cost_jerk) ---
        cost_jerk = 0.0
        
        return w_track * cost_track + w_accel * cost_accel + w_jerk * cost_jerk

    # Hard constraints:
    def lat_accel_constraint(v):
        # --- TODO: Compute lateral acceleration inequality constraint vector ---
        # Hint: return a_lat_max - (v ** 2) * road_curvature(get_s(v))
        s = get_s(v)
        kappa = road_curvature(s)
        return a_lat_max - (v ** 2) * kappa

    def speed_limit_constraint(v):
        return v_limit - v

    # [Pre-implemented] Numerical solver setup
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
    x_opt, y_opt, theta_opt = compute_2d_spatial_path(s_opt, kappa_opt)
    
    return {
        "s": s_opt,
        "time": time,
        "v": v_opt,
        "accel": accel_opt,
        "jerk": jerk_opt,
        "a_lat": a_lat_opt,
        "kappa": kappa_opt,
        "x": x_opt,
        "y": y_opt,
        "theta": theta_opt
    }


# ==========================================
# 4. Comparative Telemetry & Visualization
# ==========================================

def create_comparison_telemetry(classical_res: dict, modern_res: dict) -> pl.DataFrame:
    df_classical = pl.DataFrame({
        "time": classical_res["time"],
        "s": classical_res["s"],
        "x": classical_res["x"],
        "y": classical_res["y"],
        "velocity": classical_res["v"],
        "accel": classical_res["accel"],
        "jerk": classical_res["jerk"],
        "a_lat": classical_res["a_lat"],
        "method": ["Classical Decoupled"] * len(classical_res["time"])
    })
    
    df_modern = pl.DataFrame({
        "time": modern_res["time"],
        "s": modern_res["s"],
        "x": modern_res["x"],
        "y": modern_res["y"],
        "velocity": modern_res["v"],
        "accel": modern_res["accel"],
        "jerk": modern_res["jerk"],
        "a_lat": modern_res["a_lat"],
        "method": ["Modern Trajectory Opt"] * len(modern_res["time"])
    })
    
    return pl.concat([df_classical, df_modern])


def plot_optimization_comparison(classical_res: dict, modern_res: dict, v_limit: float = 13.0, a_lat_max: float = 2.5):
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "2D Spatial Trajectory Map (Bird's-Eye View)",
            "Velocity Profile v(s)",
            "Longitudinal Acceleration a(s)",
            "Longitudinal Jerk j(s) [Passenger Comfort]",
            "Lateral Acceleration a_lat(s)",
            "Road Curvature Profile κ(s)"
        ),
        vertical_spacing=0.09,
        horizontal_spacing=0.10
    )
    
    # 1. 2D Spatial Trajectory Map
    fig.add_trace(go.Scatter(
        x=classical_res["x"], y=classical_res["y"],
        mode='lines', name='Classical Profile',
        line=dict(color='crimson', width=3),
        legendgroup='classical'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=modern_res["x"], y=modern_res["y"],
        mode='lines', name='Modern Smooth Opt',
        line=dict(color='royalblue', width=3.5, dash='dot'),
        legendgroup='modern'
    ), row=1, col=1)
    
    # 2. Velocity Profile v(s)
    fig.add_trace(go.Scatter(
        x=classical_res["s"], y=classical_res["v_bound"],
        mode='lines', name='Geometric Bound v_bound(s)',
        line=dict(dash='dash', color='gray', width=1.5),
        legendgroup='bounds'
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=classical_res["s"], y=classical_res["v"],
        mode='lines', name='Classical Profile',
        line=dict(color='crimson', width=2),
        legendgroup='classical', showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=modern_res["s"], y=modern_res["v"],
        mode='lines', name='Modern Smooth Opt',
        line=dict(color='royalblue', width=2.5),
        legendgroup='modern', showlegend=False
    ), row=1, col=2)
    
    # 3. Acceleration Profile a(s)
    fig.add_trace(go.Scatter(
        x=classical_res["s"], y=classical_res["accel"],
        mode='lines', name='Classical Accel',
        line=dict(color='crimson', width=1.8),
        legendgroup='classical', showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=modern_res["s"], y=modern_res["accel"],
        mode='lines', name='Modern Accel',
        line=dict(color='royalblue', width=2),
        legendgroup='modern', showlegend=False
    ), row=2, col=1)
    
    # 4. Jerk Profile j(s)
    fig.add_trace(go.Scatter(
        x=classical_res["s"], y=classical_res["jerk"],
        mode='lines', name='Classical Jerk',
        line=dict(color='crimson', width=1.8),
        legendgroup='classical', showlegend=False
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=modern_res["s"], y=modern_res["jerk"],
        mode='lines', name='Modern Jerk',
        line=dict(color='royalblue', width=2),
        legendgroup='modern', showlegend=False
    ), row=2, col=2)
    
    # 5. Lateral Acceleration a_lat(s)
    fig.add_trace(go.Scatter(
        x=classical_res["s"], y=[a_lat_max] * len(classical_res["s"]),
        mode='lines', name='a_lat Max Comfort Limit',
        line=dict(dash='dash', color='orange', width=1.5),
        legendgroup='bounds', showlegend=False
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=classical_res["s"], y=classical_res["a_lat"],
        mode='lines', name='Classical a_lat',
        line=dict(color='crimson', width=1.8),
        legendgroup='classical', showlegend=False
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=modern_res["s"], y=modern_res["a_lat"],
        mode='lines', name='Modern a_lat',
        line=dict(color='royalblue', width=2),
        legendgroup='modern', showlegend=False
    ), row=3, col=1)
    
    # 6. Road Curvature Profile kappa(s)
    fig.add_trace(go.Scatter(
        x=classical_res["s"], y=classical_res["kappa"],
        mode='lines', name='Road Curvature κ(s)',
        line=dict(color='darkgreen', width=2),
        legendgroup='bounds', showlegend=False
    ), row=3, col=2)

    # Axis formatting
    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    
    fig.update_xaxes(title_text="Arc-Length s (m)", row=1, col=2)
    fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
    
    fig.update_xaxes(title_text="Arc-Length s (m)", row=2, col=1)
    fig.update_yaxes(title_text="Accel (m/s²)", row=2, col=1)
    
    fig.update_xaxes(title_text="Arc-Length s (m)", row=2, col=2)
    fig.update_yaxes(title_text="Jerk (m/s³)", row=2, col=2)
    
    fig.update_xaxes(title_text="Arc-Length s (m)", row=3, col=1)
    fig.update_yaxes(title_text="Lat Accel (m/s²)", row=3, col=1)
    
    fig.update_xaxes(title_text="Arc-Length s (m)", row=3, col=2)
    fig.update_yaxes(title_text="Curvature κ (1/m)", row=3, col=2)
    
    fig.update_layout(
        title_text="Comparison: Classical Decoupled vs. Modern Multi-Objective Trajectory Optimization",
        template="plotly_white",
        height=950,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5)
    )
    return fig


if __name__ == "__main__":
    s_points = np.linspace(0, 180, 200)
    print("Computing classical decoupled velocity profile...")
    classical_res = compute_classical_velocity_profile(s_points)
    
    print("Solving modern multi-objective trajectory optimization...")
    modern_res = solve_modern_trajectory_optimization()
    
    fig = plot_optimization_comparison(classical_res, modern_res)
    print("\n[Lab] Please complete TODO blocks to see full optimized comparison curves.")
