"""
Session 01 Solution: Frenet Frame (s, d) 2D Trajectory Optimization & Smooth Dynamic Profiling
----------------------------------------------------------------------------------------------
Instructor Reference Implementation

In this lab, we formulate and solve a 2D trajectory planning problem using:
1. Frenet Frame (curvilinear coordinates):
   - Along-track coordinate: s (road progress)
   - Cross-track coordinate: d (lateral offset from lane centerline)
2. Convex Matrix Quadratic Program (QP) with obstacle avoidance corridor bounds.
3. Smooth Convex Velocity Profiling on Kinetic Energy u(s) = v(s)^2 with hard lateral comfort bounds.
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.integrate as integrate
from scipy.optimize import minimize


# ==========================================
# 1. Reference Road & Frenet Transformation
# ==========================================

def generate_curved_reference_path(s_max: float = 150.0, ds: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a curved reference road centerline (s, kappa_ref, theta_ref, x_ref, y_ref).
    """
    N = int(s_max / ds)
    s = np.linspace(0.0, s_max, N + 1)
    
    # Gaussian road curve centered at s = 75m
    kappa_max = 0.025  # 1/m (Radius = 40m)
    s_curve = 75.0
    sigma = 20.0
    kappa_ref = kappa_max * np.exp(-((s - s_curve) ** 2) / (2.0 * (sigma ** 2)))
    
    # Integrate heading and Cartesian coordinates
    theta_ref = integrate.cumulative_trapezoid(kappa_ref, s, initial=0.0)
    x_ref = integrate.cumulative_trapezoid(np.cos(theta_ref), s, initial=0.0)
    y_ref = integrate.cumulative_trapezoid(np.sin(theta_ref), s, initial=0.0)
    
    return s, kappa_ref, theta_ref, x_ref, y_ref


def frenet_to_cartesian(s: np.ndarray, d: np.ndarray, s_ref: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray, theta_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts Frenet coordinates (s, d) to global Cartesian (x, y):
      x(s, d) = x_ref(s) - d * sin(theta_ref(s))
      y(s, d) = y_ref(s) + d * cos(theta_ref(s))
    """
    x_r = np.interp(s, s_ref, x_ref)
    y_r = np.interp(s, s_ref, y_ref)
    th_r = np.interp(s, s_ref, theta_ref)
    
    x = x_r - d * np.sin(th_r)
    y = y_r + d * np.cos(th_r)
    return x, y


# ==========================================
# 2. Obstacle & Feasible Corridor Definition
# ==========================================

def build_corridor_bounds(
    s: np.ndarray,
    road_width: float = 8.0,
    obs_s_start: float = 50.0,
    obs_s_end: float = 75.0,
    obs_d_min: float = -2.0,
    obs_d_max: float = 0.6,
    vehicle_width: float = 2.0,
    safety_buffer: float = 0.4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Defines lateral box bounds [d_min(s), d_max(s)] along the road.
    Accounts for vehicle half-width and safety buffer to ensure zero collision.
    """
    half_width = road_width / 2.0
    d_min = np.full_like(s, -half_width)
    d_max = np.full_like(s, half_width)
    
    # Safe lateral bound for vehicle centerline
    safe_lateral_bound = obs_d_max + (vehicle_width / 2.0) + safety_buffer  # 2.0m
    
    for i, s_val in enumerate(s):
        if (obs_s_start - 3.0) <= s_val <= (obs_s_end + 3.0):
            d_min[i] = safe_lateral_bound
            
    return d_min, d_max


# ==========================================
# 3. Matrix QP Optimization Formulation
# ==========================================

def build_differentiation_matrices(N: int, ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds discrete finite difference differentiation matrices D1, D2, D3 for:
      - d'(s)   = D1 * d (Heading deviation)
      - d''(s)  = D2 * d (Curvature deviation)
      - d'''(s) = D3 * d (Lateral jerk)
    """
    D1 = np.zeros((N, N + 1))
    for i in range(N):
        D1[i, i] = -1.0 / ds
        D1[i, i + 1] = 1.0 / ds
        
    D2 = np.zeros((N - 1, N + 1))
    for i in range(N - 1):
        D2[i, i] = 1.0 / (ds ** 2)
        D2[i, i + 1] = -2.0 / (ds ** 2)
        D2[i, i + 2] = 1.0 / (ds ** 2)
        
    D3 = np.zeros((N - 2, N + 1))
    for i in range(N - 2):
        D3[i, i] = -1.0 / (ds ** 3)
        D3[i, i + 1] = 3.0 / (ds ** 3)
        D3[i, i + 2] = -3.0 / (ds ** 3)
        D3[i, i + 3] = 1.0 / (ds ** 3)
        
    return D1, D2, D3


def solve_frenet_trajectory_qp(
    s: np.ndarray,
    d_min: np.ndarray,
    d_max: np.ndarray,
    w_center: float = 1.0,
    w_heading: float = 20.0,
    w_curv: float = 200.0,
    w_jerk: float = 1500.0,
    d_start: float = 0.0,
    d_prime_start: float = 0.0
) -> np.ndarray:
    """
    Solves the Convex Matrix Quadratic Program for lateral deviation d(s):
      min_d  1/2 * d^T Q d + q^T d
      s.t.   d_min <= d <= d_max
             d(0) = d_start, d'(0) = d_prime_start
    """
    N = len(s) - 1
    ds = s[1] - s[0]
    
    D1, D2, D3 = build_differentiation_matrices(N, ds)
    
    d_ref = np.zeros(N + 1)
    Q = 2.0 * (
        w_center * np.eye(N + 1) +
        w_heading * (D1.T @ D1) +
        w_curv * (D2.T @ D2) +
        w_jerk * (D3.T @ D3)
    )
    q = -2.0 * w_center * d_ref
    
    bounds = list(zip(d_min, d_max))
    bounds[0] = (d_start, d_start)
    bounds[1] = (d_start + d_prime_start * ds, d_start + d_prime_start * ds)
    
    def cost_fun(d):
        return 0.5 * d @ Q @ d + q @ d

    def cost_grad(d):
        return Q @ d + q

    d_init = np.clip(np.zeros(N + 1), d_min, d_max)
    
    res = minimize(
        cost_fun,
        d_init,
        jac=cost_grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-9, 'gtol': 1e-7, 'maxiter': 500}
    )
    
    return res.x


# ==========================================
# 4. Smooth Convex Velocity Profiling on u = v^2
# ==========================================

def solve_smooth_longitudinal_profile(
    s: np.ndarray,
    kappa_total: np.ndarray,
    v_start: float = 8.0,
    v_target: float = 12.0,
    v_limit: float = 12.0,
    a_lat_max: float = 2.0,
    w_track_u: float = 1.0,
    w_acc_u: float = 150.0,
    w_jerk_u: float = 2000.0
) -> dict:
    """
    Solves a smooth convex quadratic program on kinetic energy u(s) = v(s)^2:
      min_u  1/2 * u^T Q_u u + q_u^T u
      s.t.   u(0) = v_start^2
             0 <= u(s) <= u_bound(s) = min(v_limit^2, a_lat_max / |kappa_total(s)|)
    """
    N = len(s) - 1
    ds = s[1] - s[0]
    
    D1, D2, _ = build_differentiation_matrices(N, ds)
    
    # Exact lateral acceleration limit on kinetic energy: u <= a_lat_max / |kappa|
    v_bound = np.minimum(v_limit, np.sqrt(a_lat_max / np.maximum(np.abs(kappa_total), 1e-5)))
    u_bound = v_bound ** 2
    u_target = v_target ** 2
    
    Q_u = 2.0 * (
        w_track_u * np.eye(N + 1) +
        w_acc_u * (D1.T @ D1) +
        w_jerk_u * (D2.T @ D2)
    )
    q_u = -2.0 * w_track_u * np.full(N + 1, u_target)
    
    bounds_u = [(0.0, max(0.0, u_bound[i])) for i in range(N + 1)]
    bounds_u[0] = (v_start ** 2, v_start ** 2)
    bounds_u[1] = (v_start ** 2, v_start ** 2)
    
    u_upper = np.array([b[1] for b in bounds_u])
    u_init = np.clip(np.full(N + 1, v_start ** 2), 0.0, u_upper)
    
    res = minimize(
        lambda u: 0.5 * u @ Q_u @ u + q_u @ u,
        u_init,
        jac=lambda u: Q_u @ u + q_u,
        method='L-BFGS-B',
        bounds=bounds_u,
        options={'ftol': 1e-9, 'gtol': 1e-7, 'maxiter': 500}
    )
    
    v_opt = np.sqrt(res.x)
    
    # Time domain integration
    dt_arr = np.zeros(N + 1)
    time_arr = np.zeros(N + 1)
    for i in range(1, N + 1):
        v_avg = 0.5 * (v_opt[i] + v_opt[i - 1])
        dt_arr[i] = ds / v_avg
        time_arr[i] = time_arr[i - 1] + dt_arr[i]
        
    a_long = np.gradient(v_opt, time_arr)
    j_long = np.gradient(a_long, time_arr)
    a_lat = (v_opt ** 2) * kappa_total
    
    return {
        "time": time_arr,
        "s": s,
        "v": v_opt,
        "a_long": a_long,
        "j_long": j_long,
        "a_lat": a_lat,
        "kappa_total": kappa_total,
        "v_bound": v_bound
    }


# ==========================================
# 5. Interactive Plotly Visualization
# ==========================================

def plot_frenet_trajectory_results(
    s: np.ndarray,
    kappa_ref: np.ndarray,
    theta_ref: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    d_opt: np.ndarray,
    d_min: np.ndarray,
    d_max: np.ndarray,
    dyn_res: dict,
    road_width: float = 8.0,
    obs_s_start: float = 50.0,
    obs_s_end: float = 75.0,
    obs_d_min: float = -2.0,
    obs_d_max: float = 0.6,
    a_lat_max: float = 2.0,
    v_limit: float = 12.0
) -> go.Figure:
    half_w = road_width / 2.0
    ds = s[1] - s[0]
    
    # 2D Cartesian conversions
    x_center, y_center = frenet_to_cartesian(s, np.zeros_like(s), s, x_ref, y_ref, theta_ref)
    x_left, y_left = frenet_to_cartesian(s, np.full_like(s, half_w), s, x_ref, y_ref, theta_ref)
    x_right, y_right = frenet_to_cartesian(s, np.full_like(s, -half_w), s, x_ref, y_ref, theta_ref)
    x_opt, y_opt = frenet_to_cartesian(s, d_opt, s, x_ref, y_ref, theta_ref)
    
    # Densely sampled obstacle curved footprint in Cartesian coordinates
    s_top = np.linspace(obs_s_start, obs_s_end, 40)
    d_top = np.full_like(s_top, obs_d_max)
    s_bot = np.linspace(obs_s_end, obs_s_start, 40)
    d_bot = np.full_like(s_bot, obs_d_min)
    s_poly = np.concatenate([s_top, s_bot, [obs_s_start]])
    d_poly = np.concatenate([d_top, d_bot, [obs_d_max]])
    x_obs, y_obs = frenet_to_cartesian(s_poly, d_poly, s, x_ref, y_ref, theta_ref)
    
    d_prime = np.gradient(d_opt, ds)
    
    s_dyn = dyn_res["s"]
    v_prof = dyn_res["v"]
    a_long = dyn_res["a_long"]
    j_long = dyn_res["j_long"]
    a_lat = dyn_res["a_lat"]
    kappa_total_path = dyn_res["kappa_total"]
    v_bound = dyn_res["v_bound"]
    
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "2D Global Cartesian Map (Obstacle Avoidance)",
            "Frenet Lateral Profile d(s) & Safe Corridor",
            "Velocity Profile v(s)",
            "Total Curvature Profile κ_total(s)",
            "Smooth Longitudinal Acceleration a_long(s)",
            "Lateral Acceleration a_lat(s)",
            "Smooth Longitudinal Jerk j_long(s) [Passenger Comfort]",
            "Heading Deviation Angle d'(s) = Δθ(s)"
        ),
        vertical_spacing=0.07,
        horizontal_spacing=0.10
    )
    
    # 1. 2D Global Map
    fig.add_trace(go.Scatter(x=x_left, y=y_left, mode='lines', line=dict(color='black', width=2), name='Road Boundaries', legendgroup='road'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_right, y=y_right, mode='lines', line=dict(color='black', width=2), showlegend=False, legendgroup='road'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_center, y=y_center, mode='lines', line=dict(dash='dash', color='gray', width=1.5), name='Lane Centerline', legendgroup='center'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_obs, y=y_obs, fill='toself', fillcolor='rgba(255, 0, 0, 0.45)', line=dict(color='crimson', width=2), name='Obstacle Footprint', legendgroup='obs'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_opt, y=y_opt, mode='lines', line=dict(color='royalblue', width=3.5), name='Optimized Trajectory', legendgroup='opt'), row=1, col=1)
    
    # 2. Frenet Lateral Profile
    fig.add_trace(go.Scatter(x=s, y=d_max, mode='lines', line=dict(color='red', dash='dash', width=1.5), name='Corridor Bounds d_min/d_max', legendgroup='bounds'), row=1, col=2)
    fig.add_trace(go.Scatter(x=s, y=d_min, mode='lines', line=dict(color='red', dash='dash', width=1.5), showlegend=False, legendgroup='bounds'), row=1, col=2)
    fig.add_trace(go.Scatter(x=s, y=np.zeros_like(s), mode='lines', line=dict(color='gray', dash='dot', width=1.5), showlegend=False, legendgroup='center'), row=1, col=2)
    fig.add_trace(go.Scatter(x=s, y=d_opt, mode='lines', line=dict(color='royalblue', width=2.5), showlegend=False, legendgroup='opt'), row=1, col=2)
    
    # 3. Velocity Profile
    fig.add_trace(go.Scatter(x=s_dyn, y=v_bound, mode='lines', line=dict(color='gray', dash='dash', width=1.5), name='Geometric Bound v_bound(s)', legendgroup='vbound'), row=2, col=1)
    fig.add_trace(go.Scatter(x=s_dyn, y=v_prof, mode='lines', line=dict(color='royalblue', width=2.5), name='Velocity v(s)', legendgroup='opt', showlegend=False), row=2, col=1)
    
    # 4. Total Curvature
    fig.add_trace(go.Scatter(x=s, y=kappa_ref, mode='lines', line=dict(color='gray', dash='dash', width=1.5), name='κ_ref (Centerline)', legendgroup='kref'), row=2, col=2)
    fig.add_trace(go.Scatter(x=s, y=kappa_total_path, mode='lines', line=dict(color='teal', width=2), name='κ_total(s)', legendgroup='ktot'), row=2, col=2)
    
    # 5. Longitudinal Acceleration
    fig.add_trace(go.Scatter(x=s_dyn, y=a_long, mode='lines', line=dict(color='indigo', width=2), name='a_long(s)', legendgroup='opt', showlegend=False), row=3, col=1)
    
    # 6. Lateral Acceleration
    fig.add_trace(go.Scatter(x=s_dyn, y=[a_lat_max]*len(s_dyn), mode='lines', line=dict(color='orange', dash='dash', width=1.5), name='a_lat Max Comfort Limit', legendgroup='bounds', showlegend=False), row=3, col=2)
    fig.add_trace(go.Scatter(x=s_dyn, y=a_lat, mode='lines', line=dict(color='darkorange', width=2), name='a_lat(s)', legendgroup='opt', showlegend=False), row=3, col=2)
    
    # 7. Longitudinal Jerk
    fig.add_trace(go.Scatter(x=s_dyn, y=j_long, mode='lines', line=dict(color='firebrick', width=2), name='j_long(s)', legendgroup='opt', showlegend=False), row=4, col=1)
    
    # 8. Heading Angle Deviation
    fig.add_trace(go.Scatter(x=s, y=np.rad2deg(d_prime), mode='lines', line=dict(color='purple', width=2), name="d'(s) Heading Deviation", legendgroup='d1'), row=4, col=2)
    
    # Axis formatting
    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    
    fig.update_xaxes(title_text="Arc-Length s (m)", row=1, col=2)
    fig.update_yaxes(title_text="Lateral Offset d (m)", row=1, col=2)
    
    fig.update_xaxes(title_text="Arc-Length s (m)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
    
    fig.update_xaxes(title_text="Arc-Length s (m)", row=2, col=2)
    fig.update_yaxes(title_text="Curvature κ (1/m)", row=2, col=2)
    
    fig.update_xaxes(title_text="Arc-Length s (m)", row=3, col=1)
    fig.update_yaxes(title_text="Long Accel (m/s²)", row=3, col=1)
    
    fig.update_xaxes(title_text="Arc-Length s (m)", row=3, col=2)
    fig.update_yaxes(title_text="Lat Accel (m/s²)", row=3, col=2)
    
    fig.update_xaxes(title_text="Arc-Length s (m)", row=4, col=1)
    fig.update_yaxes(title_text="Jerk (m/s³)", row=4, col=1)
    
    fig.update_xaxes(title_text="Arc-Length s (m)", row=4, col=2)
    fig.update_yaxes(title_text="Heading Δθ (deg)", row=4, col=2)
    
    fig.update_layout(
        title_text="2D Frenet Trajectory Optimization & Smooth Dynamic Profile (Jerk Minimization)",
        template="plotly_white",
        height=1150,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


if __name__ == "__main__":
    print("1. Generating curved reference road...")
    s, kappa_ref, theta_ref, x_ref, y_ref = generate_curved_reference_path()
    
    print("2. Constructing obstacle corridor bounds with vehicle footprint clearance...")
    d_min, d_max = build_corridor_bounds(s)
    
    print("3. Solving Matrix QP for optimal lateral trajectory d(s)...")
    d_opt = solve_frenet_trajectory_qp(s, d_min, d_max)
    
    print("4. Solving smooth convex longitudinal velocity optimization...")
    d_prime = np.gradient(d_opt, s[1] - s[0])
    d_double_prime = np.gradient(d_prime, s[1] - s[0])
    kappa_total = kappa_ref + d_double_prime
    
    dyn_res = solve_smooth_longitudinal_profile(s, kappa_total)
    
    fig = plot_frenet_trajectory_results(
        s, kappa_ref, theta_ref, x_ref, y_ref,
        d_opt, d_min, d_max, dyn_res
    )
    
    print(f"\n[Result] Peak lateral displacement: {np.max(d_opt):.2f}m (Corridor boundary: 2.00m, Obstacle top: 0.60m)")
    print(f"[Result] Max Longitudinal Accel: {np.max(np.abs(dyn_res['a_long'])):.2f} m/s²")
    print(f"[Result] Max Longitudinal Jerk: {np.max(np.abs(dyn_res['j_long'])):.2f} m/s³")
    print(f"[Result] Max Lateral Accel: {np.max(np.abs(dyn_res['a_lat'])):.2f} m/s² (Comfort limit: 2.00 m/s²)")
    print("[Done] Script executed cleanly. 8-panel dashboard ready.")
