"""
Session 01 Practice: Frenet Frame (s, d) 2D Trajectory Optimization & Smooth Dynamic Profiling
----------------------------------------------------------------------------------------------
Student Lab Exercise

In this lab, you will implement:
1. Frenet Frame 2D Cartesian transformations.
2. Obstacle feasible corridor bounds accounting for vehicle width and safety buffer.
3. Convex Matrix Quadratic Program (QP) using finite difference operators (D1, D2, D3).
4. Smooth Convex Longitudinal Velocity Optimization on Kinetic Energy u(s) = v(s)^2.

Follow the TODO comments in the mathematical and definition functions below.
All plotting, reference road generation, and numerical solvers are already provided.
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
    [Pre-implemented] Generates a curved reference road centerline (s, kappa_ref, theta_ref, x_ref, y_ref).
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

    TODO (Student):
    1. Interpolate x_r, y_r, th_r from s_ref onto s.
    2. Compute x and y using the Frenet transformation formulas above.
    """
    # --- TODO: Compute x and y Cartesian coordinates ---
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
    
    TODO (Student):
    1. Initialize d_min to -half_width and d_max to +half_width.
    2. Compute safe_lateral_bound = obs_d_max + (vehicle_width / 2.0) + safety_buffer.
    3. In the obstacle region (obs_s_start - 3m to obs_s_end + 3m), restrict d_min to safe_lateral_bound.
    """
    half_width = road_width / 2.0
    
    # --- TODO: Initialize d_min and d_max ---
    d_min = np.full_like(s, -half_width)
    d_max = np.full_like(s, half_width)
    
    safe_lateral_bound = obs_d_max + (vehicle_width / 2.0) + safety_buffer
    
    # --- TODO: Apply obstacle corridor constraints ---
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

    TODO (Student):
    1. Fill D1 with 1st derivative coefficients: [-1/ds, 1/ds]
    2. Fill D2 with 2nd derivative coefficients: [1/ds^2, -2/ds^2, 1/ds^2]
    3. Fill D3 with 3rd derivative coefficients: [-1/ds^3, 3/ds^3, -3/ds^3, 1/ds^3]
    """
    D1 = np.zeros((N, N + 1))
    D2 = np.zeros((N - 1, N + 1))
    D3 = np.zeros((N - 2, N + 1))
    
    # --- TODO: Populate D1, D2, D3 ---
    for i in range(N):
        pass
        
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

    TODO (Student):
    1. Call build_differentiation_matrices to obtain D1, D2, D3.
    2. Formulate quadratic Hessian matrix Q:
       Q = 2 * (w_center * I + w_heading * D1^T D1 + w_curv * D2^T D2 + w_jerk * D3^T D3)
    3. Formulate linear vector q = -2 * w_center * d_ref.
    4. Implement cost_fun(d) and cost_grad(d).
    """
    N = len(s) - 1
    ds = s[1] - s[0]
    
    D1, D2, D3 = build_differentiation_matrices(N, ds)
    
    # --- TODO: Compute Q and q ---
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

    TODO (Student):
    1. Compute u_bound(s) = min(v_limit^2, a_lat_max / max(|kappa_total(s)|, 1e-5)).
    2. Construct Q_u using D1 and D2 differentiation matrices.
    3. Construct linear cost vector q_u targeting u_target = v_target^2.
    4. Solve using L-BFGS-B and recover optimal velocity v_opt = sqrt(u_opt).
    """
    N = len(s) - 1
    ds = s[1] - s[0]
    
    D1, D2, _ = build_differentiation_matrices(N, ds)
    
    # --- TODO: 1. Compute kinetic energy bounds ---
    v_bound = np.minimum(v_limit, np.sqrt(a_lat_max / np.maximum(np.abs(kappa_total), 1e-5)))
    u_bound = v_bound ** 2
    u_target = v_target ** 2
    
    # --- TODO: 2. Construct Q_u and q_u ---
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
    
    # [Pre-implemented] Time domain integration
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
    
    print("2. Constructing obstacle corridor bounds...")
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
    print("\n[Lab] Please complete TODO blocks to see full optimized obstacle avoidance.")
