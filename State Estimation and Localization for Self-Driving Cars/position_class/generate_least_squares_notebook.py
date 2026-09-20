"""Generates both student (scaffolded TODOs) and instructor (solution) notebooks for least_squares.py."""

import nbformat as nbf

def create_notebook(student_mode: bool = False):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"language_info": {"name": "python"}}

    title_suffix = " (Student Exercise Handout)" if student_mode else " (Complete Solution Guide)"
    
    if student_mode:
        task0_code = """def solve_least_squares(H: np.ndarray, y: np.ndarray, R: np.ndarray = None):
    \"\"\"Solves the Batch / Weighted Least Squares Normal Equations.
    
    Args:
        H: (m, n) Regressor / Observation matrix.
        y: (m, 1) or (m,) Observation vector.
        R: Optional (m, m) measurement noise covariance matrix.
        
    Returns:
        x_hat: (n, 1) parameter estimate vector.
        P: (n, n) parameter error covariance matrix.
        residuals: (m, 1) error vector e = y - H*x_hat.
    \"\"\"
    H_mat = np.asarray(H, dtype=np.float64)
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    m, n = H_mat.shape
    
    # -------------------------------------------------------------------------
    # TODO 0.1: Implement Normal Equations
    # If R is None:
    #     P = (H^T H)^-1
    #     x_hat = P @ H^T @ y
    # Else:
    #     P = (H^T R^-1 H)^-1
    #     x_hat = P @ H^T @ R^-1 @ y
    # -------------------------------------------------------------------------
    x_hat = None      # <-- TODO: YOUR CODE HERE
    P = None          # <-- TODO: YOUR CODE HERE
    residuals = None  # <-- TODO: YOUR CODE HERE
    
    return x_hat, P, residuals
"""
        task1_code = """# 1. Load Sensor Data
current_measurements = np.array([0.2, 0.3, 0.4, 0.5, 0.6])      # Amperes
voltage_measurements = np.array([1.23, 1.38, 2.06, 2.47, 3.17])  # Volts

# -----------------------------------------------------------------------------
# TODO 1.1: Build Observation Matrix H and Measurement Vector y
# Model: V_i = I_i * R + v_i
# -----------------------------------------------------------------------------
H_resistor = None  # <-- TODO: YOUR CODE HERE (Hint: current_measurements.reshape(-1, 1))
y_resistor = None  # <-- TODO: YOUR CODE HERE (Hint: voltage_measurements.reshape(-1, 1))

# TODO 1.2: Solve for Estimated Resistance R_hat and Covariance P
x_resistor, P_resistor, res_resistor = solve_least_squares(H_resistor, y_resistor)

if x_resistor is not None:
    R_est = float(x_resistor[0, 0])
    R_sigma = float(np.sqrt(P_resistor[0, 0]))
    print(f"Estimated Resistance R: {R_est:.3f} ± {3*R_sigma:.3f} Ohms (3-sigma bound)")
    
    # Plotly Interactive Fit
    I_line = np.linspace(0, 0.7, 50)
    V_line = R_est * I_line
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=current_measurements, y=voltage_measurements, mode='markers', name='Sensor Data', marker=dict(size=10, color='red')))
    fig1.add_trace(go.Scatter(x=I_line, y=V_line, mode='lines', name=f'Least Squares Fit (R = {R_est:.2f} Ω)', line=dict(color='blue', width=2)))
    fig1.update_layout(title="<b>Problem 1: Ohm's Law Resistor Calibration</b>", xaxis_title="Current I [A]", yaxis_title="Voltage V [V]")
    fig1.show()
"""
        task2_code = """# 1. Load Odometry & GPS Data
omega_encoder = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])         # rad/s
v_gps_speed   = np.array([3.28, 6.64, 9.87, 13.25, 16.48, 19.82])       # m/s
sigma_gps = 0.05                                                         # 5 cm/s GPS speed accuracy

# -----------------------------------------------------------------------------
# TODO 2.1: Build H, y, and Noise Covariance Matrix R
# Model: v_gps = r_eff * omega + v
# -----------------------------------------------------------------------------
H_wheel = None  # <-- TODO: YOUR CODE HERE
y_wheel = None  # <-- TODO: YOUR CODE HERE
R_wheel = None  # <-- TODO: YOUR CODE HERE (Hint: np.eye(len(omega_encoder)) * (sigma_gps**2))

# TODO 2.2: Solve Weighted Least Squares
x_wheel, P_wheel, res_wheel = solve_least_squares(H_wheel, y_wheel, R=R_wheel)

if x_wheel is not None:
    r_eff_est = float(x_wheel[0, 0])
    r_eff_sigma = float(np.sqrt(P_wheel[0, 0]))
    print(f"Calibrated Effective Wheel Radius: {r_eff_est:.4f} ± {3*r_eff_sigma:.4f} m (3-sigma bound)")
    
    # Plotly Calibration Curve
    omega_line = np.linspace(0, 70, 50)
    v_line = r_eff_est * omega_line
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=omega_encoder, y=v_gps_speed, mode='markers', name='Wheel Speed vs GPS', marker=dict(size=10, color='crimson')))
    fig2.add_trace(go.Scatter(x=omega_line, y=v_line, mode='lines', name=f'Calibrated Model (r_eff = {r_eff_est:.4f} m)', line=dict(color='green', width=2)))
    fig2.update_layout(title="<b>Problem 2: Wheel Odometry Effective Radius Calibration</b>", xaxis_title="Wheel Angular Velocity ω [rad/s]", yaxis_title="GPS Linear Speed v [m/s]")
    fig2.show()
"""
        task3_code = """# 1. Load Radar Ping Data
timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])                   # seconds
measured_distances = np.array([10.2, 22.1, 34.5, 46.8, 58.9, 71.3])       # meters

# -----------------------------------------------------------------------------
# TODO 3.1: Build Regressor Matrix H and Measurement Vector y
# Model: d_i = p0 + v0 * t_i + v_i
# -----------------------------------------------------------------------------
H_kinematics = None  # <-- TODO: YOUR CODE HERE (Hint: np.column_stack([np.ones_like(timestamps), timestamps]))
y_kinematics = None  # <-- TODO: YOUR CODE HERE

# TODO 3.2: Solve Least Squares
x_kinematics, P_kinematics, res_kinematics = solve_least_squares(H_kinematics, y_kinematics)

if x_kinematics is not None:
    p0_est, v0_est = x_kinematics.flatten()
    p0_sigma = np.sqrt(P_kinematics[0, 0])
    v0_sigma = np.sqrt(P_kinematics[1, 1])
    print(f"Estimated Initial Position p0: {p0_est:.3f} ± {3*p0_sigma:.3f} m")
    print(f"Estimated Velocity v0:         {v0_est:.3f} ± {3*v0_sigma:.3f} m/s")
    
    t_line = np.linspace(0, 6, 50)
    d_line = p0_est + v0_est * t_line
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=timestamps, y=measured_distances, mode='markers', name='Radar Distance Pings', marker=dict(size=10, color='darkorange')))
    fig3.add_trace(go.Scatter(x=t_line, y=d_line, mode='lines', name=f'Fitted Trajectory (p0={p0_est:.1f}m, v0={v0_est:.1f}m/s)', line=dict(color='indigo', width=2)))
    fig3.update_layout(title="<b>Problem 3: 1D Vehicle Initial State Estimation</b>", xaxis_title="Time t [s]", yaxis_title="Distance d [m]")
    fig3.show()
"""
        task4_code = """# 1. Load Synthetic LiDAR Point Cloud Returns
np.random.seed(42)
N_pts = 200
x_lidar = np.random.uniform(2.0, 25.0, N_pts)    # Forward range (m)
y_lidar = np.random.uniform(-4.0, 4.0, N_pts)    # Lateral range (m)
sigma_lidar = 0.03                               # 3 cm LiDAR ranging noise

a_true, b_true, c_true = 0.02, -0.01, -1.70
z_lidar = a_true * x_lidar + b_true * y_lidar + c_true + np.random.normal(0, sigma_lidar, N_pts)

# -----------------------------------------------------------------------------
# TODO 4.1: Build Regressor Matrix H, y, and Noise Covariance R
# Model: z_i = a * x_i + b * y_i + c + v_i
# -----------------------------------------------------------------------------
H_lidar = None      # <-- TODO: YOUR CODE HERE (Hint: np.column_stack([x_lidar, y_lidar, np.ones(N_pts)]))
y_lidar_vec = None  # <-- TODO: YOUR CODE HERE
R_lidar = None      # <-- TODO: YOUR CODE HERE

# TODO 4.2: Solve Weighted Least Squares
x_lidar_hat, P_lidar, res_lidar = solve_least_squares(H_lidar, y_lidar_vec, R=R_lidar)

if x_lidar_hat is not None:
    a_hat, b_hat, c_hat = x_lidar_hat.flatten()
    
    # -------------------------------------------------------------------------
    # TODO 4.3: Extract Physical Automotive Quantities
    # -------------------------------------------------------------------------
    norm_factor = np.sqrt(a_hat**2 + b_hat**2 + 1.0)
    normal_vector = np.array([-a_hat, -b_hat, 1.0]) / norm_factor
    sensor_height_m = abs(c_hat) / norm_factor
    pitch_deg = float(np.degrees(np.arctan(a_hat)))
    roll_deg = float(np.degrees(np.arctan(-b_hat)))
    
    print("=== LiDAR Ground Plane Estimation Results ===")
    print(f"Fitted Model:       z = {a_hat:.5f}*x + {b_hat:.5f}*y + {c_hat:.4f}")
    print(f"Road Pitch Angle:   {pitch_deg:.3f}° (True: {np.degrees(np.arctan(a_true)):.3f}°)")
    print(f"Road Roll Angle:    {roll_deg:.3f}° (True: {np.degrees(np.arctan(-b_true)):.3f}°)")
    print(f"LiDAR Sensor Height:{sensor_height_m:.3f} m (True: 1.700 m)")
    print(f"Plane Unit Normal:  {normal_vector}")
    
    xx, yy = np.meshgrid(np.linspace(2.0, 25.0, 15), np.linspace(-4.0, 4.0, 10))
    zz = a_hat * xx + b_hat * yy + c_hat
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter3d(x=x_lidar, y=y_lidar, z=z_lidar, mode='markers', marker=dict(size=3, color=z_lidar, colorscale='Turbo', opacity=0.8), name='LiDAR Ground Points'))
    fig4.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.6, colorscale='Viridis', showscale=False, name='Estimated Road Plane'))
    fig4.update_layout(title="<b>Problem 4: 3D LiDAR Road Surface Ground Plane Fitting</b>", scene=dict(xaxis_title="X [m] (Forward)", yaxis_title="Y [m] (Lateral)", zaxis_title="Z [m] (Elevation)"), margin=dict(l=0, r=0, b=0, t=40))
    fig4.show()
"""
        task5_code = """# =============================================================================
# Problem 5: Recursive Least Squares (RLS) Initialization & Streaming Update
# =============================================================================

# Streaming timestamps and radar pings
t_stream = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
d_stream = np.array([10.2, 22.1, 34.5, 46.8, 58.9, 71.3])
sigma_meas = 1.0

# -----------------------------------------------------------------------------
# TODO 5.1: Method 1 (Batch BLUE Prior Initialization from first 2 samples)
# Compute x0 and P0 from t[:2] and d[:2]
# H0 = [[1, 0], [1, 1]], y0 = [10.2, 22.1]^T
# -----------------------------------------------------------------------------
H0 = None  # <-- TODO: YOUR CODE HERE
y0 = None  # <-- TODO: YOUR CODE HERE
R0 = None  # <-- TODO: YOUR CODE HERE

# Compute x0 = (H0^T R0^-1 H0)^-1 H0^T R0^-1 y0 and P0 = (H0^T R0^-1 H0)^-1
x0_batch, P0_batch, _ = solve_least_squares(H0, y0, R=R0) if H0 is not None else (None, None, None)

# -----------------------------------------------------------------------------
# TODO 5.2: Implement the Recursive Loop for remaining streaming samples (k=2, 3, 4, 5)
# For each sample k:
#   1. nu_k = y_k - H_k @ x_hat
#   2. S_k  = H_k @ P @ H_k.T + R_k
#   3. K_k  = P @ H_k.T @ inv(S_k)
#   4. x_hat = x_hat + K_k @ nu_k
#   5. P    = (I - K_k @ H_k) @ P
# -----------------------------------------------------------------------------
if x0_batch is not None:
    x_hat = x0_batch.copy()
    P_mat = P0_batch.copy()
    history = [x_hat.flatten()]
    
    for k in range(2, len(t_stream)):
        H_k = np.array([[1.0, t_stream[k]]])
        y_k = np.array([[d_stream[k]]])
        R_k = np.array([[sigma_meas ** 2]])
        
        # <-- TODO: COMPLETE RECURSIVE UPDATE HERE
        nu_k = y_k - H_k @ x_hat
        S_k = H_k @ P_mat @ H_k.T + R_k
        K_k = P_mat @ H_k.T @ np.linalg.inv(S_k)
        x_hat = x_hat + K_k @ nu_k
        P_mat = (np.eye(2) - K_k @ H_k) @ P_mat
        history.append(x_hat.flatten())
        
    history = np.array(history)
    print("=== RLS Estimated Parameters Over Time ===")
    for idx, (p_val, v_val) in enumerate(history):
        print(f"Step k={idx+2}: p0 = {p_val:.3f} m, v0 = {v_val:.3f} m/s")
"""
    else:
        # Full Instructor Solution Code
        task0_code = """def solve_least_squares(H: np.ndarray, y: np.ndarray, R: np.ndarray = None):
    \"\"\"Solves the Batch / Weighted Least Squares Normal Equations.
    
    Args:
        H: (m, n) Regressor / Observation matrix.
        y: (m, 1) or (m,) Observation vector.
        R: Optional (m, m) measurement noise covariance matrix.
        
    Returns:
        x_hat: (n, 1) parameter estimate vector.
        P: (n, n) parameter error covariance matrix.
        residuals: (m, 1) error vector e = y - H*x_hat.
    \"\"\"
    H_mat = np.asarray(H, dtype=np.float64)
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    m, n = H_mat.shape
    
    if R is None:
        P = np.linalg.inv(H_mat.T @ H_mat)
        x_hat = P @ H_mat.T @ y_vec
    else:
        R_mat = np.asarray(R, dtype=np.float64)
        if R_mat.ndim == 1:
            R_mat = np.diag(R_mat)
        R_inv = np.linalg.inv(R_mat)
        P = np.linalg.inv(H_mat.T @ R_inv @ H_mat)
        x_hat = P @ H_mat.T @ R_inv @ y_vec
        
    residuals = y_vec - H_mat @ x_hat
    return x_hat, P, residuals

# Test Task 0
H_test = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
y_test = np.array([1.0, 3.0, 5.0, 7.0])
x_test, P_test, res_test = solve_least_squares(H_test, y_test)

assert np.allclose(x_test.flatten(), [1.0, 2.0]), "Task 0 solver failed basic test!"
print(" Task 0 Solver Verified!")
"""
        task1_code = """# 1. Load Sensor Data
current_measurements = np.array([0.2, 0.3, 0.4, 0.5, 0.6])      # Amperes
voltage_measurements = np.array([1.23, 1.38, 2.06, 2.47, 3.17])  # Volts

H_resistor = current_measurements.reshape(-1, 1)
y_resistor = voltage_measurements.reshape(-1, 1)

x_resistor, P_resistor, res_resistor = solve_least_squares(H_resistor, y_resistor)
R_est = float(x_resistor[0, 0])
R_sigma = float(np.sqrt(P_resistor[0, 0]))

print(f"Estimated Resistance R: {R_est:.3f} ± {3*R_sigma:.3f} Ohms (3-sigma bound)")
assert np.isclose(R_est, 5.0, atol=0.5), "Resistance estimate is outside expected range!"
print(" Problem 1 Passed!")

# Plotly Interactive Fit
I_line = np.linspace(0, 0.7, 50)
V_line = R_est * I_line

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=current_measurements, y=voltage_measurements, mode='markers', name='Sensor Data (V vs I)', marker=dict(size=10, color='red')))
fig1.add_trace(go.Scatter(x=I_line, y=V_line, mode='lines', name=f'Least Squares Fit (R = {R_est:.2f} Ω)', line=dict(color='blue', width=2)))
fig1.update_layout(title="<b>Problem 1: Ohm's Law Resistor Calibration</b>", xaxis_title="Current I [A]", yaxis_title="Voltage V [V]")
fig1.show()
"""
        task2_code = """# 1. Load Odometry & GPS Data
omega_encoder = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])         # rad/s
v_gps_speed   = np.array([3.28, 6.64, 9.87, 13.25, 16.48, 19.82])       # m/s
sigma_gps = 0.05                                                         # 5 cm/s GPS speed accuracy

H_wheel = omega_encoder.reshape(-1, 1)
y_wheel = v_gps_speed.reshape(-1, 1)
R_wheel = np.eye(len(omega_encoder)) * (sigma_gps ** 2)

x_wheel, P_wheel, res_wheel = solve_least_squares(H_wheel, y_wheel, R=R_wheel)
r_eff_est = float(x_wheel[0, 0])
r_eff_sigma = float(np.sqrt(P_wheel[0, 0]))

print(f"Calibrated Effective Wheel Radius: {r_eff_est:.4f} ± {3*r_eff_sigma:.4f} m (3-sigma bound)")
assert np.isclose(r_eff_est, 0.33, atol=0.01), "Wheel radius calibration failed!"
print(" Problem 2 Passed!")

omega_line = np.linspace(0, 70, 50)
v_line = r_eff_est * omega_line

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=omega_encoder, y=v_gps_speed, mode='markers', name='Wheel Speed vs GPS', marker=dict(size=10, color='crimson')))
fig2.add_trace(go.Scatter(x=omega_line, y=v_line, mode='lines', name=f'Calibrated Model (r_eff = {r_eff_est:.4f} m)', line=dict(color='green', width=2)))
fig2.update_layout(title="<b>Problem 2: Wheel Odometry Effective Radius Calibration</b>", xaxis_title="Wheel Angular Velocity ω [rad/s]", yaxis_title="GPS Linear Speed v [m/s]")
fig2.show()
"""
        task3_code = """# 1. Load Radar Ping Data
timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])                   # seconds
measured_distances = np.array([10.2, 22.1, 34.5, 46.8, 58.9, 71.3])       # meters

H_kinematics = np.column_stack([np.ones_like(timestamps), timestamps])
y_kinematics = measured_distances.reshape(-1, 1)

x_kinematics, P_kinematics, res_kinematics = solve_least_squares(H_kinematics, y_kinematics)
p0_est, v0_est = x_kinematics.flatten()
p0_sigma = np.sqrt(P_kinematics[0, 0])
v0_sigma = np.sqrt(P_kinematics[1, 1])

rho_p0_v0 = P_kinematics[0, 1] / np.sqrt(P_kinematics[0, 0] * P_kinematics[1, 1])

print(f"Estimated Initial Position p0: {p0_est:.3f} ± {3*p0_sigma:.3f} m")
print(f"Estimated Velocity v0:         {v0_est:.3f} ± {3*v0_sigma:.3f} m/s")
print(f"Parameter Correlation (p0, v0): {rho_p0_v0:.4f}")

assert np.isclose(p0_est, 10.0, atol=0.5) and np.isclose(v0_est, 12.0, atol=0.5), "Kinematics estimation failed!"
print(" Problem 3 Passed!")

t_line = np.linspace(0, 6, 50)
d_line = p0_est + v0_est * t_line

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=timestamps, y=measured_distances, mode='markers', name='Radar Distance Pings', marker=dict(size=10, color='darkorange')))
fig3.add_trace(go.Scatter(x=t_line, y=d_line, mode='lines', name=f'Fitted Trajectory (p0={p0_est:.1f}m, v0={v0_est:.1f}m/s)', line=dict(color='indigo', width=2)))
fig3.update_layout(title="<b>Problem 3: 1D Vehicle Initial State Estimation</b>", xaxis_title="Time t [s]", yaxis_title="Distance d [m]")
fig3.show()
"""
        task4_code = """# 1. Load Synthetic LiDAR Point Cloud Returns
np.random.seed(42)
N_pts = 200
x_lidar = np.random.uniform(2.0, 25.0, N_pts)    # Forward range (m)
y_lidar = np.random.uniform(-4.0, 4.0, N_pts)    # Lateral range (m)
sigma_lidar = 0.03                               # 3 cm LiDAR ranging noise

a_true, b_true, c_true = 0.02, -0.01, -1.70
z_lidar = a_true * x_lidar + b_true * y_lidar + c_true + np.random.normal(0, sigma_lidar, N_pts)

H_lidar = np.column_stack([x_lidar, y_lidar, np.ones(N_pts)])
y_lidar_vec = z_lidar.reshape(-1, 1)
R_lidar = np.eye(N_pts) * (sigma_lidar ** 2)

x_lidar_hat, P_lidar, res_lidar = solve_least_squares(H_lidar, y_lidar_vec, R=R_lidar)
a_hat, b_hat, c_hat = x_lidar_hat.flatten()

norm_factor = np.sqrt(a_hat**2 + b_hat**2 + 1.0)
normal_vector = np.array([-a_hat, -b_hat, 1.0]) / norm_factor
sensor_height_m = abs(c_hat) / norm_factor
pitch_deg = float(np.degrees(np.arctan(a_hat)))
roll_deg = float(np.degrees(np.arctan(-b_hat)))

print("=== LiDAR Ground Plane Estimation Results ===")
print(f"Fitted Model:       z = {a_hat:.5f}*x + {b_hat:.5f}*y + {c_hat:.4f}")
print(f"Road Pitch Angle:   {pitch_deg:.3f}° (True: {np.degrees(np.arctan(a_true)):.3f}°)")
print(f"Road Roll Angle:    {roll_deg:.3f}° (True: {np.degrees(np.arctan(-b_true)):.3f}°)")
print(f"LiDAR Sensor Height:{sensor_height_m:.3f} m (True: 1.700 m)")
print(f"Plane Unit Normal:  {normal_vector}")

assert np.isclose(a_hat, 0.02, atol=0.01), "Pitch slope estimate failed!"
assert np.isclose(b_hat, -0.01, atol=0.01), "Roll slope estimate failed!"
assert np.isclose(sensor_height_m, 1.70, atol=0.05), "Sensor height estimation failed!"
print(" Problem 4 Passed!")

xx, yy = np.meshgrid(np.linspace(2.0, 25.0, 15), np.linspace(-4.0, 4.0, 10))
zz = a_hat * xx + b_hat * yy + c_hat

fig4 = go.Figure()
fig4.add_trace(go.Scatter3d(
    x=x_lidar, y=y_lidar, z=z_lidar,
    mode='markers',
    marker=dict(size=3, color=z_lidar, colorscale='Turbo', opacity=0.8),
    name='LiDAR Ground Points'
))
fig4.add_trace(go.Surface(
    x=xx, y=yy, z=zz,
    opacity=0.6,
    colorscale='Viridis',
    showscale=False,
    name='Estimated Road Plane'
))
fig4.update_layout(
    title="<b>Problem 4: 3D LiDAR Road Surface Ground Plane Fitting (ISO 8855 Frame)</b>",
    scene=dict(
        xaxis_title="X [m] (Forward / Longitudinal)",
        yaxis_title="Y [m] (Lateral / Left)",
        zaxis_title="Z [m] (Elevation / Up)"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)
fig4.show()
"""
        task5_code = """# =============================================================================
# Problem 5: Recursive Least Squares (RLS) - Three Initialization Strategies
# =============================================================================
from position_class import RecursiveLeastSquares, BatchLeastSquares

# -----------------------------------------------------------------------------
# Strategy 1: Datasheet Prior Initialization (Precision Current Shunt Resistor)
# Datasheet Specs: R_nom = 5.0 Ohms, Tolerance = +/- 10% (0.50 Ohms)
# -----------------------------------------------------------------------------
R_datasheet_nom = 5.0
delta_R = 0.50  # 10% of 5.0 Ohms

# 3-Sigma Rule from Datasheet: sigma_0 = delta_R / 3.0
x0_datasheet = np.array([R_datasheet_nom])
P0_datasheet = np.array([[(delta_R / 3.0) ** 2]])

print("=== Strategy 1: Datasheet Prior Initialization ===")
print(f"Datasheet State: R0 = {x0_datasheet[0]:.2f} Ω")
print(f"Datasheet Cov:   P0 = {P0_datasheet[0,0]:.5f} Ω² (σ_0 = {np.sqrt(P0_datasheet[0,0]):.4f} Ω)")

# Streaming experimental data (True R = 5.15 Ohms)
currents_stream = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
voltages_stream = np.array([1.06, 1.57, 2.08, 2.59, 3.11])
sigma_volt = 0.05

rls_datasheet = RecursiveLeastSquares(n=1, x0=x0_datasheet, P0=P0_datasheet)
for k in range(len(currents_stream)):
    res = rls_datasheet.update(H_k=np.array([[currents_stream[k]]]), y_k=voltages_stream[k], R_k=sigma_volt**2)
    print(f"Sample {k+1} (I={currents_stream[k]}A, V={voltages_stream[k]:.2f}V): R_est = {res.x_hat[0,0]:.3f} Ω, P = {res.covariance[0,0]:.5f}")

# -----------------------------------------------------------------------------
# Strategy 2: Batch Prior Initialization (1D Vehicle Kinematics)
# -----------------------------------------------------------------------------
t_stream = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
d_stream = np.array([10.2, 22.1, 34.5, 46.8, 58.9, 71.3])
sigma_meas = 1.0

H0 = np.column_stack([np.ones(2), t_stream[:2]])
y0 = d_stream[:2]
R0 = np.eye(2) * (sigma_meas ** 2)

rls_batch = RecursiveLeastSquares(n=2)
x0_batch, P0_batch = rls_batch.initialize_from_batch(H0, y0, R0)

history_batch = [x0_batch.flatten()]
cov_p0_batch = [P0_batch[0, 0]]
cov_v0_batch = [P0_batch[1, 1]]

for k in range(2, len(t_stream)):
    H_k = np.array([[1.0, t_stream[k]]])
    y_k = d_stream[k]
    res = rls_batch.update(H_k, y_k, R_k=sigma_meas**2)
    history_batch.append(res.x_hat.flatten())
    cov_p0_batch.append(res.covariance[0, 0])
    cov_v0_batch.append(res.covariance[1, 1])

# -----------------------------------------------------------------------------
# Strategy 3: Diffuse Prior Initialization (P0 = 1000 * I)
# -----------------------------------------------------------------------------
rls_diffuse = RecursiveLeastSquares(n=2, x0=np.zeros(2), P0=np.eye(2)*1000.0)
history_diffuse = []
for k in range(len(t_stream)):
    H_k = np.array([[1.0, t_stream[k]]])
    y_k = d_stream[k]
    res = rls_diffuse.update(H_k, y_k, R_k=sigma_meas**2)
    history_diffuse.append(res.x_hat.flatten())

# Ground Truth Batch Solve
full_H = np.column_stack([np.ones_like(t_stream), t_stream])
batch_solver = BatchLeastSquares(R=np.eye(len(t_stream))*(sigma_meas**2))
full_batch_res = batch_solver.fit(full_H, d_stream)

print("\\n=== Final Convergence Comparison ===")
print(f"Full Batch Solve:     p0 = {full_batch_res.x_hat[0,0]:.3f} m,  v0 = {full_batch_res.x_hat[1,0]:.3f} m/s")
print(f"RLS (Batch Init):     p0 = {history_batch[-1][0]:.3f} m,  v0 = {history_batch[-1][1]:.3f} m/s")
print(f"RLS (Diffuse Init):   p0 = {history_diffuse[-1][0]:.3f} m,  v0 = {history_diffuse[-1][1]:.3f} m/s")

# Plot Parameter Convergence and Covariance Shrinkage
fig5 = make_subplots(rows=2, cols=1, subplot_titles=("<b>Recursive State Convergence (p0, v0)</b>", "<b>Covariance Contraction (P_k Shrinkage)</b>"))

steps_batch = list(range(2, len(t_stream) + 1))
history_batch = np.array(history_batch)

fig5.add_trace(go.Scatter(x=steps_batch, y=history_batch[:, 0], mode='lines+markers', name='Estimated p0 (m)', line=dict(color='blue', width=2)), row=1, col=1)
fig5.add_trace(go.Scatter(x=steps_batch, y=history_batch[:, 1], mode='lines+markers', name='Estimated v0 (m/s)', line=dict(color='green', width=2)), row=1, col=1)
fig5.add_hline(y=full_batch_res.x_hat[0,0], line_dash='dash', line_color='blue', annotation_text='Batch p0', row=1, col=1)
fig5.add_hline(y=full_batch_res.x_hat[1,0], line_dash='dash', line_color='green', annotation_text='Batch v0', row=1, col=1)

fig5.add_trace(go.Scatter(x=steps_batch, y=cov_p0_batch, mode='lines+markers', name='Var(p0)', line=dict(color='purple', dash='dot')), row=2, col=1)
fig5.add_trace(go.Scatter(x=steps_batch, y=cov_v0_batch, mode='lines+markers', name='Var(v0)', line=dict(color='orange', dash='dot')), row=2, col=1)

fig5.update_layout(title="<b>Problem 5: Recursive Least Squares (RLS) Parameter Convergence & Uncertainty Shrinkage</b>", height=650)
fig5.show()
"""

    cells = [
        nbf.v4.new_markdown_cell(
f"""# 🚗 Laboratory Exercise: Batch & Recursive Least Squares Parameter Estimation{title_suffix}
**Course:** State Estimation and Localization for Self-Driving Cars  
**Module:** `position_class.least_squares`  
**Focus:** Hands-on Matrix Operations, Sensor Calibration, LiDAR Ground Plane Fitting, and RLS Initialization

---

## 🎯 Laboratory Objectives
In this hands-on lab, you will:
1. Implement the **Weighted Batch Least Squares Normal Equations** from scratch using NumPy matrix operations (`@`, `.T`, `np.linalg.inv`).
2. Calibrate sensor models across **five real-world autonomous vehicle applications**:
   * **Problem 1 (Ohm's Law):** Current sensor resistor calibration ($V = I \\cdot R$).
   * **Problem 2 (Wheel Odometry):** Effective tire radius calibration ($v_{{\\text{{gps}}}} = r_{{\\text{{eff}}}} \\cdot \\omega$).
   * **Problem 3 (Vehicle Kinematics):** Initial position and velocity estimation ($d_i = p_0 + v_0 t_i$).
   * **Problem 4 (Perception):** 3D LiDAR Road Surface Ground Plane fitting ($z_i = a x_i + b y_i + c$).
   * **Problem 5 (Recursive Estimation):** Initializing $\\hat{{\\mathbf{{x}}}}_0$ and $\\mathbf{{P}}_0$ in **Recursive Least Squares (RLS)** via by-hand batch BLUE and streaming online updates.
3. Derive physical vehicle metrics (road slope pitch $\\theta$, road camber roll $\\phi$, sensor mounting height $h$, and normal vector $\\hat{{\\mathbf{{n}}}}$).
4. Compute parameter error covariance matrices $\\mathbf{{P}} = (\\mathbf{{H}}^T \\mathbf{{R}}^{{-1}} \\mathbf{{H}})^{{-1}}$ and $3\\sigma$ confidence bounds.
5. Create interactive visualizations using **Plotly** (2D scatter/error bounds and 3D point cloud surfaces).
"""
        ),
        nbf.v4.new_code_cell(
"""import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("Environment initialized! Ready to build Least Squares estimators.")
"""
        ),
        nbf.v4.new_markdown_cell(
r"""---
## 📐 Mathematical Formulation: Best Linear Unbiased Estimator (BLUE)

Given a linear observation model:
$$\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{v}, \quad \mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$$

Where:
* $\mathbf{x} \in \mathbb{R}^n$: Unknown state / parameter vector to estimate.
* $\mathbf{y} \in \mathbb{R}^m$: Measurement vector from sensors.
* $\mathbf{H} \in \mathbb{R}^{m \times n}$: Regressor / Observation matrix mapping state space to measurement space.
* $\mathbf{R} \in \mathbb{R}^{m \times m}$: Measurement noise covariance matrix.

### Normal Equations:
$$\hat{\mathbf{x}} = \left(\mathbf{H}^T \mathbf{R}^{-1} \mathbf{H}\right)^{-1} \mathbf{H}^T \mathbf{R}^{-1} \mathbf{y}$$

### Posterior Error Covariance:
$$\mathbf{P} = \mathbb{E}[(\hat{\mathbf{x}} - \mathbf{x})(\hat{\mathbf{x}} - \mathbf{x})^T] = \left(\mathbf{H}^T \mathbf{R}^{-1} \mathbf{H}\right)^{-1}$$
"""
        ),
        nbf.v4.new_markdown_cell(
r"""---
## 💻 Task 0: Generic Batch & Weighted Least Squares Solver

Complete the function below to solve the normal equations for any arbitrary $\mathbf{H}$, $\mathbf{y}$, and optional $\mathbf{R}$.
"""
        ),
        nbf.v4.new_code_cell(task0_code),
        nbf.v4.new_markdown_cell(
r"""---
## 🔹 Problem 1: Current Sensor Resistor Calibration (Ohm's Law)

**Scenario:** We need to calibrate a precision shunt resistor $R$ on a battery management system using current $I_i$ and voltage $V_i$ measurements.

**Physics Model:**
$$V_i = I_i \cdot R + v_i \implies \mathbf{y} = \begin{bmatrix} V_1 \\ V_2 \\ \vdots \\ V_N \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} I_1 \\ I_2 \\ \vdots \\ I_N \end{bmatrix}, \quad \mathbf{x} = [R]$$
"""
        ),
        nbf.v4.new_code_cell(task1_code),
        nbf.v4.new_markdown_cell(
r"""---
## 🔹 Problem 2: Wheel Odometry Effective Radius Calibration ($r_{\text{eff}}$)

**Scenario:** As tire pressure and tread wear change, the effective rolling radius $r_{\text{eff}}$ of autonomous vehicle wheels varies. We calibrate $r_{\text{eff}}$ by comparing wheel encoder angular velocity $\omega_i$ with GPS reference speed $v_{\text{gps}, i}$.

**Physics Model:**
$$v_{\text{gps}, i} = r_{\text{eff}} \cdot \omega_i + v_i, \quad v_i \sim \mathcal{N}(0, \sigma_{\text{gps}}^2)$$
"""
        ),
        nbf.v4.new_code_cell(task2_code),
        nbf.v4.new_markdown_cell(
r"""---
## 🔹 Problem 3: 1D Vehicle Kinematics ($p_0, v_0$ Initial State Estimation)

**Scenario:** A trackside radar station receives distance pings $d_i$ to a vehicle at known timestamps $t_i$. We want to estimate both the initial position $p_0$ and constant velocity $v_0$.

**Physics Model:**
$$d_i = p_0 + v_0 \cdot t_i + v_i \implies \mathbf{y} = \begin{bmatrix} d_1 \\ \vdots \\ d_N \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 1 & t_1 \\ \vdots & \vdots \\ 1 & t_N \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} p_0 \\ v_0 \end{bmatrix}$$
"""
        ),
        nbf.v4.new_code_cell(task3_code),
        nbf.v4.new_markdown_cell(
r"""---
## 🔹 Problem 4: 3D LiDAR Road Surface Ground Plane Fitting

**Scenario:** Our vehicle needs to fit a 3D road surface plane $z = ax + by + c$ from point cloud returns ahead of the vehicle (ISO 8855 Frame: $+X$ Forward, $+Y$ Left, $+Z$ Up).

**Physics Model:**
$$z_i = a x_i + b y_i + c + v_i \implies \mathbf{H} = \begin{bmatrix} x_1 & y_1 & 1 \\ \vdots & \vdots & \vdots \\ x_N & y_N & 1 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} a \\ b \\ c \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} z_1 \\ \vdots \\ z_N \end{bmatrix}$$

**Physical Quantities to Extract:**
* Road Pitch Angle: $\theta = \arctan(a)$
* Road Roll / Camber Angle: $\phi = \arctan(-b)$
* Sensor Mounting Height: $h = \frac{|c|}{\sqrt{a^2 + b^2 + 1}}$
* 3D Unit Normal Vector: $\hat{\mathbf{n}} = \frac{[-a, -b, 1]^T}{\sqrt{a^2 + b^2 + 1}}$
"""
        ),
        nbf.v4.new_code_cell(task4_code),
        nbf.v4.new_markdown_cell(
r"""---
## 🔹 Problem 5: Recursive Least Squares (RLS) - Initializing $\hat{\mathbf{x}}_0$ and $\mathbf{P}_0$

**The Core Question:** In streaming online estimation, how do we initialize $\hat{\mathbf{x}}_0$ and $\mathbf{P}_0$?

### 1. By-Hand Calculation of Initial $\hat{\mathbf{x}}_0$ and $\mathbf{P}_0$:
Using the first 2 observations ($t_1 = 0\text{ s}, d_1 = 10.0\text{ m}$ and $t_2 = 1\text{ s}, d_2 = 22.0\text{ m}$ with $\sigma_v = 1.0\text{ m}$):
$$\mathbf{H}_0 = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}, \quad \mathbf{y}_0 = \begin{bmatrix} 10.0 \\ 22.0 \end{bmatrix}$$

$$\mathbf{H}_0^T \mathbf{H}_0 = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix} \implies \mathbf{P}_0 = \sigma_v^2 (\mathbf{H}_0^T \mathbf{H}_0)^{-1} = \begin{bmatrix} 1.0 & -1.0 \\ -1.0 & 2.0 \end{bmatrix}$$
$$\hat{\mathbf{x}}_0 = \mathbf{P}_0 \mathbf{H}_0^T \mathbf{y}_0 = \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix} \begin{bmatrix} 32.0 \\ 22.0 \end{bmatrix} = \begin{bmatrix} 10.0\text{ m} \\ 12.0\text{ m/s} \end{bmatrix}$$

### 2. Streaming Recursive Update (Sample $k=3$ at $t_3=2\text{ s}, d_3=34.5\text{ m}$):
* Innovation: $\nu_3 = 34.5 - (10.0 + 12.0 \times 2) = +0.5\text{ m}$
* Innovation Covariance: $S_3 = \mathbf{H}_3 \mathbf{P}_0 \mathbf{H}_3^T + R_3 = 5.0 + 1.0 = 6.0$
* Estimator Gain: $\mathbf{K}_3 = \mathbf{P}_0 \mathbf{H}_3^T S_3^{-1} = [-0.1667, 0.5000]^T$
* Updated State: $\hat{\mathbf{x}}_1 = [9.9167\text{ m}, 12.2500\text{ m/s}]^T$
* Updated Covariance: $\mathbf{P}_1 = (\mathbf{I} - \mathbf{K}_3 \mathbf{H}_3) \mathbf{P}_0 = \begin{bmatrix} 0.8333 & -0.5000 \\ -0.5000 & 0.5000 \end{bmatrix}$
"""
        ),
        nbf.v4.new_code_cell(task5_code),
        nbf.v4.new_markdown_cell(
"""---
## 🏁 Summary & Key Takeaways

1. **The Measurement Matrix $\\mathbf{H}$:** Directly encodes physics and sensor placement geometry.
2. **The Covariance Matrix $\\mathbf{P}$:** Quantifies parameter uncertainty and cross-correlations $\\sigma_{ij}$.
3. **Weighting with $\\mathbf{R}^{-1}$:** Ensures noisy measurements receive less weight, achieving minimum variance (BLUE).
4. **Physical Transformations:** Mathematical regression parameters must always be translated into vehicle coordinates and angles (Pitch, Roll, Elevation).
5. **RLS Initialization:** Initializing with a deterministic batch solve on the first $n$ samples gives an exact, unbiased prior $(\\hat{\\mathbf{x}}_0, \\mathbf{P}_0)$ that converges perfectly to the full batch solution without initial transient errors.
"""
        )
    ]
    nb.cells = cells
    return nb

if __name__ == "__main__":
    # 1. Solution Guide Notebook
    nb_sol = create_notebook(student_mode=False)
    with open("least_squares_exercise.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb_sol, f)
    print("Generated least_squares_exercise.ipynb (Complete Solution)")

    # 2. Student Incomplete Notebook
    nb_stu = create_notebook(student_mode=True)
    with open("least_squares_exercise_student.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb_stu, f)
    print("Generated least_squares_exercise_student.ipynb (Student Exercise)")
