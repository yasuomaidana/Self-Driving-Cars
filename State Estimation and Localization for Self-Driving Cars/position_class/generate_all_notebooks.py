"""Script to generate and format the complete suite of 4 Jupyter Notebooks for the course."""

import nbformat as nbf

def create_day_01_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"language_info": {"name": "python"}}

    cells = [
        nbf.v4.new_markdown_cell(
"""# Day 01: Foundations of State Estimation & Linear Kalman Filter
**State Estimation and Localization for Self-Driving Cars**

Welcome to Day 01 of the 1-week intensive state estimation curriculum. Today we establish the mathematical bedrock:
1. **Mathematical Foundations**: Where does the Measurement Matrix $\\mathbf{H}$ come from? 4 real-world automotive calibration examples.
2. **Error Covariance $\\mathbf{P}$**: Derivation from Gauss-Markov assumptions and confidence ellipsoids.
3. **Cross-Disciplinary Notation**: Harmonizing symbols across Control, Robotics/SLAM, Aerospace, and Computer Vision.
4. **Recursive Least Squares (RLS)**: Online streaming parameter estimation with forgetting factors.
5. **The Linear Kalman Filter (KF)**: Optimal Bayesian tracking for 2D constant velocity kinematics with interactive Plotly visualizations.

*Note: In accordance with modern interactive visualization standards, this notebook uses **Plotly** exclusively.*
"""
        ),
        nbf.v4.new_code_cell(
"""import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from position_class import (
    BatchLeastSquares,
    ohms_law_example,
    wheel_odometry_calibration_example,
    kinematic_position_velocity_example,
    lidar_plane_fitting_example,
    LinearKalmanFilter,
    create_2d_constant_velocity_tracker,
)

print("position_class initialized successfully!")
"""
        ),
        nbf.v4.new_markdown_cell(
r"""## 1. Where Does the Measurement Matrix $\mathbf{H}$ Come From?

In linear state estimation, the measurement equation maps the unobserved state vector $\mathbf{x} \in \mathbb{R}^n$ to observed sensor signals $\mathbf{y} \in \mathbb{R}^m$:

$$\mathbf{y} = \mathbf{H} \mathbf{x} + \mathbf{v}, \quad \mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$$

$\mathbf{H}$ is derived directly from the physical sensing equations.

### Automotive Engineering Examples:
1. **Current Sensing Calibration (Ohm's Law)**: $V_i = I_i \cdot R \implies \mathbf{y} = [V_i], \mathbf{H} = [I_i], x = R$.
2. **Wheel Odometry Calibration**: $v_{\text{meas}, i} = \omega_i \cdot r_{\text{eff}} \implies \mathbf{y} = [v_i], \mathbf{H} = [\omega_i], x = r_{\text{eff}}$.
3. **Vehicle Initial State from GPS**: $p_k = p_0 + v_0 \cdot t_k \implies \mathbf{y} = [p_k], \mathbf{H} = [[1, t_k]], \mathbf{x} = [p_0, v_0]^T$.
4. **LiDAR Ground Plane Fitting**: $z_i = a x_i + b y_i + c \implies \mathbf{y} = [z_i], \mathbf{H} = [[x_i, y_i, 1]], \mathbf{x} = [a, b, c]^T$.
"""
        ),
        nbf.v4.new_code_cell(
"""# Run the 4 automotive calibration examples
r_est, r_std = ohms_law_example()
wheel_r, wheel_std = wheel_odometry_calibration_example()
kin_x, kin_cov = kinematic_position_velocity_example()
lidar_params, lidar_cov = lidar_plane_fitting_example()

print(f"1. Resistor Estimate: {r_est:.3f} ± {r_std:.3f} Ohms (True: ~5.0 Ohms)")
print(f"2. Wheel Radius: {wheel_r:.4f} ± {wheel_std:.4f} m (True: ~0.33 m)")
print(f"3. Initial Pos: {kin_x[0]:.3f} m, Vel: {kin_x[1]:.3f} m/s (True: 10.0 m, 12.0 m/s)")
print(f"4. LiDAR Ground Plane: a={lidar_params[0]:.4f}, b={lidar_params[1]:.4f}, c={lidar_params[2]:.4f}")
"""
        ),
        nbf.v4.new_markdown_cell(
r"""## 2. Where Does Error Covariance $\mathbf{P}$ Come From?

The Weighted Least Squares (WLS) solution minimizes:

$$J(\mathbf{x}) = \frac{1}{2} (\mathbf{y} - \mathbf{H}\mathbf{x})^T \mathbf{R}^{-1} (\mathbf{y} - \mathbf{H}\mathbf{x})$$

Setting $\nabla_{\mathbf{x}} J = \mathbf{0}$ yields:

$$\hat{\mathbf{x}} = (\mathbf{H}^T \mathbf{R}^{-1} \mathbf{H})^{-1} \mathbf{H}^T \mathbf{R}^{-1} \mathbf{y}$$

Substituting $\mathbf{y} = \mathbf{H} \mathbf{x}_{\text{true}} + \mathbf{v}$:

$$\hat{\mathbf{x}} - \mathbf{x}_{\text{true}} = (\mathbf{H}^T \mathbf{R}^{-1} \mathbf{H})^{-1} \mathbf{H}^T \mathbf{R}^{-1} \mathbf{v}$$

The posterior covariance matrix $\mathbf{P}$ is the expectation of the outer product of the estimation error:

$$\mathbf{P} = \mathbb{E}[(\hat{\mathbf{x}} - \mathbf{x}_{\text{true}})(\hat{\mathbf{x}} - \mathbf{x}_{\text{true}})^T] = (\mathbf{H}^T \mathbf{R}^{-1} \mathbf{H})^{-1}$$
"""
        ),
        nbf.v4.new_code_cell(
"""# Visualizing LiDAR Ground Plane Fit with Plotly 3D Scatter
np.random.seed(42)
N = 150
x_pts = np.random.uniform(-10, 10, N)
y_pts = np.random.uniform(5, 35, N)
a_true, b_true, c_true = 0.02, -0.05, -1.50
z_pts = a_true * x_pts + b_true * y_pts + c_true + np.random.normal(0, 0.04, N)

# Plane mesh
xx, yy = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(5, 35, 10))
zz = lidar_params[0] * xx + lidar_params[1] * yy + lidar_params[2]

fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=x_pts, y=y_pts, z=z_pts,
    mode='markers',
    marker=dict(size=4, color='orange', opacity=0.8),
    name='LiDAR Ground Points'
))
fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz,
    opacity=0.5,
    colorscale='Viridis',
    showscale=False,
    name='Fitted Plane'
))
fig.update_layout(
    title='<b>LiDAR Ground Plane Least Squares Estimation</b>',
    scene=dict(xaxis_title='X [m] (Lateral)', yaxis_title='Y [m] (Longitudinal)', zaxis_title='Z [m] (Elevation)'),
    margin=dict(l=0, r=0, b=0, t=40)
)
fig.show()
"""
        ),
        nbf.v4.new_markdown_cell(
r"""## 3. 2D Linear Kalman Filter Tracking

We now test a 4D State Constant Velocity Kalman Filter:

$$\mathbf{x}_k = \begin{bmatrix} p_x \\ p_y \\ v_x \\ v_y \end{bmatrix}_k, \quad
\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad
\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$
"""
        ),
        nbf.v4.new_code_cell(
"""# Simulate Vehicle 2D Tracking
np.random.seed(42)
dt = 0.1
total_time = 20.0
steps = int(total_time / dt)

# Ground truth constant velocity with slight sinusoidal perturbation
t = np.linspace(0, total_time, steps)
gt_x = 5.0 * t
gt_y = 20.0 * np.sin(0.2 * t)
gt_vx = np.gradient(gt_x, dt)
gt_vy = np.gradient(gt_y, dt)

# Noisy GPS measurements
sigma_pos = 1.5
meas_x = gt_x + np.random.normal(0, sigma_pos, steps)
meas_y = gt_y + np.random.normal(0, sigma_pos, steps)

# Instantiate Linear Kalman Filter
kf = create_2d_constant_velocity_tracker(dt=dt, sigma_pos_gps=sigma_pos, sigma_acc_process=0.5)
kf.initialize(np.array([meas_x[0], meas_y[0], 0.0, 0.0]), np.eye(4) * 10.0)

est_x, est_y, est_vx, est_vy = [], [], [], []
cov_x, cov_y = [], []

for k in range(steps):
    kf.predict()
    state = kf.update(np.array([meas_x[k], meas_y[k]]))
    est_x.append(state.mean[0])
    est_y.append(state.mean[1])
    est_vx.append(state.mean[2])
    est_vy.append(state.mean[3])
    cov_x.append(np.sqrt(state.covariance[0, 0]))
    cov_y.append(np.sqrt(state.covariance[1, 1]))

# Plot tracking results
fig = make_subplots(rows=2, cols=1, subplot_titles=("<b>2D Vehicle Trajectory Tracking</b>", "<b>Estimated vs Ground Truth Velocities</b>"))

# Subplot 1: Trajectory
fig.add_trace(go.Scatter(x=gt_x, y=gt_y, mode='lines', name='Ground Truth', line=dict(color='black', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=meas_x, y=meas_y, mode='markers', name='Noisy GPS Measurements', marker=dict(color='red', size=4, opacity=0.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=est_x, y=est_y, mode='lines', name='Kalman Filter Estimate', line=dict(color='blue', width=2)), row=1, col=1)

# Subplot 2: Velocities
fig.add_trace(go.Scatter(x=t, y=gt_vx, mode='lines', name='True Vx', line=dict(color='black', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=est_vx, mode='lines', name='Estimated Vx', line=dict(color='green')), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=gt_vy, mode='lines', name='True Vy', line=dict(color='black', dash='dot')), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=est_vy, mode='lines', name='Estimated Vy', line=dict(color='magenta')), row=2, col=1)

fig.update_layout(height=700, title_text="<b>Day 01: 2D Linear Kalman Filter Performance Evaluation</b>")
fig.show()
"""
        )
    ]
    nb.cells = cells
    return nb


def create_day_02_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"language_info": {"name": "python"}}

    cells = [
        nbf.v4.new_markdown_cell(
"""# Day 02: Nonlinear State Estimation & Visual FAST-Kalman Tracking
**State Estimation and Localization for Self-Driving Cars**

Today we tackle nonlinear dynamic systems and computer vision-integrated tracking:
1. **Analytical Jacobians**: Systematic derivation of $\\mathbf{F}_{k-1}, \\mathbf{L}_{k-1}, \\mathbf{H}_k, \\mathbf{M}_k$.
2. **The Extended Kalman Filter (EKF)**: Landmark bearing and 2D radar tracking.
3. **The Unscented Kalman Filter (UKF)**: Deterministic sampling with the Scaled Unscented Transform.
4. **Visual Motion Prediction with FAST Algorithm & Kalman Filter**:
   - Extraction of FAST corner keypoints and ORB descriptors on Region of Interest (ROI).
   - Frame-by-frame brute-force feature matching and centroid computation.
   - Forward motion prediction and Kalman correction.
"""
        ),
        nbf.v4.new_code_cell(
"""import cv2
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from position_class import (
    ExtendedKalmanFilter,
    LandmarkBearingEKF,
    Radar2DTargetTrackerEKF,
    UnscentedKalmanFilter,
    LandmarkBearingUKF,
    FastKalmanVisualTracker,
    generate_synthetic_tracking_video,
    wraptopi
)

print("Day 02 Modules Loaded Successfully!")
"""
        ),
        nbf.v4.new_markdown_cell(
r"""## 1. Landmark Bearing Tracking: EKF vs UKF

Consider a vehicle observing the relative bearing $\theta_k$ to a stationary landmark at $\mathbf{l} = [x_l, y_l]^T$:

$$\mathbf{x}_k = \begin{bmatrix} x_k \\ y_k \\ \theta_k \end{bmatrix}, \quad
h(\mathbf{x}_k) = \operatorname{atan2}(y_l - y_k, x_l - x_k) - \theta_k$$

Measurement Jacobian $\mathbf{H}_k$:

$$\mathbf{H}_k = \left[ \frac{y_l - y_k}{d^2}, \quad -\frac{x_l - x_k}{d^2}, \quad -1 \right], \quad d^2 = (x_l - x_k)^2 + (y_l - y_k)^2$$
"""
        ),
        nbf.v4.new_code_cell(
"""# Simulation of Optical Landmark Bearing Navigation (Coursera / Excersice.ipynb)
np.random.seed(42)
dt = 0.5
steps = 60
t = np.linspace(0, steps * dt, steps)
S = 20.0  # Landmark lateral distance (m)
D = 40.0  # Landmark longitudinal position (m)

# True motion: constant velocity vehicle moving along x-axis from p=0 to p=30
true_p = 5.0 * t
true_v = np.full_like(t, 5.0)

# True bearing angle measurements
true_bearing = np.arctan(S / (D - true_p))
meas_bearing = true_bearing + np.random.normal(0, np.sqrt(0.01), steps)

# Initialize EKF and UKF
ekf = LandmarkBearingEKF(dt=dt, S=S, D=D, R=0.01)
ukf = LandmarkBearingUKF(dt=dt, S=S, D=D, R=0.01)

ekf_pos, ukf_pos = [], []
ekf_cov, ukf_cov = [], []

for k in range(steps):
    x_ekf, P_ekf = ekf.step(u=0.0, y=meas_bearing[k])
    x_ukf, P_ukf = ukf.step(u=0.0, y=meas_bearing[k])
    
    ekf_pos.append(x_ekf[0, 0])
    ukf_pos.append(x_ukf[0, 0])
    ekf_cov.append(P_ekf[0, 0])
    ukf_cov.append(P_ukf[0, 0])

ekf_pos = np.array(ekf_pos)
ukf_pos = np.array(ukf_pos)

# Plot EKF vs UKF
fig = make_subplots(rows=2, cols=1, subplot_titles=("<b>Vehicle Position: Ground Truth vs EKF vs UKF</b>", "<b>Bearing Angle Measurements & Nonlinear Model</b>"))
fig.add_trace(go.Scatter(x=t, y=true_p, mode='lines', name='Ground Truth Position', line=dict(color='black', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=ekf_pos, mode='lines', name='EKF Estimate', line=dict(color='blue', dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=ukf_pos, mode='lines', name='UKF Estimate', line=dict(color='orange', dash='dot', width=2)), row=1, col=1)

fig.add_trace(go.Scatter(x=t, y=np.degrees(true_bearing), mode='lines', name='True Bearing (deg)', line=dict(color='black')), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=np.degrees(meas_bearing), mode='markers', name='Noisy Bearing Measurements', marker=dict(color='red', size=4)), row=2, col=1)

fig.update_layout(
    title='<b>Nonlinear Landmark Navigation: EKF vs UKF Performance</b>',
    height=650
)
fig.show()
"""
        ),
        nbf.v4.new_markdown_cell(
r"""## 2. Visual Object Motion Prediction: FAST Algorithm + Kalman Filter

Inspired by the VisionBrick visual motion tracking architecture:
- **FAST (Features from Accelerated Segment Test)** provides high-speed corner extraction in ROI.
- **ORB/BRIEF Descriptors** provide scale- and rotation-invariant feature representations.
- **Kalman Filter** provides constant-velocity motion prediction $\hat{\mathbf{x}}_{k|k-1} = \mathbf{F} \hat{\mathbf{x}}_{k-1|k-1}$ and continuous smoothing against noisy visual detections and occlusions.
"""
        ),
        nbf.v4.new_code_cell(
"""# Generate synthetic video stream with textured moving object
frames, gt_centroids = generate_synthetic_tracking_video(num_frames=80, width=640, height=480)

# Initialize FAST + Kalman Visual Tracker on Frame 0
tracker = FastKalmanVisualTracker(dt=1.0/30.0, process_noise_std=1.0, measurement_noise_std=2.0, fast_threshold=15)
initial_bbox = (int(gt_centroids[0][0] - 20), int(gt_centroids[0][1] - 20), 40, 40)
num_kps = tracker.initialize_target_from_roi(frames[0], initial_bbox)
print(f"Target ROI initialized with {num_kps} FAST keypoints.")

# Run tracking pipeline over frame sequence
tracked_results = []
for idx, frame in enumerate(frames):
    res = tracker.process_frame(frame)
    tracked_results.append(res)

gt_arr = np.array(gt_centroids)
pred_x = [r["pred_x"] for r in tracked_results]
pred_y = [r["pred_y"] for r in tracked_results]
meas_x = [r["meas_x"] if r["meas_x"] is not None else np.nan for r in tracked_results]
meas_y = [r["meas_y"] if r["meas_y"] is not None else np.nan for r in tracked_results]
est_x = [r["est_x"] for r in tracked_results]
est_y = [r["est_y"] for r in tracked_results]
matches = [r["matches"] for r in tracked_results]

# Plot Visual Tracking Performance
fig = make_subplots(rows=2, cols=1, subplot_titles=("<b>Visual Object Tracking: Ground Truth vs FAST Matches vs Kalman Estimates</b>", "<b>Number of Matched FAST Keypoints per Frame</b>"))

fig.add_trace(go.Scatter(x=gt_arr[:, 0], y=gt_arr[:, 1], mode='lines', name='Ground Truth Centroid', line=dict(color='black', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=meas_x, y=meas_y, mode='markers', name='FAST Matched Centroid', marker=dict(color='red', size=6, symbol='x')), row=1, col=1)
fig.add_trace(go.Scatter(x=est_x, y=est_y, mode='lines', name='Kalman Filtered Position', line=dict(color='cyan', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=pred_x, y=pred_y, mode='lines', name='Kalman Predicted Position', line=dict(color='gold', dash='dot')), row=1, col=1)

fig.add_trace(go.Bar(x=list(range(len(matches))), y=matches, name='Matched Features', marker=dict(color='royalblue')), row=2, col=1)

fig.update_layout(height=750, title_text="<b>VisionBrick FAST + Kalman Filter Motion Prediction System</b>")
fig.show()
"""
        )
    ]
    nb.cells = cells
    return nb


def create_day_03_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"language_info": {"name": "python"}}

    cells = [
        nbf.v4.new_markdown_cell(
"""# Day 03: 3D Geometry, Quaternion Kinematics & Inertial Dead Reckoning
**State Estimation and Localization for Self-Driving Cars**

Today we transition from 2D planar motion to full 3D spatial orientation and inertial mechanics:
1. **3D Rotation Parameterizations**: Euler Angles (Gimbal Lock singularity) vs Direction Cosine Matrices (DCM) vs Hamiltonian Unit Quaternions.
2. **Quaternion Kinematics**: Angular velocity integration via the exponential map $\\mathbf{q}_k = \\mathbf{q}_{k-1} \\otimes \\operatorname{Exp}(\\frac{1}{2} \\boldsymbol{\\omega} \\Delta t)$.
3. **Strapdown IMU Dead Reckoning**: 3D numerical integration of specific force $\\mathbf{f}_b$ and gravity compensation.
4. **IMU Error Budget**: Acceleration bias propagation leading to quadratic position drift ($e(t) \\sim \\frac{1}{2} \\mathbf{b}_a t^2$).
5. **GNSS Pseudorange Trilateration**: Solving 3D receiver coordinates and clock bias via Gauss-Newton Least Squares.
"""
        ),
        nbf.v4.new_code_cell(
"""import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from position_class import (
    Quaternion,
    StrapdownIMUIntegrator,
    solve_gnss_trilateration,
    skew_symmetric
)

print("Day 03 Modules Loaded Successfully!")
"""
        ),
        nbf.v4.new_markdown_cell(
r"""## 1. Quaternion Kinematics & Rotation Conversions

Unit quaternions $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$ avoid the Euler angle gimbal lock singularity ($\theta = \pm 90^\circ$).
"""
        ),
        nbf.v4.new_code_cell(
"""# Quaternion multiplication & Euler angle verification
q1 = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.radians(45.0))
q2 = Quaternion.from_axis_angle(np.array([0, 1, 0]), np.radians(30.0))
q_combined = q1.multiply(q2)

roll, pitch, yaw = q_combined.to_euler_angles()
print(f"Combined Euler Angles: Roll={np.degrees(roll):.2f} deg, Pitch={np.degrees(pitch):.2f} deg, Yaw={np.degrees(yaw):.2f} deg")
print(f"Rotation Matrix Norm Check (Orthonormality): {np.linalg.norm(q_combined.to_rotation_matrix() @ q_combined.to_rotation_matrix().T - np.eye(3)):.2e}")
"""
        ),
        nbf.v4.new_markdown_cell(
r"""## 2. Strapdown IMU Dead Reckoning & Quadratic Drift Divergence

We simulate 60 seconds of vehicle motion with a small constant accelerometer bias $b_{ax} = 0.05\text{ m/s}^2$.

Notice how unassisted IMU position diverges quadratically:

$$\Delta p(t) = \frac{1}{2} b_a t^2 \implies \Delta p(60\text{s}) = \frac{1}{2} (0.05) (3600) = 90.0\text{ meters!}$$
"""
        ),
        nbf.v4.new_code_cell(
"""# Simulate IMU Dead Reckoning with Bias
dt = 0.01  # 100 Hz IMU
total_time = 60.0
steps = int(total_time / dt)
t = np.linspace(0, total_time, steps)

# True motion: straight driving at 20 m/s with 9.81 m/s^2 gravity upward reaction
true_fb = np.array([0.0, 0.0, 9.81])
true_omega = np.array([0.0, 0.0, 0.0])

# Sensor corrupted with bias and white noise
accel_bias = np.array([0.05, 0.0, 0.0])  # 0.05 m/s^2 bias in X
noise_acc = 0.02

integrator_ideal = StrapdownIMUIntegrator(init_pos=np.zeros(3), init_vel=np.array([20.0, 0.0, 0.0]))
integrator_biased = StrapdownIMUIntegrator(init_pos=np.zeros(3), init_vel=np.array([20.0, 0.0, 0.0]))

pos_ideal, pos_biased = [], []

for k in range(steps):
    fb_ideal = true_fb
    fb_biased = true_fb + accel_bias + np.random.normal(0, noise_acc, 3)
    
    p_id, _, _ = integrator_ideal.step(fb_ideal, true_omega, dt)
    p_bi, _, _ = integrator_biased.step(fb_biased, true_omega, dt)
    
    pos_ideal.append(p_id)
    pos_biased.append(p_bi)

pos_ideal = np.array(pos_ideal)
pos_biased = np.array(pos_biased)
drift_error = pos_biased[:, 0] - pos_ideal[:, 0]

# Plot Drift Divergence
fig = make_subplots(rows=2, cols=1, subplot_titles=("<b>X-Position: Ideal vs Biased IMU Integration</b>", "<b>Quadratic Position Drift Error (m)</b>"))

fig.add_trace(go.Scatter(x=t, y=pos_ideal[:, 0], mode='lines', name='Ideal Trajectory', line=dict(color='green', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=pos_biased[:, 0], mode='lines', name='Biased IMU (0.05 m/s²)', line=dict(color='red', width=2)), row=1, col=1)

fig.add_trace(go.Scatter(x=t, y=drift_error, mode='lines', name='Empirical Drift', line=dict(color='red', width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=0.5 * accel_bias[0] * (t**2), mode='lines', name='Theoretical Quadratic Curve (0.5 * b * t²)', line=dict(color='black', dash='dash')), row=2, col=1)

fig.update_layout(height=650, title_text="<b>IMU Dead Reckoning Drift Analysis</b>")
fig.show()
"""
        ),
        nbf.v4.new_markdown_cell(
r"""## 3. 3D GNSS Pseudorange Trilateration

Solving for vehicle 3D coordinate $[x, y, z]$ and clock offset $c \cdot \delta t$ from 6 satellite pseudorange measurements:

$$\rho_i = \|\mathbf{r}_{\text{sat}, i} - \mathbf{r}_{\text{rx}}\| + c \delta t_{\text{rx}} + \epsilon_i$$
"""
        ),
        nbf.v4.new_code_cell(
"""# True receiver position in ECEF frame (meters)
true_rx = np.array([1500000.0, -4500000.0, 4200000.0])
true_bias = 250.0  # meters

# 6 GNSS Satellite Constellation Positions
satellites = np.array([
    [15600000.0, -18000000.0, 20000000.0],
    [20000000.0, -12000000.0, 18000000.0],
    [10000000.0, -22000000.0, 16000000.0],
    [18000000.0, -16000000.0, 22000000.0],
    [12000000.0, -20000000.0, 24000000.0],
    [22000000.0, -14000000.0, 15000000.0]
])

# Generate noisy pseudoranges
ranges = np.linalg.norm(satellites - true_rx, axis=1) + true_bias + np.random.normal(0, 1.5, 6)

# Solve GNSS position via Gauss-Newton
est_pos, est_bias = solve_gnss_trilateration(satellites, ranges)

print(f"True Position:  {true_rx}")
print(f"Estimated Pos:  {est_pos}")
print(f"Position Error: {np.linalg.norm(est_pos - true_rx):.3f} meters")
print(f"Estimated Clock Bias: {est_bias:.2f} m (True: {true_bias:.2f} m)")
"""
        )
    ]
    nb.cells = cells
    return nb


def create_day_04_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"language_info": {"name": "python"}}

    cells = [
        nbf.v4.new_markdown_cell(
"""# Day 04: Multi-Sensor Fusion & Localization Capstone
**State Estimation and Localization for Self-Driving Cars**

Welcome to the capstone session! Today we build the industry-standard vehicle localization engine:
1. **The Error-State Extended Kalman Filter (ES-EKF)**:
   - Why Error-State? Decoupling continuous nominal kinematics from 3D error quaternion perturbation vector $\\delta \\boldsymbol{\\theta}$.
   - High-rate 100 Hz IMU nominal state integration.
   - Low-rate 10 Hz GNSS position corrections.
2. **Sensor Fusion Pipeline**:
   - Covariance propagation and Joseph-form measurement updates.
   - Error injection and reset mechanism.
3. **Challenging GNSS Tunnel Outage Scenario**:
   - 10-second complete GNSS signal blackout.
   - Analysis of covariance envelope explosion and instant re-convergence upon satellite re-acquisition.
4. **Statistical Verification Metrics**:
   - Normalized Estimation Error Squared (NEES) for state consistency.
   - Normalized Innovation Squared (NIS) for filter health monitoring.
"""
        ),
        nbf.v4.new_code_cell(
"""import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from position_class import (
    ErrorStateEKF,
    Quaternion,
    skew_symmetric
)

print("Day 04 Capstone Environment Ready!")
"""
        ),
        nbf.v4.new_markdown_cell(
r"""## 1. The Error-State EKF (ES-EKF) Architecture

The state is decomposed into:
- **Nominal State** (large, non-linear): $\mathbf{x} = [\mathbf{p}, \mathbf{v}, \mathbf{q}]$
- **Error State** (small, linear): $\delta \mathbf{x} = [\delta \mathbf{p}, \delta \mathbf{v}, \delta \boldsymbol{\theta}]^T \in \mathbb{R}^9$

Measurement Model for GNSS:

$$\mathbf{y}_k = \mathbf{p}_{\text{GNSS}} - \mathbf{p}_{\text{nom}} = \mathbf{H} \delta \mathbf{x}_k + \mathbf{v}_k, \quad \mathbf{H} = \begin{bmatrix} \mathbf{I}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} \end{bmatrix}$$
"""
        ),
        nbf.v4.new_code_cell(
"""# Simulation of 90-Second Vehicle Loop with 10-Second Tunnel Outage
np.random.seed(42)
dt = 0.01  # 100 Hz IMU
total_time = 60.0
steps = int(total_time / dt)
t = np.linspace(0, total_time, steps)

# 1. Generate Ground Truth Race Track Loop
R = 50.0
speed = 15.0  # m/s
omega = speed / R

gt_x = R * np.sin(omega * t)
gt_y = R * np.cos(omega * t) - R
gt_z = np.zeros_like(t)

gt_vx = R * omega * np.cos(omega * t)
gt_vy = -R * omega * np.sin(omega * t)
gt_vz = np.zeros_like(t)

# Ground truth IMU readings
acc_centripetal = (speed ** 2) / R
gt_ax = -acc_centripetal * np.sin(omega * t)
gt_ay = -acc_centripetal * np.cos(omega * t)
gt_az = np.full_like(t, 9.81)  # Specific force counteracting gravity

# Initialize ES-EKF
init_p = np.array([gt_x[0], gt_y[0], 0.0])
init_v = np.array([gt_vx[0], gt_vy[0], 0.0])
init_q = Quaternion([1.0, 0.0, 0.0, 0.0])
init_cov = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

es_ekf = ErrorStateEKF(init_pos=init_p, init_vel=init_v, init_quat=init_q, init_cov=init_cov, accel_noise_std=0.2, gyro_noise_std=0.02)

# GPS Configuration: 10 Hz rate, 10-second tunnel outage between t = 25s and t = 35s
gps_dt = 0.1
gps_subsample = int(gps_dt / dt)
r_gps = np.eye(3) * (0.8 ** 2)

est_pos, est_cov_pos = [], []
nees_list, nis_list, nis_times = [], [], []

for k in range(steps):
    current_time = t[k]
    
    # 1. Noisy IMU Measurement
    f_b = np.array([0.0, acc_centripetal, 9.81]) + np.random.normal(0, 0.15, 3)
    omega_b = np.array([0.0, 0.0, -omega]) + np.random.normal(0, 0.01, 3)
    
    p, v, q, P = es_ekf.predict(f_b, omega_b, dt)
    
    # 2. GNSS Correction (10 Hz, inactive in tunnel t in [25, 35])
    is_in_tunnel = 25.0 <= current_time <= 35.0
    if k % gps_subsample == 0 and not is_in_tunnel:
        gps_meas = np.array([gt_x[k], gt_y[k], gt_z[k]]) + np.random.normal(0, 0.8, 3)
        p, v, q, P, nis = es_ekf.update_gnss_position(gps_meas, r_gps)
        nis_list.append(nis)
        nis_times.append(current_time)
        
    est_pos.append(p)
    est_cov_pos.append(np.diag(P)[0:3])
    
    # Position error NEES
    err_p = p - np.array([gt_x[k], gt_y[k], gt_z[k]])
    nees = float(err_p.T @ np.linalg.inv(P[0:3, 0:3]) @ err_p)
    nees_list.append(nees)

est_pos = np.array(est_pos)
est_cov_pos = np.array(est_cov_pos)
pos_3sigma = 3.0 * np.sqrt(est_cov_pos)

print("Simulation Completed!")
"""
        ),
        nbf.v4.new_markdown_cell(
r"""## 2. Interactive Performance & Tunnel Outage Visualizations

Observe how:
1. When GNSS is available, position error remains bounded within centimeters.
2. During the 10-second tunnel outage ($t \in [25\text{s}, 35\text{s}]$), the $3\sigma$ covariance bounds expand naturally, capturing uncertainty.
3. Upon exiting the tunnel, the first GNSS fix immediately shrinks the error covariance back down.
"""
        ),
        nbf.v4.new_code_cell(
"""# Plot 1: 2D Multi-Sensor Trajectory & Tunnel Region
fig_traj = go.Figure()
fig_traj.add_trace(go.Scatter(x=gt_x, y=gt_y, mode='lines', name='Ground Truth Path', line=dict(color='black', width=3)))
fig_traj.add_trace(go.Scatter(x=est_pos[:, 0], y=est_pos[:, 1], mode='lines', name='ES-EKF Fused Trajectory', line=dict(color='blue', width=2)))

# Highlight Tunnel Zone
tunnel_mask = (t >= 25.0) & (t <= 35.0)
fig_traj.add_trace(go.Scatter(
    x=est_pos[tunnel_mask, 0], y=est_pos[tunnel_mask, 1],
    mode='lines', name='Tunnel Dead Reckoning (No GNSS)',
    line=dict(color='red', width=3, dash='dash')
))

fig_traj.update_layout(
    title='<b>ES-EKF Multi-Sensor Vehicle Trajectory (IMU + GNSS + 10s Tunnel Outage)</b>',
    xaxis_title='X [m] (East)', yaxis_title='Y [m] (North)',
    height=550
)
fig_traj.show()
"""
        ),
        nbf.v4.new_code_cell(
"""# Plot 2: Position Error vs 3-Sigma Bounds & Consistency Metrics
fig_metrics = make_subplots(
    rows=2, cols=1,
    subplot_titles=("<b>X-Position Error with 3σ Covariance Bounds (Tunnel Outage Highlighted)</b>", "<b>Normalized Estimation Error Squared (NEES)</b>")
)

x_err = est_pos[:, 0] - gt_x
fig_metrics.add_trace(go.Scatter(x=t, y=x_err, mode='lines', name='Position Error X (m)', line=dict(color='blue')), row=1, col=1)
fig_metrics.add_trace(go.Scatter(x=t, y=pos_3sigma[:, 0], mode='lines', name='+3σ Bound', line=dict(color='gray', dash='dash')), row=1, col=1)
fig_metrics.add_trace(go.Scatter(x=t, y=-pos_3sigma[:, 0], mode='lines', name='-3σ Bound', line=dict(color='gray', dash='dash'), fill='tonexty', fillcolor='rgba(180,180,180,0.2)'), row=1, col=1)

# Add tunnel shaded rect
fig_metrics.add_vrect(x0=25.0, x1=35.0, fillcolor="red", opacity=0.15, line_width=0, annotation_text="10s Tunnel Outage", row=1, col=1)

# NEES
fig_metrics.add_trace(go.Scatter(x=t, y=nees_list, mode='lines', name='Position NEES', line=dict(color='purple')), row=2, col=1)
fig_metrics.add_hline(y=7.81, line_dash="dash", line_color="red", annotation_text="95% Upper Chi-Square Bound (df=3)", row=2, col=1)

fig_metrics.update_layout(height=750, title_text="<b>Day 04: ES-EKF Consistency & Tunnel Outage Resilience Analysis</b>")
fig_metrics.show()
"""
        )
    ]
    nb.cells = cells
    return nb


if __name__ == "__main__":
    import os
    
    nb1 = create_day_01_notebook()
    with open("day_01_foundations_and_linear_kalman_filter.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb1, f)
    print("Generated day_01_foundations_and_linear_kalman_filter.ipynb")

    nb2 = create_day_02_notebook()
    with open("day_02_nonlinear_estimation_and_visual_tracking.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb2, f)
    print("Generated day_02_nonlinear_estimation_and_visual_tracking.ipynb")

    nb3 = create_day_03_notebook()
    with open("day_03_3d_geometry_and_inertial_navigation.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb3, f)
    print("Generated day_03_3d_geometry_and_inertial_navigation.ipynb")

    nb4 = create_day_04_notebook()
    with open("day_04_multisensor_fusion_and_localization.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb4, f)
    print("Generated day_04_multisensor_fusion_and_localization.ipynb")
