# Position Class: State Estimation & Localization Library
## Autonomous Vehicle Estimation, Sensor Fusion & Filtering Suite

This repository contains interactive instructional notebooks, simulation benchmarks, and a modular Python library for **State Estimation and Localization in Self-Driving Cars**.

---

## 📂 Architecture & Package Structure

```text
position_class/
├── least_squares_exercise.ipynb                         # Lab Exercise: Complete Least Squares Solution & Verification Guide
├── least_squares_exercise.html                          # Standalone browser view for Solution Guide
├── least_squares_exercise_student.ipynb                 # Lab Exercise: Scaffolded Student Version (Fill-in-the-TODOs)
├── least_squares_exercise_student.html                  # Standalone browser view for Student Version
├── hour_04_linear_kalman_filter_lab.ipynb               # Hour 4 Lab: Complete 1D & 2D Linear Kalman Filter Solution Guide
├── hour_04_linear_kalman_filter_lab.html                # Standalone browser view for Hour 4 Solution
├── hour_04_linear_kalman_filter_lab_student.ipynb       # Hour 4 Lab: Scaffolded Student Version (Fill-in-the-TODOs)
├── hour_04_linear_kalman_filter_lab_student.html        # Standalone browser view for Hour 4 Student Version
├── day_01_foundations_and_linear_kalman_filter.ipynb    # Day 01: Least Squares (H, P derivations), RLS, & 2D LKF Tracker
├── day_01_foundations_and_linear_kalman_filter.html     # Pre-rendered standalone browser view
├── day_02_nonlinear_estimation_and_visual_tracking.ipynb# Day 02: EKF Jacobians, UKF, & VisionBrick FAST-Kalman Object Tracker
├── day_02_nonlinear_estimation_and_visual_tracking.html # Pre-rendered standalone browser view
├── day_03_3d_geometry_and_inertial_navigation.ipynb     # Day 03: Quaternions, Strapdown IMU Dead Reckoning, & GNSS Trilateration
├── day_03_3d_geometry_and_inertial_navigation.html      # Pre-rendered standalone browser view
├── day_04_multisensor_fusion_and_localization.ipynb     # Day 04: Error-State EKF (ES-EKF) 9D Fusion, 10s Tunnel Outage & NEES
├── day_04_multisensor_fusion_and_localization.html      # Pre-rendered standalone browser view
├── pyproject.toml                                       # UV project configuration and dependencies
├── uv.lock                                              # Exact dependency lockfile
├── README.md                                            # This documentation
└── src/
    └── position_class/
        ├── __init__.py                                  # Package exports
        ├── least_squares.py                             # Batch & Weighted Least Squares (4 Automotive calibration examples)
        ├── linear_kalman_filter.py                      # Discrete Linear Kalman Filter (LKF) & CV tracking
        ├── extended_kalman_filter.py                    # Extended Kalman Filter (EKF) with analytical Jacobians (F, L, H, M)
        ├── unscented_kalman_filter.py                   # Unscented Kalman Filter (UKF) with Scaled Unscented Transform
        ├── vision_kalman_tracker.py                     # FAST Keypoints + ORB + 2D Kalman Filter Visual Motion Tracking
        ├── inertial_navigation.py                       # 3D Quaternion Algebra, Strapdown IMU Integrator, GNSS Solver
        └── es_ekf.py                                    # Error-State Extended Kalman Filter (ES-EKF) for IMU + GNSS Fusion
```

---

## 📓 Interactive Jupyter Notebooks

### 1. [`hour_04_linear_kalman_filter_lab.ipynb`](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/position_class/hour_04_linear_kalman_filter_lab.ipynb) ([Student Version](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/position_class/hour_04_linear_kalman_filter_lab_student.ipynb))
* **1D Kinematic Tracking**: Latent velocity discovery from position-only noisy GPS pings with $\pm 3\sigma$ confidence envelopes.
* **2D Vehicle Tracking**: 4D state vector ($\mathbf{x} = [x, y, \dot{x}, \dot{y}]^T$) under Constant Velocity motion on curved highway paths.
* **Sensor Dropout Simulation**: 5-second GPS blackout in a tunnel, dead reckoning drift, and covariance expansion/collapse.
* **$\mathbf{Q}$ vs. $\mathbf{R}$ Tuning Analysis**: Overconfident model (dynamic lag) vs. underconfident model (measurement jitter).
* **Statistical Consistency & NEES**: Normalized Estimation Error Squared against 95% $\chi_4^2$ bounds.
* **Datasheet Prior Initialization**: $3\sigma$ spec-based initialization vs. cold diffuse prior.

### 2. [`day_02_nonlinear_estimation_and_visual_tracking.ipynb`](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/position_class/day_02_nonlinear_estimation_and_visual_tracking.ipynb)
* **Analytical Jacobians**: Mathematical derivation and verification of $\mathbf{F}_{k-1}, \mathbf{L}_{k-1}, \mathbf{H}_k, \mathbf{M}_k$.
* **EKF vs UKF Benchmark**: Nonlinear optical landmark bearing tracking from `Excersice.ipynb`.
* **VisionBrick FAST + Kalman Visual Motion Prediction**:
  - Extracts FAST corner features and ORB descriptors on a target Region of Interest (ROI).
  - Matches features frame-by-frame using Brute-Force Matcher (`cv2.BFMatcher`).
  - Predicts object motion in advance with the Kalman Filter and performs continuous correction upon visual detection.

### 3. [`day_03_3d_geometry_and_inertial_navigation.ipynb`](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/position_class/day_03_3d_geometry_and_inertial_navigation.ipynb)
* **3D Rotations & Quaternions**: Hamiltonian quaternion algebra, Euler angle singularity avoidance (Gimbal Lock), and rotation matrix conversions.
* **Strapdown IMU Dead Reckoning**: 3D numerical integration of accelerometer specific force and gyroscope angular rate with gravity compensation.
* **Quadratic Drift Divergence**: Analysis of uncorrected IMU bias error ($e(t) \sim \frac{1}{2} \mathbf{b}_a t^2$).
* **GNSS Pseudorange Trilateration**: Gauss-Newton iterative Least Squares for 3D vehicle coordinates and receiver clock bias.

### 4. [`day_04_multisensor_fusion_and_localization.ipynb`](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/position_class/day_04_multisensor_fusion_and_localization.ipynb)
* **The Error-State EKF (ES-EKF)**: 9D navigation filter fusing 100 Hz Strapdown IMU and 10 Hz GNSS position measurements.
* **10-Second GNSS Tunnel Outage Simulation**: Visualizes covariance envelope explosion during outage and instant sub-meter re-convergence upon satellite fix re-acquisition.
* **Statistical Verification**: Normalized Estimation Error Squared (NEES) and Normalized Innovation Squared (NIS) consistency metrics.

---

## 🚀 Running the Notebooks with UV

```bash
cd position_class

# 1. Sync dependencies
uv sync

# 2. Launch Jupyter Lab
uv run jupyter lab

# 3. Or launch classic Jupyter Notebook
uv run jupyter notebook
```

*Pre-rendered interactive HTML exports can be viewed directly in any browser:*
```bash
open day_01_foundations_and_linear_kalman_filter.html
open day_02_nonlinear_estimation_and_visual_tracking.html
open day_03_3d_geometry_and_inertial_navigation.html
open day_04_multisensor_fusion_and_localization.html
```

---

## 🛠️ Python API Overview

### 1. Visual Tracking with FAST & Kalman Filter (`vision_kalman_tracker.py`)
```python
from position_class import FastKalmanVisualTracker, generate_synthetic_tracking_video

# Generate video frames
frames, ground_truth = generate_synthetic_tracking_video(num_frames=80)

# Initialize FAST + Kalman Visual Tracker
tracker = FastKalmanVisualTracker(dt=1.0/30.0, fast_threshold=10)
initial_bbox = (80, 220, 60, 60)
tracker.initialize_target_from_roi(frames[0], initial_bbox)

# Process frame sequence
for frame in frames:
    result = tracker.process_frame(frame)
    # result contains: pred_x, pred_y, meas_x, meas_y, est_x, est_y, matches
```

### 2. Error-State EKF Multi-Sensor Fusion (`es_ekf.py`)
```python
import numpy as np
from position_class import ErrorStateEKF, Quaternion

# Initialize 9D ES-EKF
filter = ErrorStateEKF(
    init_pos=np.zeros(3),
    init_vel=np.array([15.0, 0.0, 0.0]),
    init_quat=Quaternion([1.0, 0.0, 0.0, 0.0]),
    init_cov=np.eye(9) * 0.1
)

# High-rate IMU predict (100 Hz)
p, v, q, P = filter.predict(f_b=np.array([0.0, 0.0, 9.81]), omega_b=np.zeros(3), dt=0.01)

# Low-rate GNSS correction (10 Hz)
p, v, q, P, nis = filter.update_gnss_position(p_meas=np.array([1.5, 0.1, 0.0]), r_cov=np.eye(3)*0.5)
```

---

## 📚 Master Syllabus & Course Markdown Files
* **12-Hour Master Course Syllabus:** [`Course_Syllabus_State_Estimation_and_Localization.md`](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/Course_Syllabus_State_Estimation_and_Localization.md)
* **Day 01 Lecture & Lab Notes:** [`Day_01_Foundations_of_State_Estimation/README.md`](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/Day_01_Foundations_of_State_Estimation/README.md)
* **Day 02 Lecture & Lab Notes:** [`Day_02_Nonlinear_Estimation_and_Object_Tracking/README.md`](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/Day_02_Nonlinear_Estimation_and_Object_Tracking/README.md)
* **Day 03 Lecture & Lab Notes:** [`Day_03_3d_Geometry_and_Sensor_Mechanics/README.md`](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/Day_03_3D_Geometry_and_Sensor_Mechanics/README.md)
* **Day 04 Lecture & Lab Notes:** [`Day_04_Multi_Sensor_Fusion_and_Localization/README.md`](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/Day_04_Multi_Sensor_Fusion_and_Localization/README.md)
