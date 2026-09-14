---
title: "Hour 4: Applied Programming Lab (1D & 2D Linear Tracking)"
tags:
  - hour-4
  - lab
  - python
  - programming
  - kalman-filter
created: 2026-09-08
---

# Hour 4: Applied Programming Lab (1D & 2D Linear Tracking)

> [!abstract] Lab Objective
> Build and validate a complete **2D Linear Kalman Filter** in Python from scratch. You will track a vehicle moving across a 2D plane with state vector $\mathbf{x} = [x, y, \dot{x}, \dot{y}]^T$, receiving noisy position measurements from a simulated sensor (such as a GPS or camera tracker).

---

## 1. Problem Specification

### 1.1 State Representation
The state of the vehicle at time step $k$ consists of its 2D position and 2D velocity:
$$\mathbf{x}_k = \begin{bmatrix} x_k \\ y_k \\ \dot{x}_k \\ \dot{y}_k \end{bmatrix} \in \mathbb{R}^4$$

### 1.2 Continuous & Discrete Motion Model (Constant Velocity)
Assuming constant velocity over time intervals $\Delta t$:
$$x_k = x_{k-1} + \dot{x}_{k-1}\Delta t + w_{x, k-1}$$
$$y_k = y_{k-1} + \dot{y}_{k-1}\Delta t + w_{y, k-1}$$
$$\dot{x}_k = \dot{x}_{k-1} + w_{\dot{x}, k-1}$$
$$\dot{y}_k = \dot{y}_{k-1} + w_{\dot{y}, k-1}$$

In matrix form $\mathbf{x}_k = \mathbf{F}\mathbf{x}_{k-1} + \mathbf{w}_{k-1}$:
$$\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

### 1.3 Process Noise Covariance $\mathbf{Q}$
Assuming independent acceleration disturbances along $x$ and $y$ with spectral density $\sigma_a^2$:
$$\mathbf{Q} = \begin{bmatrix} \frac{\Delta t^4}{4}\sigma_a^2 & 0 & \frac{\Delta t^3}{2}\sigma_a^2 & 0 \\ 0 & \frac{\Delta t^4}{4}\sigma_a^2 & 0 & \frac{\Delta t^3}{2}\sigma_a^2 \\ \frac{\Delta t^3}{2}\sigma_a^2 & 0 & \Delta t^2 \sigma_a^2 & 0 \\ 0 & \frac{\Delta t^3}{2}\sigma_a^2 & 0 & \Delta t^2 \sigma_a^2 \end{bmatrix}$$

### 1.4 Measurement Model
The sensor (e.g. GPS) directly measures 2D position $(x, y)$, but cannot measure velocity:
$$\mathbf{y}_k = \begin{bmatrix} x_{meas, k} \\ y_{meas, k} \end{bmatrix} = \mathbf{H}\mathbf{x}_k + \mathbf{v}_k$$
$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}, \quad \mathbf{R} = \begin{bmatrix} \sigma_{x,meas}^2 & 0 \\ 0 & \sigma_{y,meas}^2 \end{bmatrix}$$

---

## 2. Lab Implementation Tasks

Open your Python editor or Jupyter Notebook and complete the implementation tasks below.

### Task 1: Complete the Kalman Filter Class
Fill in the `predict()` and `update()` methods in the starter code below:

```python
import numpy as np
import plotly.graph_objects as go

class KalmanFilter2D:
    def __init__(self, dt: float, sigma_a: float, sigma_meas: float):
        self.dt = dt
        
        # State Transition Matrix F (4x4)
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Measurement Matrix H (2x4)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # Process Noise Covariance Q (4x4)
        dt4 = (dt ** 4) / 4.0 * (sigma_a ** 2)
        dt3 = (dt ** 3) / 2.0 * (sigma_a ** 2)
        dt2 = (dt ** 2) * (sigma_a ** 2)
        self.Q = np.array([
            [dt4, 0.0, dt3, 0.0],
            [0.0, dt4, 0.0, dt3],
            [dt3, 0.0, dt2, 0.0],
            [0.0, dt3, 0.0, dt2]
        ])
        
        # Measurement Noise Covariance R (2x2)
        self.R = np.eye(2) * (sigma_meas ** 2)
        
        # Initial State Estimate and Covariance
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 100.0  # High initial uncertainty

    def predict(self):
        """
        Step 1 & 2: State and Covariance Prediction
        """
        # ==========================================
        # TODO: Implement the Prediction equations
        # self.x = ...
        # self.P = ...
        # ==========================================
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y: np.ndarray):
        """
        Step 3, 4, 5: Measurement Update / Correction
        :param y: (2, 1) measurement vector [x_meas, y_meas]^T
        """
        # ==========================================
        # TODO: Implement Kalman Gain, State Correction, and Covariance Correction
        # 1. Innovation v = y - H @ x
        # 2. Innovation Covariance S = H @ P @ H.T + R
        # 3. Kalman Gain K = P @ H.T @ inv(S)
        # 4. Corrected State self.x = self.x + K @ v
        # 5. Corrected Covariance self.P = (I - K @ H) @ self.P
        # ==========================================
        v = y - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ v
        self.P = (np.eye(4) - K @ self.H) @ self.P
```

---

## 3. Verification & Automated Unit Tests

Run this sanity test block to confirm your matrix algebra and shapes are correct:

```python
# Unit Test Block
kf_test = KalmanFilter2D(dt=0.1, sigma_a=1.0, sigma_meas=2.0)
kf_test.x = np.array([[10.0], [20.0], [2.0], [3.0]])

# Test Prediction
kf_test.predict()
assert kf_test.x.shape == (4, 1), f"Expected shape (4, 1), got {kf_test.x.shape}"
assert np.isclose(kf_test.x[0, 0], 10.2), f"Expected x=10.2, got {kf_test.x[0, 0]}"
assert np.isclose(kf_test.x[1, 0], 20.3), f"Expected y=20.3, got {kf_test.x[1, 0]}"

# Test Update
y_test = np.array([[10.5], [20.1]])
kf_test.update(y_test)
assert kf_test.P.shape == (4, 4), f"Expected covariance shape (4, 4), got {kf_test.P.shape}"
assert np.all(np.linalg.eigvals(kf_test.P) > 0), "Covariance matrix must be positive definite!"

print("✅ ALL UNIT TESTS PASSED!")
```

---

## 4. Full Trajectory Simulation & Results Analysis

Execute this complete tracking simulation:

```python
# 1. Generate Ground Truth 2D Trajectory (Curved Path)
dt = 0.1
time = np.arange(0, 30, dt)
N = len(time)

# Vehicle speeds: vx = 15 m/s, vy = 5 * sin(0.2 * t)
true_vx = 15.0 * np.ones(N)
true_vy = 8.0 * np.sin(0.3 * time)

true_x = np.cumsum(true_vx * dt)
true_y = np.cumsum(true_vy * dt)

# 2. Add Measurement Noise
sigma_meas = 3.0  # 3 meter sensor noise
meas_x = true_x + np.random.normal(0, sigma_meas, N)
meas_y = true_y + np.random.normal(0, sigma_meas, N)

# 3. Run Kalman Filter
kf = KalmanFilter2D(dt=dt, sigma_a=1.5, sigma_meas=sigma_meas)
# Initialize with first noisy measurement
kf.x = np.array([[meas_x[0]], [meas_y[0]], [0.0], [0.0]])

est_x, est_y, est_vx, est_vy = [], [], [], []
pos_cov_x, pos_cov_y = [], []

for k in range(N):
    # Predict
    kf.predict()
    
    # Update with GPS fix
    y_k = np.array([[meas_x[k]], [meas_y[k]]])
    kf.update(y_k)
    
    # Save results
    est_x.append(kf.x[0, 0])
    est_y.append(kf.x[1, 0])
    est_vx.append(kf.x[2, 0])
    est_vy.append(kf.x[3, 0])
    pos_cov_x.append(kf.P[0, 0])
    pos_cov_y.append(kf.P[1, 1])

# 4. Compute Evaluation Metrics
rmse_pos = np.sqrt(np.mean((true_x - np.array(est_x))**2 + (true_y - np.array(est_y))**2))
raw_noise_rmse = np.sqrt(np.mean((true_x - meas_x)**2 + (true_y - meas_y)**2))
print(f"Raw GPS Sensor RMSE:      {raw_noise_rmse:.3f} m")
print(f"Kalman Filter Track RMSE: {rmse_pos:.3f} m (Error reduced by {((raw_noise_rmse - rmse_pos)/raw_noise_rmse)*100:.1f}%)")

import plotly.graph_objects as go

# 5. Interactive 2D Trajectory Visualization via Plotly
fig_lab = go.Figure()
fig_lab.add_trace(go.Scatter(x=true_x, y=true_y, mode='lines', line=dict(color='black', width=3), name='Ground Truth Trajectory'))
fig_lab.add_trace(go.Scatter(x=meas_x[::3], y=meas_y[::3], mode='markers', marker=dict(size=5, color='red', opacity=0.4), name='Noisy GPS Detections'))
fig_lab.add_trace(go.Scatter(x=est_x, y=est_y, mode='lines', line=dict(color='blue', width=2, dash='dash'), name='Kalman Filter Estimate'))

fig_lab.update_layout(
    title=f'2D Vehicle Tracking with Linear Kalman Filter (Track RMSE: {rmse_pos:.2f} m vs. GPS: {raw_noise_rmse:.2f} m)',
    xaxis_title='East Position (m)',
    yaxis_title='North Position (m)',
    template='plotly_white',
    height=500
)
fig_lab.show()
```

---

## 5. Lab Reflection & Challenge Experiments

1. **Experiment 1 (Sensor Dropout):** Modify the loop so that between $t = 10\text{ s}$ and $t = 15\text{ s}$, no sensor measurements arrive (`kf.update()` is not called). What happens to the estimated path? What happens to the covariance values?
2. **Experiment 2 (Tuning Tradeoff):** Increase `sigma_a` by $10\times$. Does the estimated path hug the raw measurements closer or further? Why?
3. **Transition to Day 2:** Notice that here the sensor directly reported Cartesian $[x, y]^T$. What if the sensor is a Radar or LiDAR on a drone reporting **Range $r$ and Bearing angle $\phi$**? That nonlinear observation problem is the subject of **Day 2 & Mini-Project 1**!
