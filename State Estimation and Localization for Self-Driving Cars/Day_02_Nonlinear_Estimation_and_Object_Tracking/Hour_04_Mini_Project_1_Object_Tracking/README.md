---
title: "Hour 4: Mini-Project 1 — 2D Object Tracking using Kalman Filters"
tags:
  - hour-4
  - mini-project-1
  - project
  - extended-kalman-filter
  - object-tracking
created: 2026-09-08
---

# Hour 4: Mini-Project 1 — 2D Object Tracking using Kalman Filters

> [!abstract] Project Mission
> In this first mini-project, you will design, implement, and tune an **Extended Kalman Filter (EKF)** in Python to recursively estimate the 2D trajectory of a moving vehicle or target using noisy odometry inputs and non-linear range-bearing sensor observations.

---

## 1. Project Background & Data Files

The vehicle is moving across a 2D plane. It receives:
1. **Control / Odometry Inputs:** Forward linear velocity $v_k$ and rotational angular velocity $\omega_k$.
2. **Sensor Measurements:** A range-and-bearing sensor (e.g. LiDAR or Radar) detects landmarks with known global positions $\mathbf{l} = [x_l, y_l]^T$, outputting:
   * Range $r_k$ (meters)
   * Bearing $\phi_k$ (radians relative to the sensor heading)

The data file is located in the workspace at:
`Estimating a Vehicle Trajectory/data/data.pickle`

---

## 2. Mathematical System Model

```
                     Landmark l = [x_l, y_l]
                            *
                           /|
                          / |
               Range r_k /  |
                        /   |
                       /    |  Bearing \phi_k (relative to heading)
                      / \phi|
                     v      |
       Robot Center  O======+=== Sensor (offset d from center)
                     \ theta
                      \
                       \---> Global X
```

### 2.1 Vehicle Motion Model
The state is the 2D pose: $\mathbf{x}_k = [x_k, y_k, \theta_k]^T \in \mathbb{R}^3$.
Driven by linear velocity $v_k$ and angular velocity $\omega_k$ over time step $\Delta t = t_k - t_{k-1}$:

$$\mathbf{x}_k = \mathbf{x}_{k-1} + \Delta t \begin{bmatrix} \cos\theta_{k-1} & 0 \\ \sin\theta_{k-1} & 0 \\ 0 & 1 \end{bmatrix} \left( \begin{bmatrix} v_k \\ \omega_k \end{bmatrix} + \mathbf{w}_k \right)$$

* Process noise covariance: $\mathbf{Q} = \operatorname{diag}(\sigma_v^2, \sigma_\omega^2)$.

### 2.2 Sensor Measurement Model
The sensor is mounted a distance $d$ in front of the vehicle center. The measurement vector for landmark $l = [x_l, y_l]^T$ is:

$$\mathbf{y}_k^l = \begin{bmatrix} r_k^l \\ \phi_k^l \end{bmatrix} = \begin{bmatrix} \sqrt{d_x^2 + d_y^2} \\ \operatorname{atan2}(d_y, d_x) - \theta_k \end{bmatrix} + \mathbf{n}_k^l$$

Where the sensor-to-landmark offsets are:
$$d_x = x_l - x_k - d \cos\theta_k$$
$$d_y = y_l - y_k - d \sin\theta_k$$

* Measurement noise covariance: $\mathbf{R} = \operatorname{diag}(\sigma_r^2, \sigma_\phi^2)$.

---

## 3. Step-by-Step Implementation Guide

Open your Jupyter environment or create a script based on `Estimating a Vehicle Trajectory/Estimating a Vehicle Trajectory.ipynb`.

### Step 1: Angle Normalization Utility
Because bearings jump at $\pm\pi$, implement `wraptopi()`:
```python
def wraptopi(angle: float) -> float:
    """Normalizes an angle to the range [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi
```

### Step 2: Implement the Measurement Update Function
For each landmark measurement $l$, compute the Jacobian $\mathbf{H}_k$, Kalman Gain $\mathbf{K}_k$, state correction, and covariance correction:

```python
def measurement_update(lk, rk, bk, P_check, x_check, d, cov_y):
    """
    lk: Landmark coordinate [x_l, y_l]
    rk: Measured range
    bk: Measured bearing
    P_check: (3, 3) predicted covariance
    x_check: (3, 1) predicted state [x, y, theta]^T
    d: sensor offset from vehicle center
    cov_y: (2, 2) measurement noise covariance R
    """
    x_k = x_check[0, 0]
    y_k = x_check[1, 0]
    theta_k = wraptopi(x_check[2, 0])
    
    x_l, y_l = lk[0], lk[1]
    
    # Compute relative offsets
    d_x = x_l - x_k - d * np.cos(theta_k)
    d_y = y_l - y_k - d * np.sin(theta_k)
    
    r = np.sqrt(d_x**2 + d_y**2)
    phi = wraptopi(np.arctan2(d_y, d_x) - theta_k)
    
    # 1. Compute Measurement Jacobian H_k (2 x 3)
    H_k = np.zeros((2, 3))
    H_k[0, 0] = -d_x / r
    H_k[0, 1] = -d_y / r
    H_k[0, 2] = d * (d_x * np.sin(theta_k) - d_y * np.cos(theta_k)) / r
    
    H_k[1, 0] = d_y / (r ** 2)
    H_k[1, 1] = -d_x / (r ** 2)
    H_k[1, 2] = -1.0 - d * (d_y * np.sin(theta_k) + d_x * np.cos(theta_k)) / (r ** 2)
    
    # 2. Predicted measurement and actual measurement
    y_pred = np.array([[r], [phi]])
    y_meas = np.array([[rk], [wraptopi(bk)]])
    
    # 3. Compute Innovation with angle wrapping on bearing
    innovation = y_meas - y_pred
    innovation[1, 0] = wraptopi(innovation[1, 0])
    
    # 4. Compute Kalman Gain
    S = H_k @ P_check @ H_k.T + cov_y
    K_k = P_check @ H_k.T @ np.linalg.inv(S)
    
    # 5. State and Covariance Correction
    x_hat = x_check + K_k @ innovation
    x_hat[2, 0] = wraptopi(x_hat[2, 0])
    P_hat = (np.eye(3) - K_k @ H_k) @ P_check
    
    return x_hat, P_hat
```

### Step 3: Implement the Main Filter Loop
Iterate over the timestamps, performing the kinematic prediction step and landmark correction:

```python
# Unpack data
import pickle
with open('data/data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']
v = data['v']
om = data['om']
b = data['b']
r = data['r']
l = data['l']
d = data['d']

# Noise variances
v_var = 0.004
om_var = 0.008
r_var = 0.001
b_var = 0.0005

Q_km = np.diag([v_var, om_var])
cov_y = np.diag([r_var, b_var])

# Initialize arrays
x_est = np.zeros((len(t), 3))
P_est = np.zeros((len(t), 3, 3))

x_est[0] = np.array([data['x_init'], data['y_init'], data['th_init']])
P_est[0] = np.diag([0.01, 0.01, 0.01])

# Run Main Filter Loop
x_check = x_est[0].reshape(3, 1)
P_check = P_est[0]

for k in range(1, len(t)):
    dt = t[k] - t[k-1]
    theta = x_check[2, 0]
    
    # 1. Prediction Step
    # Kinematic propagation
    inp = np.array([[v[k-1]], [om[k-1]]])
    F_u = np.array([
        [np.cos(theta), 0.0],
        [np.sin(theta), 0.0],
        [0.0,           1.0]
    ])
    x_check = x_check + dt * F_u @ inp
    x_check[2, 0] = wraptopi(x_check[2, 0])
    
    # Jacobians F_km and L_km
    F_km = np.array([
        [1.0, 0.0, -dt * v[k-1] * np.sin(theta)],
        [0.0, 1.0,  dt * v[k-1] * np.cos(theta)],
        [0.0, 0.0,  1.0]
    ])
    
    L_km = np.array([
        [dt * np.cos(theta), 0.0],
        [dt * np.sin(theta), 0.0],
        [0.0,                dt ]
    ])
    
    # Propagate uncertainty
    P_check = F_km @ P_check @ F_km.T + L_km @ Q_km @ L_km.T
    
    # 2. Correction Step (Update using each visible landmark)
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check, d, cov_y)
        
    x_est[k] = x_check.flatten()
    P_est[k] = P_check
```

---

## 4. Evaluation and Plotting

Generate the final evaluation figures:

```python
# Plot estimated trajectory vs landmarks
plt.figure(figsize=(10, 8))
plt.plot(x_est[:, 0], x_est[:, 1], 'b-', linewidth=2, label='EKF Estimated Trajectory')
plt.scatter(l[:, 0], l[:, 1], color='red', marker='*', s=120, label='Landmarks')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Mini-Project 1: 2D Target Trajectory Estimation')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Plot orientation and 3-sigma bounds
plt.figure(figsize=(10, 4))
plt.plot(t, x_est[:, 2], 'g-', label='Estimated Heading $\\theta$')
th_sigma = 3 * np.sqrt(P_est[:, 2, 2])
plt.fill_between(t, x_est[:, 2] - th_sigma, x_est[:, 2] + th_sigma, color='green', alpha=0.2, label='$\\pm 3\\sigma$ Bound')
plt.xlabel('Time [s]')
plt.ylabel('Heading [rad]')
plt.title('Heading Angle Estimation with $3\\sigma$ Envelopes')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 5. Mini-Project 1 Grading Rubric

| Criteria | Points | Verification Method |
| :--- | :---: | :--- |
| **Measurement Jacobian $\mathbf{H}_k$** | 25% | Correct mathematical derivation and implementation including sensor offset $d$. |
| **Angle Wrapping** | 15% | All bearing innovations and states normalized to $[-\pi, \pi]$ using `wraptopi`. |
| **Motion Prediction & Jacobians** | 25% | Correct discrete integration of kinematics and $\mathbf{F}_{km}, \mathbf{L}_{km}$ matrices. |
| **Trajectory Estimation Accuracy** | 25% | Estimated trajectory smoothly follows the true course with position error $< 0.5\text{ m}$. |
| **Plots & Diagnostics** | 10% | Clean 2D trajectory plot, heading error plot, and $\pm 3\sigma$ covariance bounds. |
| **Total** | **100%** | Full functional EKF object tracking. |
