---
title: "Hour 3: Target Tracking Models & Filter Validation Metrics"
tags:
  - hour-3
  - target-tracking
  - motion-models
  - nees
  - error-ellipses
  - filter-consistency
created: 2026-09-08
---

# Hour 3: Target Tracking Models & Filter Validation Metrics

> [!abstract] Key Learning Objectives
> 1. Formulate dynamic models for tracking target obstacles: Constant Velocity (CV), Constant Acceleration (CA), and Coordinated Turn (CT).
> 2. Model automotive Radar and LiDAR measurements including Range, Bearing, and Doppler Range-Rate.
> 3. Implement robust angle normalization ($[-\pi, \pi]$) to prevent filter divergence.
> 4. Evaluate filter health using statistical consistency metrics: NEES, NIS, and 2D Covariance Confidence Ellipses.

---

## 1. The Autonomous Object Tracking Problem

In self-driving cars, **Object Tracking** is the task of estimating the trajectory, speed, and future heading of surrounding dynamic obstacles (lead vehicles, pedestrians, cyclists).

```mermaid
flowchart LR
    Sensors["Onboard Sensors (Radar, LiDAR, Camera)"] -->|Raw Detections| DA["Data Association & Clustering"]
    DA -->|Associated Measurements [r, phi]| Tracker["Object Tracking Filter (EKF)"]
    Tracker -->|Estimated State [x, y, vx, vy]| Trajectory["Obstacle Trajectory Prediction"]
    Trajectory --> Controller["Collision Avoidance & Adaptive Cruise Control"]
```

Unlike ego-localization (where we have access to the vehicle's own throttle and steering commands $\mathbf{u}_k$), **we do not know the target's control inputs**. We do not know when a pedestrian will stop or when a lead vehicle will brake. We must model their motion using **stochastic kinematic models**.

---

## 2. Dynamic Motion Models for Target Tracking

### 2.1 Constant Velocity (CV) Model
Assumes the target moves at approximately constant linear velocity with random acceleration disturbances:
$$\mathbf{x} = \begin{bmatrix} x \\ y \\ \dot{x} \\ \dot{y} \end{bmatrix}, \quad \mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
* **Best for:** Pedestrians, cars driving steadily on a straight highway.

### 2.2 Constant Acceleration (CA) Model
Extends the state to include target acceleration $[\ddot{x}, \ddot{y}]^T$, driven by random jerk:
$$\mathbf{x} = \begin{bmatrix} x \\ y \\ \dot{x} \\ \dot{y} \\ \ddot{x} \\ \ddot{y} \end{bmatrix}, \quad \mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 & \frac{\Delta t^2}{2} & 0 \\ 0 & 1 & 0 & \Delta t & 0 & \frac{\Delta t^2}{2} \\ 0 & 0 & 1 & 0 & \Delta t & 0 \\ 0 & 0 & 0 & 1 & 0 & \Delta t \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}$$
* **Best for:** Stop-and-go urban traffic, hard braking scenarios.

### 2.3 Coordinated Turn (CT / CTRV) Model
Assumes the vehicle moves at speed $v$ while turning at an angular yaw rate $\omega$:
$$\mathbf{x} = \begin{bmatrix} x \\ y \\ v \\ \theta \\ \omega \end{bmatrix}, \quad \dot{\mathbf{x}} = \begin{bmatrix} v \cos\theta \\ v \sin\theta \\ 0 \\ \omega \\ 0 \end{bmatrix}$$
* Integrating across $\Delta t$ when $\omega \neq 0$:
  $$x_k = x_{k-1} + \frac{v}{\omega}(\sin(\theta + \omega \Delta t) - \sin\theta)$$
  $$y_k = y_{k-1} + \frac{v}{\omega}(-\cos(\theta + \omega \Delta t) + \cos\theta)$$
* **Best for:** Vehicles turning at intersections or navigating roundabouts.

---

## 3. Automotive Sensor Measurement Models

### 3.1 Automotive Radar
Radar provides three direct measurements of a target:
$$\mathbf{y}_{Radar} = \begin{bmatrix} r \\ \phi \\ \dot{r} \end{bmatrix} = \begin{bmatrix} \text{Range (Distance)} \\ \text{Azimuth / Bearing (Angle)} \\ \text{Radial Velocity (Doppler Velocity)} \end{bmatrix}$$

Measurement equations relative to ego-vehicle:
$$r = \sqrt{x^2 + y^2}$$
$$\phi = \operatorname{atan2}(y, x)$$
$$\dot{r} = \frac{x \dot{x} + y \dot{y}}{\sqrt{x^2 + y^2}}$$

### 3.2 Automotive LiDAR
LiDAR sensors return point clouds. Perception clustering algorithms compute the 3D bounding box centroid:
$$\mathbf{y}_{LiDAR} = \begin{bmatrix} r \\ \phi \end{bmatrix} \quad \text{or directly in Cartesian} \quad \begin{bmatrix} x_{meas} \\ y_{meas} \end{bmatrix}$$

---

## 4. The Angle Wrapping Problem: `wraptopi`

One of the most frequent bugs in autonomous vehicle engineering occurs in angular measurement residuals:
$$\boldsymbol{\nu}_\phi = \phi_{meas} - \check{\phi}_{pred}$$

Angles are periodic on the circle: $-\pi \equiv +\pi$.
* Suppose the target is at bearing $+179^\circ$ ($+3.124\text{ rad}$).
* The filter predicts the target is at $-179^\circ$ ($-3.124\text{ rad}$).
* Physically, the prediction is only **$2^\circ$ away**!
* Naive subtraction gives:
  $$\boldsymbol{\nu}_\phi = +3.124 - (-3.124) = +6.248\text{ rad} \approx +358^\circ$$
* The Kalman filter sees a **massive $358^\circ$ error**, multiplies it by $\mathbf{K}$, and shoots the state estimate into oblivion!

> [!important] The Robust Normalization Function
> Always wrap all angular innovations to $[-\pi, \pi]$:
> ```python
> def wraptopi(angle):
>     """Wraps an angle in radians to the range [-pi, pi]"""
>     return (angle + np.pi) % (2 * np.pi) - np.pi
> ```

---

## 5. Filter Diagnostics and Validation Metrics

How do you prove to a safety certification auditor that your Kalman filter is working properly? You use formal statistical metrics.

### 5.1 Root Mean Square Error (RMSE)
Measures average Euclidean distance error against ground truth:
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{k=1}^N \|\mathbf{x}_{true, k} - \hat{\mathbf{x}}_k\|^2}$$

### 5.2 Normalized Estimation Error Squared (NEES)
NEES evaluates whether the filter's **covariance $\mathbf{P}_k$ is statistically honest**:
$$\epsilon_k = (\mathbf{x}_{true, k} - \hat{\mathbf{x}}_k)^T \hat{\mathbf{P}}_k^{-1} (\mathbf{x}_{true, k} - \hat{\mathbf{x}}_k)$$

* If the filter is consistent, $\epsilon_k$ is a random variable distributed according to a **Chi-Square distribution** with $n$ degrees of freedom ($\chi_n^2$):
  $$\mathbb{E}[\epsilon_k] = n \quad (\text{where } n \text{ is the state dimension})$$
* For a 4D state ($n=4$), the 95% acceptance region is roughly $[1.0, 9.5]$.
* If $\epsilon_k \gg 9.5$, the filter is **overconfident** (true error is much larger than $\mathbf{P}_k$ reports).

```
NEES \epsilon_k
  ^
  |        .    .            DIVERGENT / OVERCONFIDENT FILTER
  |       / \  / \  /\  ... (Exceeds upper Chi-Square bound)
--+------/---\/---\/--\-------------------------------- 95% Upper Bound
  |     /            \
  |----/--------------\-------------------------------- 95% Lower Bound
  |                       HEALTHY / CONSISTENT FILTER
  +----------------------------------------------------> Time (k)
```

### 5.3 Normalized Innovation Squared (NIS)
NIS tests the sensor innovation $\boldsymbol{\nu}_k$ against the innovation covariance $\mathbf{S}_k$:
$$\gamma_k = \boldsymbol{\nu}_k^T \mathbf{S}_k^{-1} \boldsymbol{\nu}_k \sim \chi_m^2 \quad (\text{where } m \text{ is the measurement dimension})$$
* Used to detect **sensor outliers** (e.g., radar reflections from a beverage can). If $\gamma_k > \chi_{m, 0.99}^2$, reject the measurement!

---

## 6. Visualizing 2D Uncertainty: Covariance Confidence Ellipses

To draw the spatial uncertainty ellipse around the target's estimated $[x, y]$ position:
1. Extract the $2 \times 2$ position submatrix:
   $$\mathbf{P}_{pos} = \begin{bmatrix} P_{xx} & P_{xy} \\ P_{yx} & P_{yy} \end{bmatrix}$$
2. Compute eigenvalues $\lambda_1, \lambda_2$ and eigenvectors $\mathbf{v}_1, \mathbf{v}_2$.
3. The eigenvalues represent the squared radii of the ellipse.
   * Semi-major axis length = $3\sqrt{\lambda_1}$ ($3\sigma$ bound)
   * Semi-minor axis length = $3\sqrt{\lambda_2}$
   * Orientation angle $\alpha = \operatorname{atan2}(v_{1,y}, v_{1,x})$

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_covariance_ellipse(x, y, P_pos, ax, n_std=3.0, **kwargs):
    """
    Plots a covariance confidence ellipse around (x, y).
    """
    # 1. Eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(P_pos)
    
    # 2. Sort so largest eigenvalue is first
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]
    
    # 3. Calculate axis lengths and tilt angle
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvals)
    
    # 4. Draw Ellipse
    ellipse = Ellipse(xy=(x, y), width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

# Demo plot
fig, ax = plt.subplots(figsize=(7, 7))
P_demo = np.array([[4.0, 2.5],
                   [2.5, 3.0]])
plot_covariance_ellipse(10.0, 15.0, P_demo, ax, n_std=3.0, 
                        edgecolor='red', facecolor='pink', alpha=0.4, label='$3\sigma$ Uncertainty')
ax.scatter([10.0], [15.0], color='red', marker='x', s=100, label='Target Position')
ax.set_xlim(0, 20)
ax.set_ylim(5, 25)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Target State 3-Sigma Covariance Ellipse')
ax.legend()
ax.grid(True)
plt.show()
```

---

## 7. Self-Assessment & Checkpoint Questions

1. **Why does a constant velocity model work reasonably well for tracking human-driven cars, even though human drivers do accelerate?**
   * *Answer:* Because the acceleration is treated as a zero-mean stochastic disturbance captured by the process noise covariance $\mathbf{Q}$. Over short sampling times ($\Delta t = 0.05\text{--}0.1\text{ s}$), vehicle acceleration cannot change speed instantaneously due to physical vehicle inertia.

2. **If your filter's average NEES is $\bar{\epsilon} = 15.2$ for a 4D state model, what does this indicate?**
   * *Answer:* The expected value for a healthy 4D filter is $4.0$. A value of $15.2$ proves the filter is severely **overconfident**—the true errors are much larger than the filter's covariance matrix claims. The engineer should increase process noise $\mathbf{Q}$ or measurement noise $\mathbf{R}$.

3. **What is the difference between NEES and NIS?**
   * *Answer:* NEES evaluates the **internal state error** $(\mathbf{x}_{true} - \hat{\mathbf{x}})$ and requires ground truth data (used in offline testing). NIS evaluates the **sensor innovation residual** $(\mathbf{y} - \check{\mathbf{y}})$ and does not require ground truth, making it suitable for **real-time online health monitoring and outlier rejection**.
