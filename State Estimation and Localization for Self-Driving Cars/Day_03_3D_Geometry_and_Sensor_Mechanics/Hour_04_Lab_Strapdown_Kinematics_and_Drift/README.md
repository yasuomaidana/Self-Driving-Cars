---
title: "Hour 4: Applied Programming Lab (Strapdown Kinematics & Drift Analysis)"
tags:
  - hour-4
  - lab
  - python
  - imu
  - dead-reckoning
  - strapdown
created: 2026-09-08
---

# Hour 4: Applied Programming Lab (Strapdown Kinematics & Drift Analysis)

> [!abstract] Lab Objective
> Implement an open-loop **3D Strapdown Inertial Navigation** algorithm in Python using real automotive IMU data (`pt1_data.pkl`). You will integrate high-rate specific forces and angular velocities using quaternion kinematics, measure the rate of quadratic position divergence, and experience why autonomous vehicles cannot navigate on dead reckoning alone.

---

## 1. Lab Setup & Dataset Overview

We will use real vehicle logs from the course workspace:
* **Dataset Path:** `Vehicle State Estimation on a Roadway/data/pt1_data.pkl`
* **Rotation Library:** `Vehicle State Estimation on a Roadway/rotations.py`

### 1.1 What the Data Contains
```python
with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

gt = data['gt']        # Ground truth (p: position, v: velocity, r: orientation)
imu_f = data['imu_f']  # 100 Hz specific force [f_x, f_y, f_z] in vehicle frame
imu_w = data['imu_w']  # 100 Hz rotational velocity [w_x, w_y, w_z] in vehicle frame
```

---

## 2. Mathematical Recipe for Strapdown Integration

For each time step $k = 1, 2, \dots, N$ with $\Delta t = t_k - t_{k-1}$:

1. **Rotate Specific Force to Navigation Frame:**
   Convert current orientation quaternion $\mathbf{q}_{k-1}$ to rotation matrix $\mathbf{C}_{ns}$:
   $$\mathbf{f}_{nav} = \mathbf{C}_{ns} \mathbf{f}_{k-1}$$

2. **Add Gravity Compensation:**
   In ENU (East-North-Up), gravity acts downward: $\mathbf{g} = [0, 0, -9.81]^T$:
   $$\mathbf{a}_{net} = \mathbf{f}_{nav} + \mathbf{g}$$

3. **Position Integration (Second-Order Kinematics):**
   $$\mathbf{p}_k = \mathbf{p}_{k-1} + \Delta t \cdot \mathbf{v}_{k-1} + \frac{\Delta t^2}{2} \cdot \mathbf{a}_{net}$$

4. **Velocity Integration:**
   $$\mathbf{v}_k = \mathbf{v}_{k-1} + \Delta t \cdot \mathbf{a}_{net}$$

5. **Attitude Quaternion Propagation:**
   Convert angular rate vector $\boldsymbol{\omega}_{k-1}\Delta t$ to incremental quaternion and right-multiply:
   $$\Delta\mathbf{q} = \operatorname{Quaternion}(\text{axis\_angle} = \boldsymbol{\omega}_{k-1}\Delta t)$$
   $$\mathbf{q}_k = \mathbf{q}_{k-1} \otimes \Delta\mathbf{q}$$

---

## 3. Student Implementation Code

Create a Python script or notebook in `Vehicle State Estimation on a Roadway/` and complete the integration loop:

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# Ensure rotations module is accessible
from rotations import Quaternion

# 1. Load Data
with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']

N = imu_f.data.shape[0]
print(f"Total IMU Samples: {N} (approx {N/100:.1f} seconds of driving)")

# 2. Initialize Arrays for Dead Reckoning
p_dr = np.zeros([N, 3])  # Position estimates
v_dr = np.zeros([N, 3])  # Velocity estimates
q_dr = np.zeros([N, 4])  # Quaternion estimates

# Initialize with exact ground truth at k = 0
p_dr[0] = gt.p[0]
v_dr[0] = gt.v[0]
q_dr[0] = Quaternion(euler=gt.r[0]).to_numpy()

g = np.array([0.0, 0.0, -9.81])  # Gravity vector in ENU navigation frame

# 3. Main Strapdown Integration Loop (Pure Dead Reckoning)
for k in range(1, N):
    dt = imu_f.t[k] - imu_f.t[k - 1]
    
    # Current quaternion as rotation matrix
    C_ns = Quaternion(*q_dr[k - 1]).to_mat()
    
    # Specific force in sensor frame rotated to navigation frame
    f_nav = C_ns @ imu_f.data[k - 1]
    a_net = f_nav + g
    
    # -------------------------------------------------------------
    # TODO: Implement position, velocity, and quaternion updates
    # -------------------------------------------------------------
    p_dr[k] = p_dr[k - 1] + dt * v_dr[k - 1] + 0.5 * (dt ** 2) * a_net
    v_dr[k] = v_dr[k - 1] + dt * a_net
    
    delta_q = Quaternion(axis_angle=imu_w.data[k - 1] * dt)
    q_dr[k] = Quaternion(*q_dr[k - 1]).quat_mult_right(delta_q).to_numpy()

print("✅ Dead reckoning integration complete!")
```

---

## 4. Visualizing the Catastrophic Drift

Run this visualization block to compare open-loop dead reckoning against ground truth:

```python
# 1. Compute Drift Error
drift_error = np.linalg.norm(p_dr - gt.p, axis=1)
time_axis = (imu_f.t - imu_f.t[0])

# 2. Plot 3D Trajectory Comparison
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(gt.p[:, 0], gt.p[:, 1], gt.p[:, 2], 'g-', linewidth=2, label='Ground Truth Path')
ax1.plot(p_dr[:, 0], p_dr[:, 1], p_dr[:, 2], 'r--', linewidth=2, label='IMU Dead Reckoning')
ax1.set_xlabel('Easting [m]')
ax1.set_ylabel('Northing [m]')
ax1.set_zlabel('Up [m]')
ax1.set_title('3D Trajectory: True vs. Standalone IMU')
ax1.legend()

# 3. Plot Position Error vs Time
ax2 = fig.add_subplot(122)
ax2.plot(time_axis, drift_error, 'r-', linewidth=2)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Total Position Error [m]')
ax2.set_title('Inertial Drift Over Time')
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"Drift Error after 10 seconds: {drift_error[1000]:.2f} meters")
print(f"Drift Error after 30 seconds: {drift_error[3000]:.2f} meters")
print(f"Final Drift Error:            {drift_error[-1]:.2f} meters")
```

---

## 5. Lab Reflection Questions

1. **Why does the estimated vertical position ($Z$) usually diverge faster than horizontal positions ($X$ and $Y$)?**
   * *Answer:* Because the vertical channel must continuously subtract the massive acceleration of gravity ($9.81\text{ m/s}^2$). Any $0.1\%$ scale-factor error in the $Z$-axis accelerometer translates to an uncompensated residual acceleration of $\approx 0.01\text{ m/s}^2$, launching the car into space or subterranean depths.

2. **At what point does the vehicle drift completely out of a standard $3.5\text{ m}$ highway lane?**
   * *Answer:* Usually within the first $8\text{--}12\text{ seconds}$ of open-loop driving.

3. **How does this motivate Day 4?**
   * *Answer:* An IMU alone is useless after 10 seconds. But an IMU combined with **GPS position updates** in an **Extended Kalman Filter** will hold accuracy under $0.5\text{ meters}$ indefinitely! This sets up **Mini-Project 2 on Day 4**!
