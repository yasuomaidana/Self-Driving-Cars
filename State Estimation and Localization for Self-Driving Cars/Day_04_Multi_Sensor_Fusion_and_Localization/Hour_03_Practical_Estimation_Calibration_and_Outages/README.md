---
title: "Hour 3: Practical Estimation: Calibration, Outliers & Dropouts"
tags:
  - hour-3
  - sensor-calibration
  - outlier-rejection
  - mahalanobis-distance
  - gps-outages
  - production-robotics
created: 2026-09-08
---

# Hour 3: Practical Estimation: Calibration, Outliers & Dropouts

> [!abstract] Key Learning Objectives
> 1. Understand extrinsic spatial calibration ($\mathbf{C}_{li}, \mathbf{t}_i^l$) and compensate for sensor lever arms.
> 2. Implement Chi-Square ($\chi^2$) Mahalanobis distance gating to reject spurious sensor outliers.
> 3. Analyze filter covariance behavior during sensor dropouts (GPS denial in tunnels).
> 4. Design autonomous vehicle safety thresholds based on real-time estimation uncertainty.

---

## 1. Extrinsic Sensor Calibration & Lever-Arm Offsets

In theoretical papers, sensors are drawn as mathematical points located at the exact origin of the vehicle $[0, 0, 0]^T$.
**In real vehicles, sensors are physically distributed across the body:**
* The **IMU** is mounted near the vehicle center of gravity (inside the central console).
* The **GNSS antenna** is mounted high on the roof for maximum satellite visibility.
* The **LiDAR** is mounted on the front bumper or roof rack.

```
       [GNSS Antenna] (Roof)
            |
            |   t_lever (1.2m Up, 0.5m Back)
            v
     +--------------+
     |  [IMU Body]  | (Center of Gravity)
     +--------------+
            ^
            |   t_lidar (0.8m Forward, 0.3m Down)
            |
      [Front LiDAR] (Bumper)
```

### 1.1 Extrinsic Transformation Matrix
To fuse LiDAR or GPS measurements into the IMU frame, we must apply a rigid 3D transformation:
$$\mathbf{p}_{IMU} = \mathbf{C}_{li} \mathbf{p}_{LiDAR} + \mathbf{t}_i^l$$

Where:
* $\mathbf{C}_{li} \in SO(3)$ is the $3\times3$ rotation matrix aligning the sensor axes with the vehicle axes.
* $\mathbf{t}_i^l \in \mathbb{R}^3$ is the translation vector from the vehicle origin to the sensor.

### 1.2 The Lever-Arm Effect
Because the GPS antenna is mounted at distance $\mathbf{r}_{lever}$ from the IMU, when the vehicle rotates at angular rate $\boldsymbol{\omega}$, the antenna experiences an extra tangential velocity:
$$\mathbf{v}_{antenna} = \mathbf{v}_{IMU} + \boldsymbol{\omega} \times \mathbf{r}_{lever}$$

If lever-arm offsets are ignored, a sharp turn will cause the filter to observe false position jumps and velocity discrepancies, degrading the filter covariance!

---

## 2. Outlier Rejection: Mahalanobis Distance Gating

Sensors produce erroneous measurements:
* A GPS receiver in an urban canyon receives a multipath reflection off a glass skyscraper, jumping by $50\text{ meters}$ for a single frame.
* A LiDAR beam strikes airborne dust or exhaust vapor, producing an obstacle measurement in empty space.

If the Kalman filter blindly updates on these spurious measurements, the state estimate will violently jump. We need an **algorithmic gatekeeper**.

### 2.1 The Chi-Square ($\chi^2$) Innovation Gate
Recall the measurement innovation $\boldsymbol{\nu}_k$ and its expected covariance $\mathbf{S}_k$:
$$\boldsymbol{\nu}_k = \mathbf{y}_k - \mathbf{h}(\check{\mathbf{x}}_k)$$
$$\mathbf{S}_k = \mathbf{H} \check{\mathbf{P}}_k \mathbf{H}^T + \mathbf{R}$$

We define the **Squared Mahalanobis Distance** $d_M^2$:
$$d_M^2 = \boldsymbol{\nu}_k^T \mathbf{S}_k^{-1} \boldsymbol{\nu}_k$$

Under normal Gaussian conditions, $d_M^2$ is distributed according to a **Chi-Square distribution** with $m$ degrees of freedom ($\chi_m^2$, where $m = \dim(\mathbf{y})$):
* For a 3D position measurement ($m=3$):
  * $95\%$ of valid measurements have $d_M^2 \le 7.81$.
  * $99\%$ of valid measurements have $d_M^2 \le 11.34$.
  * $99.9\%$ of valid measurements have $d_M^2 \le 16.27$.

```
Probability Density
    ^
    |    ...---...  Valid Measurements (d_M^2 <= 11.34)
    |  .'         '.   --> ACCEPT & UPDATE
    | /             \
----+----------------\-----------------------------|---------------------> d_M^2
   0                 11.34                       16.27
                                              Outlier Region (d_M^2 > 11.34)
                                              --> REJECT MEASUREMENT!
```

> [!important] Gating Rule
> Before performing a measurement update:
> ```python
> d_M_sq = innovation.T @ np.linalg.inv(S) @ innovation
> if d_M_sq > 11.34:
>     # Reject measurement as an outlier!
>     # Skip update step; do not corrupt state or covariance!
>     continue
> ```

---

## 3. Sensor Dropouts & Loss of GPS (The Tunnel Scenario)

What happens when an autonomous vehicle enters a $2\text{-kilometer}$ highway tunnel?
* GPS signal is instantly lost ($0\text{ satellites in view}$).
* The filter must rely purely on the **Prediction step** (IMU integration and wheel odometry).

### 3.1 What Happens to the Filter Internally?
1. **State Propagation:** The nominal state $(\mathbf{p}, \mathbf{v}, \mathbf{q})$ is propagated using strapdown equations.
2. **Covariance Expansion:** Because no measurements contract $\mathbf{P}$, the covariance expands with every step:
   $$\check{\mathbf{P}}_k = \mathbf{F}_{k-1} \check{\mathbf{P}}_{k-1} \mathbf{F}_{k-1}^T + \mathbf{Q}$$
   * Position variance grows **quadratically with time** ($P_{xx}, P_{yy} \sim t^2$).
   * If gyros are high quality, heading uncertainty $\delta\phi_z$ grows slowly and linearly ($\sim t$).

```
Position Covariance 3-Sigma Envelope
  ^
  |                                        / Tunnel Exit (GPS Reacquired)
  |                       .---'''''''----./  Uncertainty contracts immediately!
  |                 .---''
  |           .---''      Inside Tunnel:
  |     .---''            Uncertainty grows quadratically
  |---''
--+------------------------------------------------------------> Time
  0s                      15s                                  30s
```

### 3.2 Safety Gating in Self-Driving Cars
The autonomous vehicle's motion planner monitors the trace of the position covariance matrix:
$$\sigma_{total} = \sqrt{P_{xx} + P_{yy}}$$
* $\sigma_{total} < 0.2\text{ m}$: Full autonomous lane centering active.
* $0.2\text{ m} \le \sigma_{total} < 0.8\text{ m}$: Degraded mode; vehicle increases following distance.
* $\sigma_{total} \ge 1.0\text{ m}$: **Disengagement threshold!** Vehicle triggers emergency hazard lights and initiates a safe stop in the lane.

---

## 4. Python Implementation: Mahalanobis Gating & Extrinsics

```python
import numpy as np

# 1. Extrinsic Calibration Transformation: LiDAR to IMU
# Rotation: 5 degree tilt around X, Y, Z
# Translation: 0.5m forward, 0.1m right, 0.5m down
C_li = np.array([
    [ 0.99376, -0.09722,  0.05466],
    [ 0.09971,  0.99401, -0.04475],
    [-0.04998,  0.04992,  0.99750]
])
t_i_li = np.array([0.5, 0.1, -0.5])

# Transform a raw LiDAR detection point to vehicle IMU frame
raw_lidar_point = np.array([15.0, 2.0, -0.2])  # 15m ahead, 2m right
p_imu = C_li @ raw_lidar_point + t_i_li
print("Transformed Point in IMU Frame:", np.round(p_imu, 3))

# 2. Mahalanobis Distance Outlier Rejection Function
def validate_measurement(innovation, S, threshold=11.34):
    """
    Computes Mahalanobis distance to gate 3D position observations.
    threshold = 11.34 corresponds to 99% Chi-Square confidence (df=3).
    """
    d_M_sq = innovation.T @ np.linalg.inv(S) @ innovation
    is_valid = d_M_sq <= threshold
    return is_valid, d_M_sq.item()

# Test with a normal measurement
S_matrix = np.diag([0.25, 0.25, 0.5])  # expected variance
normal_innov = np.array([[0.3], [-0.2], [0.1]])

valid, dist = validate_measurement(normal_innov, S_matrix)
print(f"\nNormal Measurement: d_M^2 = {dist:.2f} -> Valid: {valid}")

# Test with a spurious multipath outlier (15 meter error)
outlier_innov = np.array([[15.0], [-8.0], [2.0]])
valid_outlier, dist_outlier = validate_measurement(outlier_innov, S_matrix)
print(f"Spurious Outlier:   d_M^2 = {dist_outlier:.2f} -> Valid: {valid_outlier}")
assert not valid_outlier, "Outlier should have been rejected!"
print("✅ Outlier gate successfully rejected corrupt measurement!")
```

---

## 5. Self-Assessment & Checkpoint Questions

1. **What happens if extrinsic translation $\mathbf{t}_i^l$ is mismeasured by $20\text{ cm}$?**
   * *Answer:* The filter will have a permanent structural bias. During turns, the lever-arm mismatch will cause the filter to see unexpected position shifts that do not match the gyroscope's angular rates, driving up innovation residuals and inflating covariance.

2. **If a sensor fails completely and sends values of `NaN` or `[0, 0, 0]`, how does Mahalanobis gating protect the vehicle?**
   * *Answer:* When the vehicle is moving at $(100, 200, 0)$, a sudden measurement of $(0, 0, 0)$ produces a massive innovation $\boldsymbol{\nu} = [-100, -200, 0]^T$. The Mahalanobis distance $d_M^2$ will evaluate to thousands ($d_M^2 \gg 11.34$), causing the gatekeeper to immediately discard the corrupt frame.

3. **When exiting a tunnel and regaining GPS, why doesn't the vehicle state jump instantly to the GPS position?**
   * *Answer:* The Kalman gain $\mathbf{K}$ balances the expanded covariance $\check{\mathbf{P}}$ against the measurement noise $\mathbf{R}$. While the correction will be significant, the filter acts as a low-pass filter, smoothly pulling the estimate back to the true trajectory over several consecutive frames rather than teleporting discontinuously.
