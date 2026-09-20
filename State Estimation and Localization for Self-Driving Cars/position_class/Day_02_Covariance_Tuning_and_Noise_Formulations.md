# Day 2: Covariance Tuning & Physical Noise Modeling Guide

In Kalman Filtering (EKF, ES-EKF, UKF), the performance, convergence rate, and statistical consistency of the estimator depend fundamentally on the choice of:
1. **Q (Process Noise Covariance Matrix):** Models the uncertainty in our mathematical motion model (unmodeled physics, actuator disturbances, integration approximations).
2. **R (Measurement Noise Covariance Matrix):** Models the sensor measurement precision and noise characteristics.

---

## 1. General Principles of Covariance Formulation

Because state and sensor variables often have completely different physical units (meters, meters/second, radians), we construct Q and R as diagonal matrices when noise sources are uncoupled:

$$\mathbf{Q} = \operatorname{diag}(\sigma_{x_1}^2, \sigma_{x_2}^2, \dots, \sigma_{x_n}^2), \quad
\mathbf{R} = \operatorname{diag}(\sigma_{y_1}^2, \sigma_{y_2}^2, \dots, \sigma_{y_m}^2)$$

where each $\sigma_i$ is the $1\sigma$ standard deviation in the physical units of that component.

---

## 2. Extended Kalman Filter (EKF) Notebook

In `Day_02_Extended_Kalman_Filter_*.ipynb`:

```python
# Process Noise Covariance
Q = np.diag([0.05**2, 0.05**2, 0.1**2, 0.02**2])

# Measurement Noise Covariance (Radar)
R = np.diag([0.5**2, np.deg2rad(1.0)**2])
```

### Process Noise $\mathbf{Q}$ ($\text{State: } \mathbf{x} = [p_x, p_y, v, \theta]^T$)
* **$\sigma_{p_x} = 0.05\text{ m}, \ \sigma_{p_y} = 0.05\text{ m}$:**
  Represents tire compliance, lateral slip, and the truncation error of 1st-order discrete Euler kinematic integration over a sampling interval $\Delta t = 0.1\text{ s}$.
* **$\sigma_v = 0.1\text{ m/s}$:**
  Represents actuator throttle/braking response lags, powertrain torque ripple, and road gradient disturbances.
* **$\sigma_\theta = 0.02\text{ rad} \approx 1.15^\circ$:**
  Represents steering gear backlash, tire slip angle variations, and crosswinds.

### Measurement Noise $\mathbf{R}$ ($\text{Measurement: } \mathbf{y} = [r, \phi]^T$)
* **$\sigma_r = 0.5\text{ m}$:**
  Radial distance precision of an automotive millimeter-wave radar sensor based on Time-of-Flight (ToF) and chirp bandwidth.
* **$\sigma_\phi = 1.0^\circ = \frac{\pi}{180} \approx 0.0175\text{ rad}$:**
  Azimuth angular resolution governed by the radar antenna array beamwidth.

---

## 3. Error-State Kalman Filter (ES-EKF) Notebook

In `Day_02_Error_State_Kalman_Filter_*.ipynb`:

```python
dt_imu = 0.01  # 100 Hz IMU Sampling Rate
Q_imu = np.diag([0.01**2, 0.01**2, 0.05**2, 0.01**2]) * dt_imu
R_gps = np.diag([1.5**2, 1.5**2])
```

### Process Noise $\mathbf{Q}_{\text{imu}}$ (Continuous-to-Discrete Scaling)
In high-rate inertial navigation (100 Hz, $\Delta t = 0.01\text{ s}$), sensor noise is modeled as **continuous white noise spectral densities**:
$$\mathbf{Q}_k \approx \mathbf{Q}_{\text{continuous}} \cdot \Delta t_{\text{imu}}$$

* **$\sigma_v = 0.05\text{ m/s}/\sqrt{\text{s}}$:**
  Accelerometer Velocity Random Walk (VRW) describing how high-frequency accelerometer noise integrates into velocity uncertainty.
* **$\sigma_\theta = 0.01\text{ rad/s}/\sqrt{\text{s}}$:**
  Rate Gyroscope Angular Random Walk (ARW) describing how gyro noise accumulates into orientation drift.
* **$\Delta t_{\text{imu}} = 0.01\text{ s}$:**
  Scales continuous noise variance over the 10 ms step, ensuring uncertainty growth remains invariant to sampling frequency.

### Measurement Noise $\mathbf{R}_{\text{gps}}$ (10 Hz GPS Fixes)
* **$\sigma_{p_x, \text{gps}} = 1.5\text{ m}, \ \sigma_{p_y, \text{gps}} = 1.5\text{ m}$:**
  Typical horizontal pseudorange standard deviation for civilian single-frequency GPS (without differential RTK corrections).

---

## 4. Unscented Kalman Filter (UKF) Benchmark Notebook

In `Day_02_Unscented_Kalman_Filter_*.ipynb`:

```python
Q = np.diag([0.05**2, 0.05**2, 0.1**2, 0.03**2])
R = np.diag([0.8**2, np.deg2rad(3.0)**2])
```

### Process Noise $\mathbf{Q}$
* **$\sigma_\theta = 0.03\text{ rad} \approx 1.72^\circ$:**
  Increased heading uncertainty to account for aggressive transient maneuvers with rapid turn-rate reversals (S-curves and slalom).

### Measurement Noise $\mathbf{R}$
* **$\sigma_r = 0.8\text{ m}$:**
  Slightly higher range noise to simulate challenging environmental clutter.
* **$\sigma_\phi = 3.0^\circ = \frac{3\pi}{180} \approx 0.0524\text{ rad}$:**
  High bearing uncertainty intentionally selected to **stress-test trigonometric non-linearities** ($r \cos\phi, r \sin\phi$).
  * *Why this is important:* When angular noise is $3^\circ$, the first-order Taylor expansion used in the EKF underestimates the lateral curvature of the distribution, leading to degraded performance or divergence when initial errors are large ($45^\circ$).
  * The UKF leverages the Unscented Transform to propagate the exact 2nd- and 3rd-order moments through the nonlinear coordinate transformation without analytical linearization.

---

## 5. Summary Reference Table

| Notebook | Matrix | State / Sensor Elements | Values & Units | Physical Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **EKF** | $\mathbf{Q}$ | $[p_x, p_y, v, \theta]^T$ | $\operatorname{diag}(0.05^2\text{ m}^2, 0.05^2\text{ m}^2, 0.1^2(\text{m/s})^2, 0.02^2\text{ rad}^2)$ | Euler discretization error, tire slip, steering backlash |
| **EKF** | $\mathbf{R}$ | $[r, \phi]^T$ | $\operatorname{diag}(0.5^2\text{ m}^2, (1.0^\circ)^2)$ | Automotive radar range ToF & azimuth resolution |
| **ES-EKF** | $\mathbf{Q}_{\text{imu}}$ | $[p_x, p_y, v, \theta]^T \cdot \Delta t$ | $\operatorname{diag}(0.01^2, 0.01^2, 0.05^2, 0.01^2) \times 0.01$ | Accelerometer (VRW) & Gyro (ARW) spectral noise densities |
| **ES-EKF** | $\mathbf{R}_{\text{gps}}$ | $[p_x, p_y]^T$ | $\operatorname{diag}(1.5^2\text{ m}^2, 1.5^2\text{ m}^2)$ | Standalone civil GNSS horizontal positioning accuracy |
| **UKF** | $\mathbf{Q}$ | $[p_x, p_y, v, \theta]^T$ | $\operatorname{diag}(0.05^2, 0.05^2, 0.1^2, 0.03^2)$ | Dynamic agility & high yaw rate variance |
| **UKF** | $\mathbf{R}$ | $[r, \phi]^T$ | $\operatorname{diag}(0.8^2\text{ m}^2, (3.0^\circ)^2)$ | High angular noise designed to evaluate UT nonlinear handling |

---

## 6. How $\mathbf{Q}$ and $\mathbf{R}$ Govern Estimator Behavior

The ratio between $\mathbf{Q}$ and $\mathbf{R}$ sets the Kalman Gain $\mathbf{K}$:

$$\mathbf{K} \approx \frac{\mathbf{P}}{\mathbf{P} + \mathbf{R}}$$

* **Large $\mathbf{Q}$ / Small $\mathbf{R}$:**
  * Filter trusts incoming measurements heavily.
  * Fast response to dynamic changes, but state estimate becomes noisy.
* **Small $\mathbf{Q}$ / Large $\mathbf{R}$:**
  * Filter trusts its kinematic motion model heavily.
  * Smooth state trajectory, but slow response to aggressive vehicle maneuvers and potential risk of filter divergence if the model is inaccurate.
* **Statistical Consistency (NEES / NIS):**
  * When $\mathbf{Q}$ and $\mathbf{R}$ accurately reflect the physical noise, the **Normalized Estimation Error Squared (NEES)** will lie within the theoretical $95\%$ $\chi^2$ confidence interval.
