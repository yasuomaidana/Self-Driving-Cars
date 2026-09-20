---
title: "Hour 4: Applied Programming Lab (Ego-Vehicle Localization & Target Tracking)"
tags:
  - hour-4
  - lab
  - python
  - programming
  - kalman-filter
  - ego-localization
  - object-tracking
created: 2026-09-08
---

# Hour 4: Applied Programming Lab (Ego-Vehicle Localization & Target Tracking)

> [!abstract] Lab Objective
> 1. Formulate and implement the complete **Linear Kalman Filter** in Python from first principles, including the mathematical origins of $\mathbf{F}, \mathbf{G}, \mathbf{H}, \mathbf{Q}, \mathbf{R}$, and $\mathbf{P}$.
> 2. **Part 1 (Ego-Vehicle Localization with Known Control Inputs)**: Track our own vehicle using GPS position observations and known driver throttle/braking commands ($\check{\mathbf{x}}_k = \mathbf{F}\hat{\mathbf{x}}_{k-1} + \mathbf{G}\mathbf{u}_{k-1}$).
> 3. **Part 2 (Mountain Tunnel Outage & Dead Reckoning)**: Navigate through a 5-second GPS blackout in a tunnel using dead reckoning with known inputs $\mathbf{G}\mathbf{u}$, observing covariance $\mathbf{P}$ expansion and reacquisition.
> 4. **Part 3 (Target Object Tracking & Car-Following)**: Track an external lead vehicle for Adaptive Cruise Control (ACC) where control inputs are unknown ($\mathbf{u} = \mathbf{0}$, $\mathbf{G}$ omitted) and maneuvers are modeled via process noise $\mathbf{Q}$.
> 5. **Part 4 (Filter Tuning & Statistical Consistency)**: Master the trade-offs of $\mathbf{Q}$ vs. $\mathbf{R}$, NEES consistency analysis, and Datasheet Prior initialization.

---

## 1. Deep Mathematical Derivations of System Matrices ($\mathbf{F}, \mathbf{G}, \mathbf{H}, \mathbf{Q}, \mathbf{R}$)

The Discrete Linear Kalman Filter (LKF) operates on the continuous-time physical world sampled at discrete intervals $\Delta t$:

$$\mathbf{x}_k = \mathbf{F}_{k-1}\mathbf{x}_{k-1} + \mathbf{G}_{k-1}\mathbf{u}_{k-1} + \mathbf{w}_{k-1}, \qquad \mathbf{w}_{k-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_{k-1})$$
$$\mathbf{y}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k, \qquad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

---

### 1.1 Where Do $\mathbf{F}$ and $\mathbf{G}$ Come From? (Continuous-to-Discrete Kinematics)

In continuous time, vehicle motion is governed by continuous differential equations:
$$\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t)$$

For a 1D kinematic particle where $\mathbf{x}(t) = [p(t), v(t)]^T$ and control input is commanded acceleration $u(t) = a_{\text{cmd}}(t)$:
$$\dot{p}(t) = v(t), \qquad \dot{v}(t) = a_{\text{cmd}}(t)$$

In continuous state-space matrix form:
$$\begin{bmatrix} \dot{p}(t) \\ \dot{v}(t) \end{bmatrix} = \underbrace{\begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}}_{\mathbf{A}} \begin{bmatrix} p(t) \\ v(t) \end{bmatrix} + \underbrace{\begin{bmatrix} 0 \\ 1 \end{bmatrix}}_{\mathbf{B}} a_{\text{cmd}}(t)$$

To obtain the discrete-time state transition matrix $\mathbf{F}$ over sampling time $\Delta t$, we evaluate the **Matrix Exponential**:
$$\mathbf{F} = e^{\mathbf{A}\Delta t} = \mathbf{I} + \mathbf{A}\Delta t + \frac{\mathbf{A}^2 \Delta t^2}{2!} + \dots$$

Since $\mathbf{A}^2 = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$, higher-order terms vanish identically:
$$\mathbf{F} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} + \begin{bmatrix} 0 & \Delta t \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}$$

For the discrete control input matrix $\mathbf{G}$:
$$\mathbf{G} = \int_0^{\Delta t} e^{\mathbf{A}\tau} \mathbf{B} \, d\tau = \int_0^{\Delta t} \begin{bmatrix} 1 & \tau \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} d\tau = \int_0^{\Delta t} \begin{bmatrix} \tau \\ 1 \end{bmatrix} d\tau = \begin{bmatrix} \frac{\Delta t^2}{2} \\ \Delta t \end{bmatrix}$$

For 2D vehicle motion ($\mathbf{x} = [p_x, p_y, v_x, v_y]^T$ and $\mathbf{u} = [a_x, a_y]^T$):
$$\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \qquad \mathbf{G} = \begin{bmatrix} \frac{\Delta t^2}{2} & 0 \\ 0 & \frac{\Delta t^2}{2} \\ \Delta t & 0 \\ 0 & \Delta t \end{bmatrix}$$

---

### 1.2 Where Does Process Noise Covariance $\mathbf{Q}$ Come From? (Continuous White Noise Acceleration)

No dynamic model is exact. Vehicles experience unmodeled road slope variations, wind gusts, tire slippage, and engine torque ripple. We model these disturbances as a zero-mean continuous Gaussian white noise acceleration $w(t)$ with power spectral density $\sigma_a^2$:

$$\mathbb{E}[w(t)] = 0, \qquad \mathbb{E}[w(t)w(\tau)] = \sigma_a^2 \delta(t - \tau)$$

The continuous stochastic differential equation is:
$$\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{L} w(t), \qquad \mathbf{L} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

The discrete process noise covariance $\mathbf{Q}$ is obtained by integrating the noise through the system dynamics over time step $\Delta t$:
$$\mathbf{Q} = \int_0^{\Delta t} e^{\mathbf{A}(\Delta t - \tau)} \mathbf{L} \, \sigma_a^2 \, \mathbf{L}^T e^{\mathbf{A}^T(\Delta t - \tau)} \, d\tau$$

Let substitute variable $s = \Delta t - \tau$:
$$e^{\mathbf{A}s}\mathbf{L} = \begin{bmatrix} 1 & s \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} s \\ 1 \end{bmatrix}$$

$$\mathbf{Q} = \sigma_a^2 \int_0^{\Delta t} \begin{bmatrix} s \\ 1 \end{bmatrix} \begin{bmatrix} s & 1 \end{bmatrix} ds = \sigma_a^2 \int_0^{\Delta t} \begin{bmatrix} s^2 & s \\ s & 1 \end{bmatrix} ds$$

Evaluating each integral element:
$$\int_0^{\Delta t} s^2 ds = \frac{\Delta t^3}{3}, \qquad \int_0^{\Delta t} s \, ds = \frac{\Delta t^2}{2}, \qquad \int_0^{\Delta t} 1 \, ds = \Delta t$$

#### 1. Continuous White Noise Acceleration (CWNA) Form:
$$\mathbf{Q}_{\text{1D, CWNA}} = \begin{bmatrix} \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & \Delta t \end{bmatrix} \sigma_a^2$$

#### 2. Discrete Piecewise-Constant White Noise Acceleration (DWNA) Form:
If acceleration disturbance is assumed constant across the interval $\Delta t$ ($w_k \sim \mathcal{N}(0, \sigma_a^2)$), then $\mathbf{w}_k = \mathbf{G} w_k$:
$$\mathbf{Q}_{\text{1D, DWNA}} = \mathbf{G} \, \sigma_a^2 \, \mathbf{G}^T = \begin{bmatrix} \frac{\Delta t^2}{2} \\ \Delta t \end{bmatrix} \sigma_a^2 \begin{bmatrix} \frac{\Delta t^2}{2} & \Delta t \end{bmatrix} = \begin{bmatrix} \frac{\Delta t^4}{4} & \frac{\Delta t^3}{2} \\ \frac{\Delta t^3}{2} & \Delta t^2 \end{bmatrix} \sigma_a^2$$

For 2D tracking with independent $x$ and $y$ disturbances:
$$\mathbf{Q}_{\text{2D}} = \begin{bmatrix} 
\frac{\Delta t^4}{4}\sigma_a^2 & 0 & \frac{\Delta t^3}{2}\sigma_a^2 & 0 \\ 
0 & \frac{\Delta t^4}{4}\sigma_a^2 & 0 & \frac{\Delta t^3}{2}\sigma_a^2 \\ 
\frac{\Delta t^3}{2}\sigma_a^2 & 0 & \Delta t^2 \sigma_a^2 & 0 \\ 
0 & \frac{\Delta t^3}{2}\sigma_a^2 & 0 & \Delta t^2 \sigma_a^2 
\end{bmatrix}$$

---

### 1.3 Where Does Measurement Noise Covariance $\mathbf{R}$ Come From? (Sensor Datasheet Specs)

Sensor hardware (GNSS/GPS receiver, Radar, Camera, Wheel Speed Encoders) outputs noisy measurements:
$$\mathbf{y}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k, \qquad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

* **Measurement Matrix $\mathbf{H}$**: Extracts the observable components of the state vector. If a GPS measures only 2D position $(p_x, p_y)$ but cannot measure velocity:
  $$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

* **Sensor Noise Covariance $\mathbf{R}$**: Determined from manufacturer component datasheets or offline calibration:
  $$\mathbf{R} = \begin{bmatrix} \sigma_{x, \text{meas}}^2 & 0 \\ 0 & \sigma_{y, \text{meas}}^2 \end{bmatrix}$$
  - **U-blox ZED-F9P RTK GNSS**: $\sigma_{\text{pos}} \approx 0.02\,\text{m} \implies R = 0.0004\,\text{m}^2$.
  - **Standard Consumer Automotive GPS**: $\sigma_{\text{pos}} \approx 2.5\,\text{m} \implies R = 6.25\,\text{m}^2$.
  - **Continental ARS408 Automotive Radar**: Range std $\sigma_r \approx 0.4\,\text{m} \implies R = 0.16\,\text{m}^2$.

---

### 1.5 Target Object Tracking: Adaptive Cruise Control (ACC) & Autonomous Emergency Braking (AEB)

In ADAS and Autonomous Driving, **Adaptive Cruise Control (ACC)** and **Autonomous Emergency Braking (AEB)** regulate following distance to a lead car.

#### 1. Constant Time-Gap Policy:
$$d_{\text{safe}}(t) = d_0 + T_{\text{gap}} \cdot v_{\text{ego}}(t)$$
where $d_0 = 5.0\,\text{m}$ (standstill buffer) and $T_{\text{gap}} = 1.5\,\text{s}$ (time headway).

#### 2. Time-To-Collision (TTC) Metric:
$$\text{TTC}(t) = \begin{cases} \frac{d(t)}{-\Delta v(t)} & \text{if } \Delta v(t) < 0 \text{ (closing distance / hazard)} \\[4pt] +\infty & \text{if } \Delta v(t) \ge 0 \text{ (opening gap)} \end{cases}$$
When $\text{TTC} < 2.5\,\text{s}$, AEB initiates automated braking.

#### 3. Why Finite Differences Fail vs. The Kalman Observer:
If radar distance $d_{\text{meas}}$ has noise $\sigma_r = 0.5\,\text{m}$, direct numerical differentiation produces:
$$\sigma_{v, \text{diff}} = \frac{\sqrt{2}\sigma_r}{\Delta t} = \frac{\sqrt{2} \cdot 0.5}{0.1} \approx 7.07\,\text{m/s} \quad (\approx 25.5\,\text{km/h}!)$$
This massive noise would cause the vehicle brakes to chatter violently. The Kalman Filter eliminates $>90\%$ of this noise by estimating the smooth latent closing velocity $\Delta v$.

#### 4. Why $\mathbf{G}$ is Omitted in Target Tracking:
* We cannot access the other driver's throttle/brake commands ($\mathbf{u} = \mathbf{0}$).
* Therefore, $\mathbf{G}\mathbf{u}$ is omitted from prediction ($\check{\mathbf{x}} = \mathbf{F}\hat{\mathbf{x}}$).
* Target maneuvers are modeled as stochastic process noise $\mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$.

---

### 1.6 The 5 Discrete Kalman Filter Equations

$$\begin{array}{lll}
\hline
\textbf{Phase} & \textbf{Operation} & \textbf{Mathematical Equation} \\
\hline
\text{1. State Prediction} & \text{Propagate State Mean} & \check{\mathbf{x}}_k = \mathbf{F}_{k-1}\hat{\mathbf{x}}_{k-1} + \mathbf{G}_{k-1}\mathbf{u}_{k-1} \\[4pt]
\text{2. Covariance Prediction} & \text{Propagate Uncertainty} & \check{\mathbf{P}}_k = \mathbf{F}_{k-1}\hat{\mathbf{P}}_{k-1}\mathbf{F}_{k-1}^T + \mathbf{Q}_{k-1} \\[4pt]
\text{3. Kalman Gain} & \text{Optimal Weighting Matrix} & \mathbf{K}_k = \check{\mathbf{P}}_k \mathbf{H}_k^T \left( \mathbf{H}_k \check{\mathbf{P}}_k \mathbf{H}_k^T + \mathbf{R}_k \right)^{-1} \\[4pt]
\text{4. State Correction} & \text{Incorporate Measurement} & \hat{\mathbf{x}}_k = \check{\mathbf{x}}_k + \mathbf{K}_k (\mathbf{y}_k - \mathbf{H}_k \check{\mathbf{x}}_k) \\[4pt]
\text{5. Covariance Correction} & \text{Joseph Stabilized Form} & \hat{\mathbf{P}}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \check{\mathbf{P}}_k (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T \\[4pt]
\hline
\end{array}$$

---

## 2. Python Implementation

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class LinearKalmanFilter:
    """Discrete Linear Kalman Filter supporting Control Matrix G and Joseph Covariance Form."""
    
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 G: np.ndarray = None, x0: np.ndarray = None, P0: np.ndarray = None):
        self.F = np.asarray(F, dtype=np.float64)
        self.H = np.asarray(H, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        
        self.n = self.F.shape[0]
        self.m = self.H.shape[0]
        self.G = np.zeros((self.n, 1)) if G is None else np.asarray(G, dtype=np.float64)
        
        self.x = np.zeros((self.n, 1)) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(self.n, 1)
        self.P = np.eye(self.n) * 100.0 if P0 is None else np.asarray(P0, dtype=np.float64).reshape(self.n, self.n)

    def predict(self, u: np.ndarray = None):
        """Prediction: x_check = F * x_hat + G * u,  P_check = F * P_hat * F^T + Q"""
        if u is not None:
            u_vec = np.asarray(u, dtype=np.float64).reshape(-1, 1)
            self.x = self.F @ self.x + self.G @ u_vec
        else:
            self.x = self.F @ self.x
            
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, y: np.ndarray):
        """Correction (Joseph Form): x_hat = x_check + K*nu,  P_hat = (I-KH)P(I-KH)^T + KRK^T"""
        y_vec = np.asarray(y, dtype=np.float64).reshape(self.m, 1)
        nu = y_vec - self.H @ self.x
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ nu
        I_KH = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy(), self.P.copy()
```

---

## 3. Interactive Jupyter Notebooks

Two complete Jupyter Notebooks are available in `position_class/`:

1. **[Instructor Complete Solution Guide](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/position_class/hour_04_linear_kalman_filter_lab.ipynb)** ([HTML View](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/position_class/hour_04_linear_kalman_filter_lab.html))
   - Full mathematical derivations, 1D Ego throttle/braking tracking, 2D Mountain Tunnel dead reckoning with $\mathbf{G}\mathbf{u}$, ACC Car-Following, and NEES consistency verification.
2. **[Student Scaffolded Exercise Handout](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/position_class/hour_04_linear_kalman_filter_lab_student.ipynb)** ([HTML View](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/position_class/hour_04_linear_kalman_filter_lab_student.html))
   - Lab handout with `# TODO` scaffolding for students to construct matrices and complete the filtering pipeline.

---

## 4. Key Takeaways & Discussion

1. **Why use $\mathbf{G}\mathbf{u}$ for Ego-Vehicle?**
   - Knowing our own throttle and braking eliminates estimation lag during dynamic pedal tip-ins and hard braking.
2. **What happens during a Tunnel Blackout?**
   - The Kalman Filter skips the correction step and executes **dead reckoning** with known control commands $\mathbf{G}\mathbf{u}$.
   - The error covariance $\mathbf{P}_k = \mathbf{F}\mathbf{P}_{k-1}\mathbf{F}^T + \mathbf{Q}$ expands quadratically, signaling growing position uncertainty to the motion planner.
3. **Why is $\mathbf{G}$ omitted in Object Tracking?**
   - We cannot read the intent or control inputs of other vehicles ($\mathbf{u} = \mathbf{0}$). Unknown accelerations are modeled via process noise $\mathbf{Q}$.
4. **Why use the Joseph Form?**
   - It guarantees symmetry and strict positive definiteness ($P_{ii} > 0$), preventing negative variance cancellation errors and $\text{NaN}$ crashes in production code.
