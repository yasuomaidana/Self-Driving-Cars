---
title: "Hour 2: Inertial Sensors & Dead Reckoning"
tags:
  - hour-2
  - imu
  - inertial-navigation
  - accelerometers
  - gyroscopes
  - dead-reckoning
created: 2026-09-08
---

# Hour 2: Inertial Sensors & Dead Reckoning

> [!abstract] Key Learning Objectives
> 1. Understand the physical operating principles of MEMS accelerometers and gyroscopes.
> 2. Explain why a stationary accelerometer on a desk measures $+9.81\text{ m/s}^2$ upward.
> 3. Master the continuous and discrete strapdown inertial navigation equations.
> 4. Derive why accelerometer bias causes **quadratic position drift** ($\sim t^2$) and gyro bias causes **cubic position drift** ($\sim t^3$).

---

## 1. What is an Inertial Measurement Unit (IMU)?

An **Inertial Measurement Unit (IMU)** is the heartbeat of an autonomous vehicle's navigation system. Unlike GPS, cameras, or LiDAR—which depend on external satellites, light, or landmarks—an IMU is **completely self-contained**. It functions in underground parking garages, inside tunnels, and during blinding blizzards.

A 6-DOF (Degrees of Freedom) IMU contains two tri-axial sensor clusters mounted orthogonally:
1. **3-axis Accelerometer:** Measures specific force $\mathbf{f} = [f_x, f_y, f_z]^T$ along the vehicle's $x, y, z$ axes.
2. **3-axis Gyroscope:** Measures angular rotation rates $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^T$ around the vehicle's $x, y, z$ axes.

```
                  Z (Yaw / Down or Up)
                   ^
                   |    X (Roll / Forward)
                   |   /
                   |  /
                   | /
                   +--------------> Y (Pitch / Right)
```

In modern self-driving cars, **strapdown IMUs** are used: the sensors are rigidly bolted to the vehicle chassis without mechanical gimbals.

---

## 2. Physics of the Accelerometer: The Specific Force Concept

Most beginners believe an accelerometer measures the second derivative of position ($\ddot{\mathbf{r}}$). **This is incorrect!**

An accelerometer consists of a microscopic proof mass $m$ suspended by springs. It measures the physical compression of the springs caused by all **non-gravitational contact forces**:

$$\mathbf{a}_{meas} = \mathbf{f} = \frac{\mathbf{F}_{\text{non-gravitational}}}{m}$$

Einstein's **Principle of Equivalence** states that gravity affects every atom of the proof mass and sensor housing equally. Gravity does **not** compress the internal spring! 

The fundamental accelerometer equation in an inertial frame is:
$$\mathbf{f} = \ddot{\mathbf{r}} - \mathbf{g}$$

Where:
* $\mathbf{f}$ is the **specific force** measured by the accelerometer.
* $\ddot{\mathbf{r}}$ is the true kinematic coordinate acceleration.
* $\mathbf{g}$ is the gravitational acceleration vector pointing toward the center of the Earth ($|\mathbf{g}| \approx 9.81\text{ m/s}^2$).

### Mind-Bending Examples:
1. **A car parked still on a flat desk:**
   $$\ddot{\mathbf{r}} = \mathbf{0} \implies \mathbf{f} = \mathbf{0} - \mathbf{g} = -\mathbf{g}$$
   The table exerts an upward normal force on the sensor to prevent it from falling through the floor. Therefore, the accelerometer measures **$+9.81\text{ m/s}^2$ UPWARD**!
2. **A phone dropped in free fall:**
   $$\ddot{\mathbf{r}} = \mathbf{g} \implies \mathbf{f} = \mathbf{g} - \mathbf{g} = \mathbf{0}\text{ m/s}^2$$
   In free fall, an accelerometer measures zero acceleration!

> [!important] Gravity Compensation in Navigation Frame
> To find the true coordinate acceleration of the vehicle in the navigation frame, we must rotate the specific force to navigation coordinates and **add back the gravity vector**:
> $$\ddot{\mathbf{r}}_n = \mathbf{C}_{ns} \mathbf{f}_s + \mathbf{g}_n$$
> (In an ENU frame where $Z$ is Up, $\mathbf{g}_n = [0, 0, -9.81]^T$, so $\ddot{\mathbf{r}}_n = \mathbf{C}_{ns}\mathbf{f}_s + [0, 0, -9.81]^T$.)

---

## 3. Physics of the Rate Gyroscope & Sensor Error Models

Micro-Electro-Mechanical Systems (MEMS) gyroscopes use vibrating silicon structures that oscillate continuously. When the vehicle rotates, **Coriolis forces** deflect the vibrating mass, which is measured capacitively to deduce angular rate $\boldsymbol{\omega}$.

### Sensor Error Characteristics
Real-world IMUs are corrupted by three distinct noise phenomena:

$$\boldsymbol{\omega}_{meas}(t) = \boldsymbol{\omega}_{true}(t) + \mathbf{b}_{gyro}(t) + \mathbf{n}_{gyro}(t)$$
$$\mathbf{f}_{meas}(t) = \mathbf{f}_{true}(t) + \mathbf{b}_{accel}(t) + \mathbf{n}_{accel}(t)$$

1. **White Noise ($\mathbf{n}(t)$):** High-frequency thermo-mechanical vibration modeled as zero-mean Gaussian noise.
2. **Constant & Turn-on Bias ($\mathbf{b}$):** A persistent non-zero offset even when the sensor is completely motionless.
3. **Bias Drift (Random Walk):** Slowly evolving bias caused by internal temperature changes and mechanical stress.

---

## 4. The Strapdown Inertial Navigation Algorithm

Inertial dead reckoning is the process of integrating high-rate ($100\text{--}200\text{ Hz}$) IMU readings forward in time from step $k-1$ to $k$:

```mermaid
flowchart TD
    Gyro["Gyroscope: omega_k-1"] -->|Quaternion Integration| Att["1. Attitude Update: q_k = q_k-1 (x) q(omega*dt)"]
    Att -->|Rotation Matrix C_ns(q)| Transform["2. Rotate Acceleration: C_ns * f_k-1 + g"]
    Accel["Accelerometer: f_k-1"] --> Transform
    Transform -->|Single Integration| Vel["3. Velocity Update: v_k = v_k-1 + a_net * dt"]
    Vel -->|Double Integration| Pos["4. Position Update: p_k = p_k-1 + v*dt + 0.5*a*dt^2"]
```

### The Discrete Integration Equations:
1. **Attitude Propagation:**
   $$\mathbf{q}_k = \mathbf{q}_{k-1} \otimes \mathbf{q}(\boldsymbol{\omega}_{k-1}\Delta t)$$
2. **Net Acceleration in Navigation Frame:**
   $$\mathbf{a}_{net, k-1} = \mathbf{C}_{ns}(\mathbf{q}_{k-1}) \mathbf{f}_{k-1} + \mathbf{g}_n$$
3. **Velocity Propagation:**
   $$\mathbf{v}_k = \mathbf{v}_{k-1} + \Delta t \cdot \mathbf{a}_{net, k-1}$$
4. **Position Propagation:**
   $$\mathbf{p}_k = \mathbf{p}_{k-1} + \Delta t \cdot \mathbf{v}_{k-1} + \frac{\Delta t^2}{2} \cdot \mathbf{a}_{net, k-1}$$

---

## 5. The Mathematical Origin of Dead-Reckoning Drift

Why can't an autonomous car navigate using only an IMU?

### 5.1 Accelerometer Bias $\to$ Quadratic Position Drift ($t^2$)
Suppose the accelerometer has a tiny uncorrected bias of $b_a = 0.05\text{ m/s}^2$ (roughly $0.5\%$ of gravity):
$$\ddot{p}_{error} = b_a$$
Integrating once for velocity:
$$\Delta v(t) = \int_0^t b_a d\tau = b_a t$$
Integrating a second time for position:
$$\Delta p(t) = \int_0^t b_a \tau d\tau = \frac{1}{2} b_a t^2$$

* After $10\text{ seconds}$: $\Delta p = \frac{1}{2}(0.05)(10^2) = \mathbf{2.5\text{ meters}}$ (already out of the lane!).
* After $60\text{ seconds}$: $\Delta p = \frac{1}{2}(0.05)(60^2) = \mathbf{90\text{ meters}}$!

### 5.2 Gyroscope Bias $\to$ Cubic Position Drift ($t^3$)
Even worse, suppose the gyroscope has an uncorrected bias of $b_g = 0.1^\circ/\text{s} = 0.00175\text{ rad/s}$:
$$\delta\theta(t) = b_g t$$
Because the attitude estimate tilts by $\delta\theta$, the gravity vector $\mathbf{g}$ leaks into the horizontal accelerometer channels:
$$a_{leak} = g \sin(\delta\theta) \approx g \cdot b_g t$$
Integrating once for velocity:
$$\Delta v(t) = \int_0^t g b_g \tau d\tau = \frac{1}{2} g b_g t^2$$
Integrating again for position:
$$\Delta p(t) = \int_0^t \frac{1}{2} g b_g \tau^2 d\tau = \frac{1}{6} g b_g t^3$$

> [!important] The Inevitability of Divergence
> Gyro bias causes position errors to grow with the **cube of time ($t^3$)**!
> Within 60 seconds, a consumer-grade IMU will drift by hundreds of meters. **Standalone dead reckoning is physically incapable of long-term localization.** It requires an external reference sensor that does not drift—namely **GPS (GNSS)**!

---

## 6. Python Walkthrough: Simulating IMU Drift

```python
import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, 60, 600)  # 60 seconds
dt = 0.1

# Constant sensor biases typical of automotive MEMS
b_accel = 0.05       # 0.05 m/s^2 bias
b_gyro = np.deg2rad(0.1)  # 0.1 deg/s bias
g = 9.81

# Analytical drift formulas
accel_drift = 0.5 * b_accel * (time ** 2)
gyro_drift = (1.0 / 6.0) * g * b_gyro * (time ** 3)
total_drift = accel_drift + gyro_drift

plt.figure(figsize=(9, 5))
plt.plot(time, accel_drift, 'b--', label='Accel Bias Drift ($O(t^2)$)')
plt.plot(time, gyro_drift, 'r--', label='Gyro Gravity Leak Drift ($O(t^3)$)')
plt.plot(time, total_drift, 'k-', linewidth=2, label='Total Position Error')
plt.xlabel('Time (seconds)')
plt.ylabel('Position Error (meters)')
plt.title('Why Standalone IMUs Diverge: Quadratic & Cubic Drift')
plt.grid(True)
plt.legend()
plt.show()

print(f"Position Error after 10s: {total_drift[100]:.2f} meters")
print(f"Position Error after 60s: {total_drift[-1]:.2f} meters")
```

---

## 7. Self-Assessment & Checkpoint Questions

1. **Why does an accelerometer on the International Space Station measure $0\text{ m/s}^2$, even though Earth's gravity is strong there?**
   * *Answer:* Because the space station and proof mass are in continuous free fall (orbit) around the Earth. The gravitational acceleration is identical on both the housing and the proof mass, resulting in zero net spring compression.

2. **If an autonomous vehicle comes to a complete standstill at a red light, how can we use this to eliminate gyro and accelerometer bias?**
   * *Answer:* This is called a **Zero Velocity Update (ZUPT)**. Since we know the true velocity is exactly $\mathbf{v} = \mathbf{0}$ and acceleration is $\mathbf{0}$, any measured horizontal acceleration or angular velocity is direct sensor bias, allowing the Kalman filter to estimate and subtract the bias precisely.

3. **Why is gyro bias more dangerous to position estimation than accelerometer bias?**
   * *Answer:* Accelerometer bias causes quadratic drift ($\sim t^2$). Gyro bias causes an orientation tilt that mistakenly projects a fraction of Earth's massive gravity ($9.81\text{ m/s}^2$) into the horizontal plane, causing catastrophic **cubic drift ($\sim t^3$)**.
