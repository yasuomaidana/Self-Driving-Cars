---
title: "Hour 1: 3D Reference Frames, Rotations & Unit Quaternions"
tags:
  - hour-1
  - 3d-geometry
  - reference-frames
  - rotation-matrices
  - quaternions
  - so3
created: 2026-09-08
---

# Hour 1: 3D Reference Frames, Rotations & Unit Quaternions

> [!abstract] Key Learning Objectives
> 1. Define standard coordinate reference frames: ECI, ECEF, Navigation (NED/ENU), and Vehicle Body.
> 2. Understand Direction Cosine Matrices ($\mathbf{C} \in SO(3)$) and why Euler angles suffer from Gimbal Lock.
> 3. Master the algebra and kinematics of Unit Quaternions ($\mathbf{q} \in \mathbb{H}$).
> 4. Implement discrete quaternion attitude updates in Python using angular velocity inputs.

---

## 1. Coordinate Reference Frames in Autonomous Driving

Before we can estimate where a vehicle is located, we must ask: **relative to what?**

In autonomous driving, four primary coordinate frames are used:

```mermaid
flowchart TD
    ECI["Earth-Centered Inertial (ECI)
    (Non-rotating, celestial reference)"] -->|Earth Rotation| ECEF["Earth-Centered Earth-Fixed (ECEF)
    (Rotates with Earth, WGS-84)"]
    ECEF -->|Local Tangent Plane| Nav["Navigation Frame (NED / ENU)
    (Origin on roadway, aligns with gravity)"]
    Nav -->|Vehicle Pose [p, q]| Body["Vehicle Body Frame
    (Origin at IMU / rear axle)"]
```

1. **Earth-Centered Inertial Frame (ECI - $i$):**
   * Origin at Earth's center of mass.
   * Non-rotating with respect to distant stars. Newton's laws of motion hold strictly here.
2. **Earth-Centered Earth-Fixed Frame (ECEF - $e$):**
   * Origin at Earth's center of mass.
   * Rotates with the Earth ($1\text{ rev / }24\text{ hours}$). Raw GPS coordinates (Latitude, Longitude, Altitude) are defined here using the WGS-84 ellipsoid.
3. **Navigation Frame (Local Level - $n$):**
   * Attached to a local point on the Earth's surface.
   * Two standard conventions:
     * **NED:** X = North, Y = East, Z = Down (toward Earth center).
     * **ENU:** X = East, Y = North, Z = Up (away from Earth center).
   * **Gravity is aligned with the vertical axis**, which makes local kinematics easy to compute!
4. **Body Frame (Vehicle / Sensor - $b$ or $s$):**
   * Fixed to the vehicle (typically centered at the IMU or center of gravity).
   * Standard convention (in NED navigation): X = Forward, Y = Right, Z = Down.

---

## 2. 3D Rotation Representations

Representing a 3D orientation requires describing how the axes of the Body frame ($b$) are oriented relative to the Navigation frame ($n$).

### 2.1 The Direction Cosine Matrix (Rotation Matrix $\mathbf{C}_{nb} \in SO(3)$)
A Rotation Matrix transforms a vector from body coordinates to navigation coordinates:
$$\mathbf{v}_n = \mathbf{C}_{nb} \mathbf{v}_b$$

It belongs to the **Special Orthogonal Group** $SO(3)$ and satisfies:
1. **Orthogonality:** $\mathbf{C}^T \mathbf{C} = \mathbf{C} \mathbf{C}^T = \mathbf{I}_{3\times3} \implies \mathbf{C}^{-1} = \mathbf{C}^T$.
2. **Determinant:** $\det(\mathbf{C}) = +1$ (preserves length and handedness; no reflections).
* **Limitation:** It uses 9 numbers to represent 3 degrees of freedom (overparameterized with 6 constraints: 3 column norms $=1$, 3 mutual dot products $=0$).

### 2.2 Euler Angles (Roll $\phi$, Pitch $\theta$, Yaw $\psi$)
Any 3D rotation can be broken into three successive single-axis rotations:
1. Rotate about Z by **Yaw $\psi$** (Heading: North $\to$ East)
2. Rotate about Y by **Pitch $\theta$** (Nose up/down)
3. Rotate about X by **Roll $\phi$** (Tilt left/right)

$$\mathbf{C}_{nb} = \mathbf{R}_z(\psi) \mathbf{R}_y(\theta) \mathbf{R}_x(\phi)$$

#### The Fatal Flaw: Gimbal Lock
When the vehicle pitches straight up or down by $\theta = \pm 90^\circ$, the roll axis and yaw axis become collinear! 
The mathematical kinematic relationship:
$$\begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix} = \begin{bmatrix} 1 & \sin\phi\tan\theta & \cos\phi\tan\theta \\ 0 & \cos\phi & -\sin\phi \\ 0 & \sin\phi\sec\theta & \cos\phi\sec\theta \end{bmatrix} \begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}$$
Notice the terms $\tan\theta$ and $\sec\theta = \frac{1}{\cos\theta}$. When $\theta \to \pm 90^\circ$, $\cos(90^\circ) = 0$, causing **division by zero and infinite angular velocity**!

> [!warning] Euler Angle Singularity
> Euler angles **must never be used** as internal state representations in a 3D Kalman Filter because of Gimbal Lock singularities.

---

## 3. Unit Quaternions ($\mathbb{H}$): The Industry Standard

Invented by Sir William Rowan Hamilton in 1843, **Unit Quaternions** provide a singularity-free, computationally efficient 4-parameter representation of 3D rotations based on **Euler's Rotation Theorem**: *any 3D rotation is equivalent to a single rotation by angle $\theta$ around a unit axis $\mathbf{u} = [u_x, u_y, u_z]^T$*.

### 3.1 Quaternion Definition
A unit quaternion $\mathbf{q} \in \mathbb{H}$ is defined as:
$$\mathbf{q} = \begin{bmatrix} q_w \\ \mathbf{q}_v \end{bmatrix} = \begin{bmatrix} q_w \\ q_x \\ q_y \\ q_z \end{bmatrix} = \begin{bmatrix} \cos\left(\frac{\theta}{2}\right) \\ \mathbf{u} \sin\left(\frac{\theta}{2}\right) \end{bmatrix}$$

With the unit-norm constraint:
$$\|\mathbf{q}\|^2 = q_w^2 + q_x^2 + q_y^2 + q_z^2 = 1$$

* **Identity Rotation:** $\mathbf{q}_{identity} = [1, 0, 0, 0]^T$ ($\theta = 0$).
* **Conjugate / Inverse:** $\mathbf{q}^* = [q_w, -q_x, -q_y, -q_z]^T$. For unit quaternions, $\mathbf{q}^{-1} = \mathbf{q}^*$.
* **Double Cover:** Both $+\mathbf{q}$ and $-\mathbf{q}$ represent the exact same physical 3D orientation (rotating by $\theta$ around $\mathbf{u}$ is identical to rotating by $360^\circ - \theta$ around $-\mathbf{u}$).

### 3.2 Quaternion Product (Composition of Rotations)
Rotating by $\mathbf{q}_1$ followed by $\mathbf{q}_2$ is given by the quaternion product $\mathbf{q}_{net} = \mathbf{q}_2 \otimes \mathbf{q}_1$:

$$\mathbf{p} \otimes \mathbf{q} = \begin{bmatrix} p_w q_w - \mathbf{p}_v \cdot \mathbf{q}_v \\ p_w \mathbf{q}_v + q_w \mathbf{p}_v + \mathbf{p}_v \times \mathbf{q}_v \end{bmatrix}$$

In matrix multiplication form:
$$\mathbf{p} \otimes \mathbf{q} = \boldsymbol{\Omega}_L(\mathbf{p}) \mathbf{q} = \begin{bmatrix} p_w & -p_x & -p_y & -p_z \\ p_x & p_w & -p_z & p_y \\ p_y & p_z & p_w & -p_x \\ p_z & -p_y & p_x & p_w \end{bmatrix} \begin{bmatrix} q_w \\ q_x \\ q_y \\ q_z \end{bmatrix}$$

### 3.3 Converting a Quaternion to a Rotation Matrix
To rotate a 3D vector $\mathbf{v}$ using a quaternion without matrix multiplication, or to obtain $\mathbf{C}_{ns}(\mathbf{q})$:

$$\mathbf{C}(\mathbf{q}) = (q_w^2 - \|\mathbf{q}_v\|^2)\mathbf{I}_{3\times3} + 2 \mathbf{q}_v \mathbf{q}_v^T + 2 q_w [\mathbf{q}_v]_\times$$

Explicitly:
$$\mathbf{C}(\mathbf{q}) = \begin{bmatrix} 1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\ 2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\ 2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2) \end{bmatrix}$$

---

## 4. Quaternion Kinematics: Integrating Angular Velocity

Suppose our vehicle's gyroscope measures angular rates $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^T$ in sensor frame coordinates.
The time derivative of the orientation quaternion is:

$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega} \end{bmatrix} = \frac{1}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}) \mathbf{q}$$

Where:
$$\boldsymbol{\Omega}(\boldsymbol{\omega}) = \begin{bmatrix} 0 & -\omega_x & -\omega_y & -\omega_z \\ \omega_x & 0 & \omega_z & -\omega_y \\ \omega_y & -\omega_z & 0 & \omega_x \\ \omega_z & \omega_y & -\omega_x & 0 \end{bmatrix}$$

### Discrete Integration over $\Delta t$
Assuming angular velocity $\boldsymbol{\omega}_{k-1}$ is constant over the short time interval $\Delta t$:
1. Compute the incremental rotation vector: $\boldsymbol{\theta} = \boldsymbol{\omega}_{k-1} \Delta t$.
2. Compute the incremental quaternion:
   $$\mathbf{q}(\boldsymbol{\theta}) = \begin{bmatrix} \cos\left(\frac{\|\boldsymbol{\theta}\|}{2}\right) \\ \frac{\boldsymbol{\theta}}{\|\boldsymbol{\theta}\|} \sin\left(\frac{\|\boldsymbol{\theta}\|}{2}\right) \end{bmatrix}$$
3. Update the state quaternion by right-multiplication:
   $$\mathbf{q}_k = \mathbf{q}_{k-1} \otimes \mathbf{q}(\boldsymbol{\theta})$$
4. **Normalize:** In floating point arithmetic, always re-normalize to preserve unit length:
   $$\mathbf{q}_k \leftarrow \frac{\mathbf{q}_k}{\|\mathbf{q}_k\|}$$

---

## 5. Python Walkthrough: Using `rotations.py`

In your workspace folder `Vehicle State Estimation on a Roadway/`, open and explore [`rotations.py`](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State%20Estimation%20and%20Localization%20for%20Self-Driving%20Cars/Vehicle%20State%20Estimation%20on%20a%20Roadway/rotations.py). It provides a full Quaternion class:

```python
import numpy as np
import sys
# Add rotations module path
sys.path.append('../../Vehicle State Estimation on a Roadway')
from rotations import Quaternion, skew_symmetric

# 1. Create a Quaternion from Euler Angles (e.g. Roll=0.1, Pitch=0.2, Yaw=1.5 rad)
q_initial = Quaternion(euler=[0.1, 0.2, 1.5])
print("Initial Quaternion (w, x, y, z):")
print(np.round(q_initial.to_numpy(), 4))

# 2. Convert to Direction Cosine Matrix C_ns
C_mat = q_initial.to_mat()
print("\n3x3 Rotation Matrix C:")
print(np.round(C_mat, 4))
# Verify orthogonality: C * C^T = I
assert np.allclose(C_mat @ C_mat.T, np.eye(3)), "Matrix must be orthogonal!"

# 3. Simulate gyro measurement: omega = [0.01, 0.0, 0.5] rad/s over dt = 0.1 s
omega = np.array([0.01, 0.0, 0.5])
dt = 0.1

# 4. Integrate attitude
delta_q = Quaternion(axis_angle=omega * dt)
q_updated = q_initial.quat_mult_right(delta_q)

print("\nUpdated Quaternion after 0.1s turn:")
print(np.round(q_updated.to_numpy(), 4))
print(f"Updated Yaw: {q_updated.to_euler()[2]:.4f} rad")
```

---

## 6. Self-Assessment & Checkpoint Questions

1. **Why does multiplying a quaternion by $-1$ ($-\mathbf{q}$) produce the exact same rotation matrix?**
   * *Answer:* Look at the formula for $\mathbf{C}(\mathbf{q})$: every term involves products of two quaternion entries (e.g., $q_w^2$, $q_x q_y$, $q_w q_z$). If every component changes sign, $(-q_a)(-q_b) = +q_a q_b$, leaving the resulting rotation matrix $\mathbf{C}$ completely unchanged.

2. **Why do we right-multiply $\mathbf{q}_k = \mathbf{q}_{k-1} \otimes \mathbf{q}(\boldsymbol{\omega} \Delta t)$ rather than left-multiply?**
   * *Answer:* Because the gyro angular rates $\boldsymbol{\omega}$ are measured in the **sensor body frame** (moving frame). In Hamilton's quaternion convention, incremental rotations expressed in the local body frame multiply from the right, whereas rotations expressed in the global navigation frame multiply from the left.

3. **Why do we still normalize $\mathbf{q} \leftarrow \frac{\mathbf{q}}{\|\mathbf{q}\|}$ after multiplying if unit quaternions theoretically stay unit length?**
   * *Answer:* Standard computer floating-point truncation introduces small roundoff errors on the order of $10^{-16}$. Over millions of sensor cycles, these tiny errors accumulate, causing $\|\mathbf{q}\|$ to drift away from $1.0$, which distorts vector lengths during coordinate rotations.
