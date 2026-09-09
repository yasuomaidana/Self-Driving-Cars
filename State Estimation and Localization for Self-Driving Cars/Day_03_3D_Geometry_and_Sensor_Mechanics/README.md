---
title: "Day 3: 3D Geometry and Sensor Mechanics (GNSS & INS)"
tags:
  - day-3
  - 3d-geometry
  - quaternions
  - imu
  - gnss
  - dead-reckoning
created: 2026-09-08
---

# Day 3: 3D Geometry and Sensor Mechanics (GNSS & INS)

Welcome to **Day 3**! Today we leave the 2D planar world and transition to full **3D spatial navigation**. 

In autonomous driving, vehicles pitch over hills, roll through banked curves, and yaw through intersections. To localize in 3D, we must master the mathematics of 3D rotations, learn why **Unit Quaternions** are superior to Euler angles, and deeply understand the physics of the two primary localization sensors: **Inertial Measurement Units (IMUs)** and **Global Navigation Satellite Systems (GNSS / GPS)**.

---

## 🧭 Day 3 Roadmap

```mermaid
flowchart LR
    H1["Hour 1: 3D Rotations & Quaternions"] --> H2["Hour 2: Inertial Sensors & Dead Reckoning"]
    H2 --> H3["Hour 3: GNSS & GPS Positioning"]
    H3 --> H4["Hour 4: Lab (Strapdown Kinematics & Drift)"]
```

---

## 📂 Module Index

1. **[[Hour_01_3D_Rotations_and_Quaternions/README|Hour 1: 3D Reference Frames, Rotations & Unit Quaternions]]**
   * Coordinate frames: ECI, ECEF, Navigation (NED/ENU), and Vehicle Body.
   * Rotation matrices $SO(3)$ and the Gimbal Lock singularity of Euler angles.
   * Unit Quaternions ($\mathbb{H}$): Algebra, compound rotations, and derivative kinematics.

2. **[[Hour_02_Inertial_Sensors_and_Dead_Reckoning/README|Hour 2: Inertial Measurement Units (IMU) & Dead Reckoning]]**
   * Accelerometers and the physics of specific force ($\mathbf{f} = \ddot{\mathbf{r}} - \mathbf{g}$).
   * Rate gyroscopes, sensor biases, and random walk.
   * Strapdown inertial navigation equations and quadratic error divergence.

3. **[[Hour_03_GNSS_and_GPS_Positioning/README|Hour 3: GNSS & GPS Satellite Positioning]]**
   * Satellite constellations, atomic clocks, and pseudorange observation models.
   * Trilateration using non-linear least squares to solve 3D position and clock bias.
   * Geometric Dilution of Precision (GDOP), multipath, and atmospheric delays.

4. **[[Hour_04_Lab_Strapdown_Kinematics_and_Drift/README|Hour 4: Applied Programming Lab (Strapdown Kinematics & Drift Analysis)]]**
   * Hands-on implementation of open-loop IMU dead reckoning using real roadway data.
   * Forward quaternion integration and gravity compensation.
   * Visualizing and measuring quadratic position drift to motivate sensor fusion.

---

## 🎯 Day 3 Learning Outcomes

By the end of this session, you should be able to:
- [ ] Convert orientations interchangeably between Euler angles, rotation matrices, and quaternions.
- [ ] Implement quaternion attitude integration using discrete angular velocity readings.
- [ ] Explain why a stationary accelerometer measures upward acceleration of $+9.81\text{ m/s}^2$.
- [ ] Formulate pseudorange equations and explain receiver clock bias.
- [ ] Measure the drift rate of standalone inertial dead reckoning on real vehicle datasets.
