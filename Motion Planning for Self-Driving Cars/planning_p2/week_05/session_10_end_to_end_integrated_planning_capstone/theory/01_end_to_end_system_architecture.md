# End-to-End Autonomous Driving Planning Architecture

## 1. System Integration Pipeline

In production autonomous driving stacks (and in the CARLA simulator), data flows sequentially through perception, prediction, planning, and control:

```mermaid
flowchart LR
    Sensors["LiDAR / Radar / Cameras"] --> Localization["Localization & State Estimator\n[x, y, θ, v, a] @ 100Hz"]
    Sensors --> Perception["Perception & Tracking\n[Bounding Boxes] @ 20Hz"]
    Perception --> Prediction["Motion Prediction\n[Future Paths] @ 10Hz"]
    
    Mission["Mission Router (A*)\n[Global Lanelets]"] --> Behaviour["Behaviour Planner (HSM)\n[Target Lane & Speed] @ 10Hz"]
    Localization --> Behaviour
    Prediction --> Behaviour
    
    Behaviour --> LocalPlan["Local Planner\n[Optimized Path & Velocity] @ 10-20Hz"]
    Localization --> LocalPlan
    Prediction --> LocalPlan
    
    LocalPlan --> Controller["Stanley / MPC Controller\n[Steer, Throttle, Brake] @ 50-100Hz"]
    Localization --> Controller
    Controller --> Vehicle["Vehicle Actuators / CARLA Simulator"]
```

---

## 2. Module Interface Contracts

- **Mission $\to$ Behaviour:** Ordered list of lanelet IDs / reference path waypoints.
- **Behaviour $\to$ Local:**
  - `target_lane_id`: Centerline reference for local lattice generation.
  - `target_velocity`: Cruising or stop-approach speed target.
  - `mandatory_stop_point`: $[s_{stop}, x_{stop}, y_{stop}]$ or `None`.
- **Local $\to$ Control:**
  - Time-parameterized trajectory $\tau = [x(t), y(t), \theta(t), v(t), \kappa(t)]$ for lookahead horizon $T_{lookahead} \approx 2.0\sim 3.0s$.
