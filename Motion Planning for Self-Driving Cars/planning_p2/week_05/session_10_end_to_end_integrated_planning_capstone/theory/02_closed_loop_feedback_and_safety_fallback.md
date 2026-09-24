# Closed-Loop Feedback & Safety Fallback Layers

## 1. Real-Time Closed-Loop Replanning

At each planning tick (e.g. every $\Delta t_{plan} = 100\text{ms}$):
1. Receive updated vehicle ego pose from localization.
2. Determine current position along the active trajectory and find nearest reference path waypoint.
3. Propagate forward from current dynamic state $[x_0, y_0, \theta_0, v_0]$ to ensure smooth continuity (preventing control step jumps).
4. Re-evaluate static and dynamic obstacles and regenerate/optimize the local trajectory.

---

## 2. Safety Verification & Fallback Hierarchy

If the local optimizer fails to find a valid collision-free trajectory or if an unexpected obstacle cuts into the vehicle's safety envelope:

```mermaid
flowchart TD
    L1["<b>Level 1: Nominal Local Plan</b><br/>Smooth Conformal Lattice / Optimized Spline Path"]
    L2["<b>Level 2: In-Lane Comfort Deceleration</b><br/>Follow lead vehicle / Decelerate smoothly at -2.0 m/s²"]
    L3["<b>Level 3: Emergency Automatic Braking (AEB)</b><br/>Max deceleration (-6.0 to -8.0 m/s²) to full stop"]
    L4["<b>Level 4: Minimum Risk Maneuver (MRM)</b><br/>Pull over to drivable shoulder & activate hazard flashers"]
    
    L1 -->|Failure: Infeasible path or dynamic intrusion| L2
    L2 -->|Failure: Critical collision risk TTC < 1.2s| L3
    L3 -->|Failure: Actuator fault / critical sensor loss| L4
    
    classDef l1 fill:#e8f5e9,stroke:#388e3c,stroke-width:1.5px;
    classDef l2 fill:#fff8e1,stroke:#fbc02d,stroke-width:1.5px;
    classDef l3 fill:#ffebee,stroke:#d32f2f,stroke-width:1.5px;
    classDef l4 fill:#424242,stroke:#212121,color:#fff,stroke-width:2px;
    class L1 l1;
    class L2 l2;
    class L3 l3;
    class L4 l4;
```
