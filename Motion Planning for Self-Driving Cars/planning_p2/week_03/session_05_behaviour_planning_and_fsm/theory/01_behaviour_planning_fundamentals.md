# Behaviour Planning Fundamentals

## 1. Role of the Behaviour Layer in Autonomous Driving

The **Behaviour Planner (Decision-Making Layer)** bridges the high-level topological mission route (e.g. "Take Main Street to 5th Ave") and the low-level geometric trajectory optimizer:

```
[Mission Route: Lanelets A -> B -> C]
                 |
                 v
      [Behaviour Planner (HSM)]
     - Decides: Maintain Lane, Prepare Lane Change, Yield, Stop, Pass
     - Sets target speed, target lane, stopping points, and boundary constraints
                 |
                 v
   [Local Trajectory Planner & Controller]
```

### Key Responsibilities
1. **Scenario Identification:** Recognizing current driving context (highway cruising, multi-lane road, 4-way stop-sign intersection, traffic light).
2. **Rule & Protocol Enforcement:** Right-of-way yielding, stop line stopping, speed limit adherence.
3. **Maneuver Output:** Providing unambiguous constraints (target speed $v_{target}$, target lane centerline, mandatory stop horizon $s_{stop}$) to the local trajectory generator.
