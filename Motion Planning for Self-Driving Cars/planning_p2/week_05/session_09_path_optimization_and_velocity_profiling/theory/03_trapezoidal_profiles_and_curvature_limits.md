# Trapezoidal Profiles & Dynamic Limits

## 1. Classical Trapezoidal Velocity Profile

For straight line segments, a velocity profile accelerating at $+a_{max}$, cruising at $v_{cruise}$, and decelerating at $-a_{max}$ forms a trapezoid in $(t, v)$ or parabolic profile in $(s, v^2)$:

```mermaid
flowchart LR
    Start["<b>Start / Accelerate</b><br/>v = 0 ──▶ v_cruise<br/>Acceleration: +a_max"] --> Cruise["<b>Constant Cruise Phase</b><br/>v = v_cruise<br/>Acceleration: a = 0"]
    Cruise --> Decel["<b>Comfort Braking Phase</b><br/>v_cruise ──▶ 0<br/>Deceleration: -a_decel_max"]
    Decel --> Stop["<b>Target Goal Stop</b><br/>s = S_goal, v = 0"]
    
    classDef acc fill:#e8f5e9,stroke:#388e3c,stroke-width:1.5px;
    classDef cru fill:#e3f2fd,stroke:#1976d2,stroke-width:1.5px;
    classDef dec fill:#ffebee,stroke:#d32f2f,stroke-width:1.5px;
    classDef stp fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    class Start acc;
    class Cruise cru;
    class Decel dec;
    class Stop stp;
```

### Critical Braking Distance
To come to a complete stop from speed $v_0$ with maximum comfort deceleration $a_{brake} \approx 2.5 m/s^2$:

$$d_{brake} = \frac{v_0^2}{2 a_{brake}}$$

If remaining distance to stop sign $d_{remaining} < d_{brake}$, the vehicle cannot stop smoothly without emergency friction braking.
