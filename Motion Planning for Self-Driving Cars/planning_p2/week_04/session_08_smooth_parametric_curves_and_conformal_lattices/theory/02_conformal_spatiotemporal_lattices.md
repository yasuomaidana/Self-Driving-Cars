# Conformal Spatiotemporal Lattices

## 1. Conformal Lattice Concept

Rather than generating arbitrary random paths in global Cartesian space $(x, y)$, a **Conformal Lattice** samples trajectories aligned with the road geometry:

```mermaid
flowchart LR
    Start["Vehicle State<br/>(s₀, d₀)"]
    
    subgraph LatticeGoals["Conformal Sampling Horizons (s_f, d_f)"]
        L_Left["Left Lane Offset<br/>d_f = +3.5m"]
        L_NudgeLeft["Left Nudge<br/>d_f = +1.75m"]
        L_Center["Lane Centerline<br/>d_f = 0.0m"]
        L_NudgeRight["Right Nudge<br/>d_f = -1.75m"]
        L_Right["Right Lane Offset<br/>d_f = -3.5m"]
    end
    
    Start -->|Cubic Spiral / Polynomial| L_Left
    Start -->|Cubic Spiral / Polynomial| L_NudgeLeft
    Start -->|Cubic Spiral / Polynomial| L_Center
    Start -->|Cubic Spiral / Polynomial| L_NudgeRight
    Start -->|Cubic Spiral / Polynomial| L_Right
    
    classDef center fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef side fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px;
    class L_Center center;
    class L_Left,L_NudgeLeft,L_NudgeRight,L_Right side;
```

A discrete grid of target boundary states $(s_f, d_f)$ is placed at fixed lookahead distances:
- Longitudinal horizons: $s_f \in \{20m, 30m, 40m, 50m\}$
- Lateral offsets: $d_f \in \{-3.5m, -1.75m, 0.0m, +1.75m, +3.5m\}$

For each $(s_f, d_f)$, an optimal quintic polynomial or cubic spiral curve is generated, forming a structured bundle of lane-conforming paths.
