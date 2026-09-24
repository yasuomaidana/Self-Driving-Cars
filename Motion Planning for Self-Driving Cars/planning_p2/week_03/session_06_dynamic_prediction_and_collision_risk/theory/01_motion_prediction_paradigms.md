# Dynamic Motion Prediction Paradigms

## 1. Classification of Motion Prediction Models

To safely plan motion around other traffic participants (vehicles, cyclists, pedestrians), autonomous vehicles forecast future trajectory distributions $\mathcal{P}(\mathbf{x}_{1:T}^{obs} \mid \mathbf{x}_{0}^{obs}, \text{Map})$:

```mermaid
flowchart TD
    subgraph PredModels["Motion Prediction Paradigms"]
        direction LR
        P1["<b>Physics-Based</b><br/><i>Short horizon (< 1-2s)</i><br/>• Constant Velocity (CV)<br/>• Constant Acceleration (CA)<br/>• Constant Turn Rate (CTRV)"]
        P2["<b>Maneuver-Based</b><br/><i>Medium horizon (2-5s)</i><br/>• Lane Following<br/>• Lane Change Left/Right<br/>• Intersection Turning"]
        P3["<b>Interaction-Aware</b><br/><i>Long horizon (> 5s)</i><br/>• Game Theory Models<br/>• Multi-Agent Attention<br/>• Yielding & Courtesies"]
    end
    
    classDef p1 fill:#e1f5fe,stroke:#0288d1,stroke-width:1.5px;
    classDef p2 fill:#fff3e0,stroke:#f57c00,stroke-width:1.5px;
    classDef p3 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1.5px;
    class P1 p1;
    class P2 p2;
    class P3 p3;
```

---

## 2. Physics-Based State Propagations

### Constant Velocity (CV) Model
Assuming velocity $v$ and heading $\theta$ remain constant over short horizons $\Delta t$:
$$x(t) = x_0 + v \cos(\theta) t, \quad y(t) = y_0 + v \sin(\theta) t$$

### Constant Turn Rate and Velocity (CTRV) Model
For yaw rate $\omega = \dot{\theta} \neq 0$:
$$x(t) = x_0 + \frac{v}{\omega} \left( \sin(\theta_0 + \omega t) - \sin(\theta_0) \right)$$
$$y(t) = y_0 + \frac{v}{\omega} \left( -\cos(\theta_0 + \omega t) + \cos(\theta_0) \right)$$
$$\theta(t) = \theta_0 + \omega t$$
