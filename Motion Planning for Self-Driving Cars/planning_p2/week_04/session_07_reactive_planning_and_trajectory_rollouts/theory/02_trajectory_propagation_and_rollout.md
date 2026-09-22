# Trajectory Rollout Algorithm

## 1. Candidate Rollout Generation

To generate $K$ candidate trajectories:
1. Discretize the lateral steering angle $\delta \in [-\delta_{max}, \delta_{max}]$ into $K$ samples (e.g. $K = 15$ or $21$).
2. For each sampled $\delta_k$, numerically integrate the kinematic bicycle model for a lookahead time $T_{horizon}$ (e.g. $T = 3.0$ seconds):

```python
for delta_k in linspace(-delta_max, delta_max, K):
    trajectory_k = []
    state = x_current
    for t in arange(0, T, dt):
        state = kinematic_bicycle_step(state, delta_k, a_target, dt)
        trajectory_k.append(state)
```

---

## 2. Cost Evaluation & Selection

Each rollout trajectory $\tau_k$ is scored against a composite objective:

$$J(\tau_k) = w_{goal} \cdot d_{goal}(\tau_k) + w_{obs} \cdot \operatorname{Cost}_{obs}(\tau_k) + w_{lane} \cdot d_{centerline}(\tau_k)$$

The collision-free trajectory with the lowest cost $J(\tau_k)$ is passed to the vehicle controller.
