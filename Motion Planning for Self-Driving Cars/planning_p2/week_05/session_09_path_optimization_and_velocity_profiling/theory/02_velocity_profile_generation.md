# Velocity Profile Generation

## 1. Velocity Profile Formulation

Once a geometric spatial path $[x(s), y(s)]$ is planned, the motion planner must assign an optimal longitudinal velocity $v(s)$ to every point along the path arc length $s \in [0, S]$.

### Governing Constraints
1. **Speed Limit Constraint:** $v(s) \le v_{limit}(s)$.
2. **Stopping Point Constraint:** $v(S_{stop}) = 0$.
3. **Lateral Acceleration (Cornering) Constraint:**
   $$a_{lat}(s) = v(s)^2 |\kappa(s)| \le a_{lat, max} \implies v(s) \le \sqrt{\frac{a_{lat, max}}{|\kappa(s)|}}$$
4. **Longitudinal Acceleration Limits:**
   $$-a_{decel, max} \le a_{lon}(s) = v(s) \frac{dv}{ds} \le a_{accel, max}$$

---

## 2. Forward and Backward Integration Algorithm

To generate a feasible velocity profile across discrete path points $s_0, s_1, \dots, s_N$:

```python
# Step 1: Initialize maximum velocity envelope from speed limits and curvature
v_max = [min(v_limit, sqrt(a_lat_max / max(1e-5, abs(kappa[i])))) for i in range(N)]
v_max[N-1] = 0.0 # Stop at destination

# Step 2: Backward pass (enforce braking deceleration limits)
v_prof = list(v_max)
for i in range(N-2, -1, -1):
    ds = s[i+1] - s[i]
    v_prof[i] = min(v_prof[i], sqrt(v_prof[i+1]**2 + 2 * a_decel_max * ds))

# Step 3: Forward pass (enforce acceleration limits)
v_prof[0] = v_initial
for i in range(0, N-1):
    ds = s[i+1] - s[i]
    v_prof[i+1] = min(v_prof[i+1], sqrt(v_prof[i]**2 + 2 * a_accel_max * ds))
```
