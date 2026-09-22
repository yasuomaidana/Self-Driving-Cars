# Reactive Planning & Dynamic Window Approach (DWA)

## 1. Principle of Reactive Motion Planning

Reactive planners sample a set of feasible control commands or target offsets, propagate trajectories forward over a short horizon (e.g. 2–5 seconds), evaluate each candidate against a multi-criteria cost function, and select the optimal collision-free path.

---

## 2. Dynamic Window Approach (DWA)

The Dynamic Window Approach restricts the sampled velocity search space $V_s$ to only those velocities that can be safely achieved within the next time step $\Delta t$ given acceleration limits:

$$V_d = \{ (v, \omega) \mid v \in [v - a_{min}\Delta t, v + a_{max}\Delta t], \quad \omega \in [\omega - \dot{\omega}_{max}\Delta t, \omega + \dot{\omega}_{max}\Delta t] \}$$

### Admissible Velocities ($V_a$)
A pair $(v, \omega)$ is admissible if the vehicle can come to a complete stop before hitting the nearest obstacle:
$$V_a = \{ (v, \omega) \mid v \le \sqrt{2 \cdot \operatorname{dist}(v, \omega) \cdot a_{brake}} \}$$

The final search space is $V_r = V_s \cap V_d \cap V_a$.
