# Time-to-Collision (TTC) & Dynamic Safety Metrics

## 1. Longitudinal Time-to-Collision (TTC)

For an ego vehicle following a lead vehicle with relative distance $\Delta s = s_{lead} - s_{ego}$ and relative closing speed $v_{rel} = v_{ego} - v_{lead} > 0$:

$$\text{TTC} = \frac{\Delta s - d_{min}}{v_{rel}}$$

Where:
- $\text{TTC} < 1.5\text{ s}$: Critical impending collision zone (Emergency Automatic Braking required).
- $1.5\text{ s} \le \text{TTC} \le 3.0\text{ s}$: Warning / Tactical deceleration zone.
- $\text{TTC} > 3.0\text{ s}$: Safe headway.

---

## 2. Spatial-Temporal Cross-Trajectory Collision Checking

When ego trajectory $\tau_{ego}(t)$ and dynamic obstacle trajectory $\tau_{obs}(t)$ cross in 2D space:

1. At each time step $t_k \in [0, T]$:
   - Construct oriented bounding box polygon $\mathcal{P}_{ego}(t_k)$ of dimensions $(L_{ego}, W_{ego})$ rotated by $\theta_{ego}(t_k)$.
   - Construct oriented bounding box polygon $\mathcal{P}_{obs}(t_k)$ rotated by $\theta_{obs}(t_k)$.
   - Collision occurs if $\mathcal{P}_{ego}(t_k) \cap \mathcal{P}_{obs}(t_k) \neq \emptyset$.
2. The dynamic TTC is the first timestamp where polygons intersect:

$$\text{TTC} = \min \{ t_k \ge 0 \mid \mathcal{P}_{ego}(t_k) \cap \mathcal{P}_{obs}(t_k) \neq \emptyset \}$$
