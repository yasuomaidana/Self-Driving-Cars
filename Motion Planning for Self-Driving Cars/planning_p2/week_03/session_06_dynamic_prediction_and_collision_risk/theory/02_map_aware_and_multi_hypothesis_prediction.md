# Map-Aware & Multi-Hypothesis Prediction

## 1. Map-Aware Path Projection

Physics-based models diverge rapidly when roads curve. Map-aware prediction constrains obstacle trajectories to follow underlying lanelet centerlines:

```
Step 1: Match vehicle (x_0, y_0, θ_0) to nearest lanelet centerline L_i.
Step 2: Project position into Frenet frame (s_0, d_0).
Step 3: Propagate longitudinal arc length along centerline: s(t) = s_0 + v * t.
Step 4: Transform (s(t), 0) back to Cartesian coordinates (x(t), y(t)).
```

---

## 2. Multi-Hypothesis Branching at Decision Points

At intersections or lane split points, a vehicle may take multiple discrete maneuvers:

$$\mathcal{H} = \{ (h_k, P(h_k)) \}_{k=1}^K$$

- **Hypothesis 1 ($P=0.65$):** Continue Straight along Main St.
- **Hypothesis 2 ($P=0.30$):** Turn Right onto 2nd Ave (indicated by turn signal and deceleration).
- **Hypothesis 3 ($P=0.05$):** Turn Left across traffic.

Motion planners must evaluate collision clearance against all high-probability hypotheses ($P(h_k) > \tau_{risk}$).
