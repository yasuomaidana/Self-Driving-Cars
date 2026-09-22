# Intersection & Lane Change Decision Logic

## 1. Stop-Sign Intersection Negotiation Protocol

At an all-way stop sign intersection, the autonomous vehicle must follow deterministic safety rules:

```
1. Approach & Decelerate:
   - Identify stop line s_stop.
   - Smoothly decelerate: a_req = - v^2 / (2 * (s_stop - s_curr)).
   
2. Full Stop & Timer:
   - Transition to STAY_STOPPED when |v| < 0.1 m/s and dist_to_stop_line < 1.0 m.
   - Start 3.0-second countdown timer.

3. Yield / Right-of-Way Evaluation:
   - Check dynamic objects: Detect vehicles that arrived at earlier timestamps.
   - Verify intersection conflict zones are clear of oncoming traffic and pedestrians.

4. Proceed & Cross:
   - Accelerate smoothly through intersection along reference turn centerline.
   - Transition back to LANE_FOLLOW once vehicle rear axle clears intersection exit.
```

---

## 2. Lane Change Safety Envelopes

A lane change requires verifying lead and lag safety gaps in the target lane:

$$\text{Lead Gap}: \quad s_{lead\_target} - s_{ego} > d_{safe\_lead}(v_{ego})$$

$$\text{Lag Gap}: \quad s_{ego} - s_{lag\_target} > d_{safe\_lag}(v_{lag})$$

Where $d_{safe}(v) = d_{min} + t_{headway} \cdot v + \frac{(v_{rel})^2}{2 a_{decel}}$.
