# Heuristic Design, Admissibility & Consistency

## 1. Admissibility and Optimality Guarantees

For A* to guarantee finding the **globally optimal (cheapest)** path, the heuristic function $h(n)$ must be **admissible**:

$$0 \le h(n) \le h^*(n) \quad \forall n \in \mathcal{V}$$

Where $h^*(n)$ is the true optimal cost from node $n$ to the goal. An admissible heuristic *never overestimates* the true cost to reach the goal.

### Consequences of Inadmissibility
- If $h(n) > h^*(n)$ for some nodes, A* may prune optimal paths prematurely and return a sub-optimal route.

---

## 2. Consistency (Monotonicity)

A heuristic is **consistent (or monotonic)** if for every node $n$ and every neighbor $n'$ connected by an edge with cost $c(n, n')$:

$$h(n) \le c(n, n') + h(n')$$

and $h(goal) = 0$.

### Why Consistency Matters
- If $h(n)$ is consistent, the $f$-values along any path are monotonically non-decreasing.
- Crucially, when a node is expanded (popped from the priority queue), its $g(n)$ is already guaranteed to be optimal. No closed nodes ever need to be reopened!

---

## 3. Designing Domain-Specific Heuristics for Autonomous Driving

When edge weights represent **travel time** ($T = \frac{\text{dist}}{v}$):
- Let $v_{max\_network}$ be the highest speed limit across the entire city/highway network (e.g., $120 \text{ km/h} \approx 33.3 \text{ m/s}$).
- The admissible Euclidean Travel Time Heuristic is:

$$h(n, goal) = \frac{\sqrt{(x_n - x_{goal})^2 + (y_n - y_{goal})^2}}{v_{max\_network}}$$

Because no vehicle can ever travel faster than $v_{max\_network}$ in a straight line, this heuristic is strictly admissible and consistent!
