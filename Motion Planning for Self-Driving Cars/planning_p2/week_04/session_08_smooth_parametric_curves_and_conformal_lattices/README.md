# Session 08: Parametric Curves & Conformal Spatiotemporal Lattices

**Week:** 4 | **Session:** 8  
**Time Allocation:** 3 Hours Total
- ⏱️ **Theory (2 Hours):** Continuous parametric curves, Cubic Spirals, Polynomial splines, Frenet-Serret ($s, d$) coordinates, Conformal lattice decomposition
- 💻 **Practice (1 Hour):** Frenet Frame Transformations & Conformal Lattice Generator with `Polars` and `Plotly`

---

## 🎯 Learning Objectives
By the end of this session, students will be able to:
1. Explain why circular arcs produce discontinuous curvature and why continuous-curvature curves (Clothoids / Cubic Spirals) are necessary for steering comfort.
2. Formulate Frenet-Serret coordinate transformations: Global $(x, y, \theta)$ to Road-Aligned $(s, d, \theta_{rel})$ and vice-versa.
3. Construct a Conformal Spatiotemporal Lattice by discretizing longitudinal horizons $s_f$ and lateral offsets $d_f$.
4. Generate smooth Quintic Polynomial paths ($d(s)$) satisfying boundary initial and final conditions $[d_0, \dot{d}_0, \ddot{d}_0]$ and $[d_f, \dot{d}_f, \ddot{d}_f]$.
5. Visualize the resulting conformal lattice curves over road lanes with `Plotly`.

---

## 📚 Session Outline

### 1. Theory (2 Hours)
- [01. Parametric Curves & Cubic Spirals](theory/01_parametric_curves_cubic_spirals.md): Curvature continuity $C^2$, Clothoids, Cubic Spirals $\kappa(s)$, Quintic Polynomials.
- [02. Conformal Spatiotemporal Lattices](theory/02_conformal_spatiotemporal_lattices.md): Lattice sampling grids, goal sets, road alignment.
- [03. Frenet-Serret Coordinate Frame](theory/03_frenet_frame_transformations.md): Mathematical derivations for curvilinear projections and Cartesian conversions.

### 2. Practice Lab (1 Hour)
- Lab Exercise: [`practice/exercise_spiral_lattice.py`](practice/exercise_spiral_lattice.py)
- Reference Solution: [`practice/solution_spiral_lattice.py`](practice/solution_spiral_lattice.py)
- Run the lab:
  ```bash
  source ../../.venv/bin/activate
  python practice/exercise_spiral_lattice.py
  ```
