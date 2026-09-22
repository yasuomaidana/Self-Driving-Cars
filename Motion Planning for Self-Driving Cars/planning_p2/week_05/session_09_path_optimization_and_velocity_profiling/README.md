# Session 09: Numerical Path Optimization & Velocity Profiling

**Week:** 5 | **Session:** 9  
**Time Allocation:** 3 Hours Total
- ⏱️ **Theory (2 Hours):** Non-linear path optimization, Bending energy minimization, Trapezoidal & S-curve velocity profiles, Curvature speed limits
- 💻 **Practice (1 Hour):** SciPy Path Optimizer & Curvature-Constrained Velocity Profiler with `Polars` and `Plotly`

---

## 🎯 Learning Objectives
By the end of this session, students will be able to:
1. Formulate path smoothing as a constrained non-linear optimization problem (BVP) using `scipy.optimize`.
2. Construct objective functions penalizing total curvature (bending energy) and deviations from reference waypoints.
3. Compute path curvature $\kappa(s)$ along arbitrary parametric curves.
4. Calculate maximum allowable cornering speeds based on lateral acceleration limits $v_{max}(s) = \sqrt{a_{lat}^{max} / |\kappa(s)|}$.
5. Construct backward and forward passes for Trapezoidal acceleration-limited velocity profiles.

---

## 📚 Session Outline

### 1. Theory (2 Hours)
- [01. Nonlinear Path Optimization](theory/01_nonlinear_path_optimization.md): Optimization formulations, boundary conditions, objective terms.
- [02. Velocity Profile Generation](theory/02_velocity_profile_generation.md): Longitudinal profile dynamics, forward/backward integration passes.
- [03. Trapezoidal Profiles & Curvature Limits](theory/03_trapezoidal_profiles_and_curvature_limits.md): Handling sharp turns, braking distances, deceleration limits.

### 2. Practice Lab (1 Hour)
- Lab Exercise: [`practice/exercise_path_optimizer_velocity.py`](practice/exercise_path_optimizer_velocity.py)
- Reference Solution: [`practice/solution_path_optimizer_velocity.py`](practice/solution_path_optimizer_velocity.py)
- Run the lab:
  ```bash
  source ../../.venv/bin/activate
  python practice/exercise_path_optimizer_velocity.py
  ```
