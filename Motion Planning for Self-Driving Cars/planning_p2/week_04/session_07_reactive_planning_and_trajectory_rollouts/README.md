# Session 07: Reactive Planning & Trajectory Rollouts

**Week:** 4 | **Session:** 7  
**Time Allocation:** 3 Hours Total
- ⏱️ **Theory (2 Hours):** Dynamic Window Approach (DWA), Trajectory rollouts across action spaces, Multi-circle conservative collision checking, Swaths
- 💻 **Practice (1 Hour):** Trajectory Rollout Generator & Collision Evaluator with `Polars` and `Plotly`

---

## 🎯 Learning Objectives
By the end of this session, students will be able to:
1. Understand sampling-based reactive trajectory generation and the Dynamic Window Approach (DWA).
2. Generate a bundle of candidate trajectory rollouts by forward integrating the kinematic bicycle model across steering angles $\delta_i$.
3. Compute the vehicle's swept volume (swath) using conservative overlapping circle approximations.
4. Filter out collision trajectories and score valid candidates based on progress, clearance, and comfort.
5. Log rollout candidates into `Polars` DataFrames and visualize the rollout bundle with `Plotly`.

---

## 📚 Session Outline

### 1. Theory (2 Hours)
- [01. Reactive Planning & DWA](theory/01_reactive_planning_and_dwa.md): Velocity space sampling, dynamic windows, reactive obstacle avoidance.
- [02. Trajectory Propagation & Rollouts](theory/02_trajectory_propagation_and_rollout.md): Forward integration, horizon selection, lateral action space discretization.
- [03. Swaths & Circle Collision Checking](theory/03_swath_generation_and_circle_collision_checking.md): Vehicle bounding box approximations with 3-4 bounding circles, distance-to-obstacle checks.

### 2. Practice Lab (1 Hour)
- Lab Exercise: [`practice/exercise_trajectory_rollout.py`](practice/exercise_trajectory_rollout.py)
- Reference Solution: [`practice/solution_trajectory_rollout.py`](practice/solution_trajectory_rollout.py)
- Run the lab:
  ```bash
  source ../../.venv/bin/activate
  python practice/exercise_trajectory_rollout.py
  ```
