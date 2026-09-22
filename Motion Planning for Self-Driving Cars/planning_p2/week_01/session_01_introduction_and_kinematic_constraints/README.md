# Session 01: The Planning Problem, Hierarchy & Vehicle Constraints

**Week:** 1 | **Session:** 1  
**Time Allocation:** 3 Hours Total
- ⏱️ **Theory (2 Hours):** Planning decomposition, kinematic & dynamic constraints, objective functions
- 💻 **Practice (1 Hour):** Discrete-time Kinematic Bicycle Model & Constraint Analysis with `Polars` and `Plotly`

---

## 🎯 Learning Objectives
By the end of this session, students will be able to:
1. Articulate the 4-layer hierarchical architecture of autonomous driving systems (Mission, Behaviour, Local, Control).
2. Mathematically formulate the kinematic bicycle model and non-holonomic constraints of Ackermann-steered vehicles.
3. Quantify physical vehicle limits including maximum steer angle, maximum curvature, and lateral acceleration friction limits (Kamm's circle).
4. Construct a multi-objective cost function balancing collision risk, passenger comfort, travel time, and traffic rule adherence.
5. Simulate vehicle trajectories using `Polars` for fast state telemetry and visualize state evolution with `Plotly`.

---

## 📚 Session Outline

### 1. Theory (2 Hours)
- [01. Motion Planning Hierarchy](theory/01_motion_planning_hierarchy.md): Mission Planner vs Behavior Planner vs Local Motion Planner.
- [02. Vehicle Kinematics & Constraints](theory/02_vehicle_kinematics_and_constraints.md): Front-wheel steered kinematic bicycle model, Ackermann steering, turning radius, and friction ellipse limits.
- [03. Objective Functions for Autonomous Driving](theory/03_objective_functions.md): Mathematical definition of safety, comfort (jerk, lateral acceleration), progress, and rule compliance costs.

### 2. Practice Lab (1 Hour)
- Lab Exercise: [`practice/exercise_kinematic_bicycle.py`](practice/exercise_kinematic_bicycle.py)
- Reference Solution: [`practice/solution_kinematic_bicycle.py`](practice/solution_kinematic_bicycle.py)
- Run the lab:
  ```bash
  source ../../.venv/bin/activate
  python practice/exercise_kinematic_bicycle.py
  ```
