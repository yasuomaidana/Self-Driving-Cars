# Session 05: Behaviour Planning & Hierarchical State Machines (HSMs)

**Week:** 3 | **Session:** 5  
**Time Allocation:** 3 Hours Total
- ⏱️ **Theory (2 Hours):** Tactical decision making, FSMs vs HSMs, Entry/Exit/Transition actions, Guard conditions, Intersection & Stop sign protocols
- 💻 **Practice (1 Hour):** Deterministic HSM Behaviour Planner implementation in Python with `Polars` and `Plotly`

---

## 🎯 Learning Objectives
By the end of this session, students will be able to:
1. Distinguish between tactical behaviour planning and continuous geometric trajectory generation.
2. Formulate Finite State Machines (FSMs) and Hierarchical State Machines (HSMs) to manage state explosion.
3. Define strict guard conditions for state transitions (e.g. stop line distance, time-to-stop, lead vehicle gap).
4. Implement rule-based handling for complex multi-lane scenarios, stop-sign 3-second wait timers, and lane changes.
5. Trace and log state transitions into `Polars` DataFrames and visualize state timelines with `Plotly`.

---

## 📚 Session Outline

### 1. Theory (2 Hours)
- [01. Behaviour Planning Fundamentals](theory/01_behaviour_planning_fundamentals.md): Role of behaviour layer, interface with mission and motion planners.
- [02. FSMs & Hierarchical State Machines](theory/02_finite_state_machines_and_hsm.md): Formal state machine definition, state explosion problem, sub-states and super-states.
- [03. Intersection & Lane Change Logic](theory/03_intersection_and_lane_changing_logic.md): Priority rules, stop signs, yield conditions, safe gaps.

### 2. Practice Lab (1 Hour)
- Lab Exercise: [`practice/exercise_hsm_behaviour_planner.py`](practice/exercise_hsm_behaviour_planner.py)
- Reference Solution: [`practice/solution_hsm_behaviour_planner.py`](practice/solution_hsm_behaviour_planner.py)
- Run the lab:
  ```bash
  source ../../.venv/bin/activate
  python practice/exercise_hsm_behaviour_planner.py
  ```
