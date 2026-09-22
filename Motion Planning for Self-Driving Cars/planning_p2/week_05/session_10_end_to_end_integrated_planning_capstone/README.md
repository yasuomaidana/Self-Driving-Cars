# Session 10: End-to-End Autonomous Driving Planning Capstone

**Week:** 5 | **Session:** 10  
**Time Allocation:** 3 Hours Total
- ⏱️ **Theory (2 Hours):** Full-stack system integration (Mission $\to$ Behaviour $\to$ Local $\to$ Control), Closed-loop replanning, Safety fallback layers, and CARLA simulation pipeline
- 💻 **Practice (1 Hour):** Integrated Autonomous Motion Planning Capstone Pipeline & Closed-Loop Benchmark with `Polars` and `Plotly`

---

## 🎯 Learning Objectives
By the end of this session, students will be able to:
1. Integrate all subsystems developed across weeks 1–5 into a unified real-time autonomous driving pipeline.
2. Formulate closed-loop replanning loops (10–20 Hz) responding dynamically to new perception detections.
3. Design fail-safe emergency fallback mechanisms (Emergency Stop, Minimum Risk Maneuver).
4. Run an integrated scenario simulation with static obstacles, dynamic vehicles, and stop signs.
5. Export full-trip telemetry into a high-performance `Polars` DataFrame and generate a comprehensive interactive HTML dashboard with `Plotly`.

---

## 📚 Session Outline

### 1. Theory (2 Hours)
- [01. End-to-End System Architecture](theory/01_end_to_end_system_architecture.md): Interface contracts between modules, data synchronization, thread management.
- [02. Closed-Loop Feedback & Fallbacks](theory/02_closed_loop_feedback_and_safety_fallback.md): Replanning cycles, horizon stability, fail-safe minimum risk maneuvers.
- [03. CARLA Simulation Integration](theory/03_carla_simulation_integration.md): CARLA client-server API, sensor streams, map ingestion, and benchmark metrics.

### 2. Practice Lab (1 Hour)
- Lab Capstone: [`practice/integrated_planner_capstone.py`](practice/integrated_planner_capstone.py)
- Reference Solution: [`practice/solution_planner_capstone.py`](practice/solution_planner_capstone.py)
- Run the capstone:
  ```bash
  source ../../.venv/bin/activate
  python practice/integrated_planner_capstone.py
  ```
