# Session 06: Dynamic Environments & Object Motion Prediction

**Week:** 3 | **Session:** 6  
**Time Allocation:** 3 Hours Total
- ⏱️ **Theory (2 Hours):** Motion prediction paradigms (Physics, Maneuver, Interaction-aware), Map-aware path forecasting, Time-to-Collision (TTC) & Dynamic risk metrics
- 💻 **Practice (1 Hour):** Multi-Hypothesis Motion Prediction & Spatial-Temporal TTC Evaluator with `Shapely`, `Polars`, and `Plotly`

---

## 🎯 Learning Objectives
By the end of this session, students will be able to:
1. Compare physics-based (CV, CA, CTRV) vs maneuver-based (lane follow, lane change) prediction models.
2. Formulate map-aware dynamic object trajectory prediction along lanelet reference paths.
3. Compute exact mathematical Time-to-Collision (TTC) for intersecting trajectories.
4. Model vehicle bounding boxes and perform spatial-temporal intersection tests using `Shapely`.
5. Profile obstacle trajectories and collision metrics using `Polars` and visualize collision zones with `Plotly`.

---

## 📚 Session Outline

### 1. Theory (2 Hours)
- [01. Motion Prediction Paradigms](theory/01_motion_prediction_paradigms.md): Physics-based models (CV, CA, CTRV), Maneuver-based models, and Interaction-aware models.
- [02. Map-Aware & Multi-Hypothesis Prediction](theory/02_map_aware_and_multi_hypothesis_prediction.md): Snapping obstacles to lanelets, branching future hypotheses with probabilities.
- [03. Time-to-Collision (TTC) & Safety Metrics](theory/03_time_to_collision_and_safety_metrics.md): Formal TTC calculation, Time Headway (TH), Post-Encroachment Time (PET).

### 2. Practice Lab (1 Hour)
- Lab Exercise: [`practice/exercise_dynamic_prediction_ttc.py`](practice/exercise_dynamic_prediction_ttc.py)
- Reference Solution: [`practice/solution_dynamic_prediction_ttc.py`](practice/solution_dynamic_prediction_ttc.py)
- Run the lab:
  ```bash
  source ../../.venv/bin/activate
  python practice/exercise_dynamic_prediction_ttc.py
  ```
