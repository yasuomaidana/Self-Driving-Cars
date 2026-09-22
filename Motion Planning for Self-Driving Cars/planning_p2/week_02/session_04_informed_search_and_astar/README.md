# Session 04: Informed Search & A* Route Planning

**Week:** 2 | **Session:** 4  
**Time Allocation:** 3 Hours Total
- ⏱️ **Theory (2 Hours):** Informed search principles, Heuristic admissibility & consistency, Maneuver penalties (left/right turns, lane changes)
- 💻 **Practice (1 Hour):** A* Road Network Router vs. Dijkstra Benchmark with `Polars` and `Plotly`

---

## 🎯 Learning Objectives
By the end of this session, students will be able to:
1. Explain the evaluation function $f(n) = g(n) + h(n)$ of the A* search algorithm.
2. Formulate and prove mathematical conditions for heuristic **admissibility** ($h(n) \le h^*(n)$) and **consistency** (triangle inequality).
3. Design realistic domain-specific heuristics for autonomous driving (Euclidean distance divided by max network speed).
4. Integrate turn penalties (unprotected left turns vs easy right turns) into edge costs.
5. Benchmark search performance (expanded nodes and runtime) between Dijkstra and A* using `Polars` and visualize search frontiers with `Plotly`.

---

## 📚 Session Outline

### 1. Theory (2 Hours)
- [01. A* Search Algorithm](theory/01_astar_search_algorithm.md): Informed graph search, priority queue sorting by $f(n)$, optimality guarantees.
- [02. Heuristic Design & Admissibility](theory/02_heuristic_design_and_admissibility.md): Admissible vs inadmissible heuristics, consistent heuristics, Manhattan vs Euclidean vs Dubins distance heuristics.
- [03. Lane-Level Routing & Turn Penalties](theory/03_lane_level_routing_and_turn_penalties.md): Modeling complex intersections, U-turns, and lane change penalties.

### 2. Practice Lab (1 Hour)
- Lab Exercise: [`practice/exercise_astar.py`](practice/exercise_astar.py)
- Reference Solution: [`practice/solution_astar.py`](practice/solution_astar.py)
- Run the lab:
  ```bash
  source ../../.venv/bin/activate
  python practice/exercise_astar.py
  ```
