# Session 04: Informed Search & A* Route Planning

**Week:** 2 | **Session:** 4  
**Time Allocation:** 3 Hours Total
- ⏱️ **Theory (2 Hours):** Informed search principles, Heuristic admissibility & consistency, Maneuver penalties (left/right turns, lane changes), Bidirectional A* Search, IDA* (Iterative Deepening A*) & Algorithm Trade-offs
- 💻 **Practice (1 Hour):** A*, Bidirectional A* & IDA* Benchmark with `Polars`, `Plotly`, and 3-Way Synchronized Video

---

## 🎯 Learning Objectives
By the end of this session, students will be able to:
1. Explain the evaluation function $f(n) = g(n) + h(n)$ of the A* search algorithm.
2. Formulate and prove mathematical conditions for heuristic **admissibility** ($h(n) \le h^*(n)$) and **consistency** (triangle inequality).
3. Design realistic domain-specific heuristics for autonomous driving (Euclidean / Haversine distance divided by max network speed).
4. Implement and analyze **Bidirectional A* Search**, dual wavefront convergence, safe stopping conditions, and path reconstruction across directed road networks.
5. Implement **Iterative Deepening A* (IDA*)** and analyze space-time trade-offs ($\mathcal{O}(d)$ memory vs node re-expansions).
6. Compare modern search paradigms used in self-driving stacks: Dijkstra, Forward A*, Bidirectional A*, IDA*, $D^*$ Lite, $ARA^*$, and Hybrid A*.
7. Benchmark search performance (expanded nodes, search space %, runtime, memory) using `Polars` and visualize search frontiers with `Plotly`.

---

## 📚 Session Outline

### 1. Theory (2 Hours)
- [01. A* Search Algorithm](theory/01_astar_search_algorithm.md): Informed graph search, priority queue sorting by $f(n)$, optimality guarantees.
- [02. Heuristic Design & Admissibility](theory/02_heuristic_design_and_admissibility.md): Admissible vs inadmissible heuristics, consistent heuristics, Manhattan vs Euclidean vs Dubins distance heuristics.
- [03. Lane-Level Routing & Turn Penalties](theory/03_lane_level_routing_and_turn_penalties.md): Modeling complex intersections, U-turns, and lane change penalties.
- [04. Bidirectional A* Search](theory/04_bidirectional_astar_search.md): Dual search frontiers, directed edge traversal, meeting node connection, stopping criteria, and search space complexity reduction.
- [05. IDA* & Search Algorithm Trade-Offs](theory/05_ida_star_and_search_tradeoffs.md): Memory-constrained DFS with $f$-thresholds, trade-off matrix, and advanced motion planning algorithms ($D^*$ Lite, $ARA^*$, Hybrid A*, Contraction Hierarchies).

### 2. Practice Lab (1 Hour)
- Lab Exercise: [`practice/exercise_astar.py`](practice/exercise_astar.py) & [`practice/exercise_astar.ipynb`](practice/exercise_astar.ipynb)
- Reference Solution: [`practice/solution_astar.py`](practice/solution_astar.py) & [`practice/solution_astar.ipynb`](practice/solution_astar.ipynb)
- Synchronized Evolution Video: [`practice/generate_astar_search_video.py`](practice/generate_astar_search_video.py)
- Run the lab:
  ```bash
  uv run python practice/exercise_astar.py
  ```


