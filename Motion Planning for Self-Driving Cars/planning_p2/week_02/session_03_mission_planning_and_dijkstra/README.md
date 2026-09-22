# Session 03: Mission Planning & Dijkstra's Shortest Path Algorithm

**Week:** 2 | **Session:** 3  
**Time Allocation:** 3 Hours Total
- ⏱️ **Theory (2 Hours):** Road network graphs, node/edge formulations, topological routing, Dijkstra's algorithm & heap priority queues
- 💻 **Practice (1 Hour):** Road Network Graph Construction & Dijkstra Solver with `NetworkX`, `Polars`, and `Plotly`

---

## 🎯 Learning Objectives
By the end of this session, students will be able to:
1. Model complex urban road networks as directed, weighted graphs with intersection nodes and lane segment edges.
2. Formulate realistic edge cost functions incorporating physical length, lane speed limits, and intersection turning costs.
3. Understand and prove the optimality and completeness of Dijkstra's algorithm.
4. Implement Dijkstra's algorithm using binary min-heaps (`heapq`) and parent pointers for path reconstruction.
5. Visualize the road network topology and shortest path frontiers with `Plotly`.

---

## 📚 Session Outline

### 1. Theory (2 Hours)
- [01. Road Network Graphs](theory/01_road_network_graphs.md): Graph representation, node coordinate embeddings, edge attributes.
- [02. Dijkstra's Shortest Path](theory/02_dijkstra_shortest_path.md): Algorithmic steps, relaxation, proof sketch, complexity analysis.
- [03. Priority Queues & Graph Structures](theory/03_priority_queues_and_graph_data_structures.md): Min-heap mechanics, decrease-key operations, memory footprints.

### 2. Practice Lab (1 Hour)
- Lab Exercise: [`practice/exercise_dijkstra.py`](practice/exercise_dijkstra.py)
- Reference Solution: [`practice/solution_dijkstra.py`](practice/solution_dijkstra.py)
- Run the lab:
  ```bash
  source ../../.venv/bin/activate
  python practice/exercise_dijkstra.py
  ```
