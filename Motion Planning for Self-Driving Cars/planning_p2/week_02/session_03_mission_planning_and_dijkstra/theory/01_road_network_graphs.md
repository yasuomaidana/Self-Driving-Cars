# Road Network Graph Representation

## 1. Topological Graph Formulation for Mission Routing

Mission planning computes the sequence of road segments connecting origin and destination coordinates across large spatial maps. The road network is formalized as a directed, weighted graph:

$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{W})$$

Where:
- $\mathcal{V} = \{v_1, v_2, \dots, v_n\}$ is the set of **Nodes (Vertices)**: Intersections, decision points, lane merges, lane splits, or endpoints of road segments. Each node has geographic or local coordinates $(x_i, y_i)$.
- $\mathcal{E} = \{e_{ij} = (v_i, v_j)\}$ is the set of **Directed Edges**: Drivable lane segments moving from node $v_i$ to $v_j$. One-way streets have edges in only one direction; two-way streets have dual opposing directed edges.
- $\mathcal{W}: \mathcal{E} \to \mathbb{R}^+$ is the **Edge Weight / Cost Function**.

---

## 2. Edge Weight Formulations

In autonomous driving, the cost $w(e_{ij})$ is rarely simple Euclidean distance. Instead, it balances:

1. **Traversal Travel Time:** $T_{travel} = \frac{L_{ij}}{v_{limit, ij}}$
2. **Road Class Weighting:** Multiplier favoring highways and arterial roads over narrow residential streets.
3. **Maneuver / Intersection Costs:** Penalties for making left turns across oncoming traffic vs easy right turns.
4. **Lane Count Factor:** Multiple lanes offer higher reliability and overtaking capacity.

$$w(e_{ij}) = \alpha \cdot \frac{L_{ij}}{v_{limit, ij}} + \beta \cdot C_{turn}(e_{pre}, e_{ij}) + \gamma \cdot C_{traffic}$$
