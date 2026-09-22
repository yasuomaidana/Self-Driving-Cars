# Dijkstra's Shortest Path Algorithm

## 1. Algorithmic Overview

Dijkstra's algorithm finds the shortest paths from a single source node $s \in \mathcal{V}$ to all other nodes (or a specified goal node $g$) in a graph with non-negative edge weights ($w(u, v) \ge 0$).

### Core Principle
Maintain for each node $v \in \mathcal{V}$:
- $g(v)$: The current shortest known cost from start $s$ to $v$. Initialized to $\infty$, with $g(s) = 0$.
- $\pi(v)$: The predecessor (parent) pointer of $v$ along the shortest path.
- A **Priority Queue (Min-Heap)** $\mathcal{Q}$ of frontier nodes ordered by $g(v)$.
- A **Visited Set (Closed Set)** $\mathcal{S}$ of nodes whose optimal distance is finalized.

---

## 2. Pseudocode & Step-by-Step Execution

```python
def dijkstra(Graph G, Node start, Node goal):
    g_score = {node: infinity for node in G.nodes}
    g_score[start] = 0
    parent = {node: None for node in G.nodes}
    
    # Priority Queue of (cost, node)
    frontier = PriorityQueue()
    frontier.push((0, start))
    visited = set()
    
    while not frontier.is_empty():
        current_cost, u = frontier.pop_min()
        
        if u in visited:
            continue
        visited.add(u)
        
        if u == goal:
            return reconstruct_path(parent, goal)
            
        for neighbor, edge_weight in G.neighbors(u):
            tentative_g = g_score[u] + edge_weight
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                parent[neighbor] = u
                frontier.push((tentative_g, neighbor))
                
    return None # No path found
```

### Time & Space Complexity
- With a Binary Min-Heap: $\mathcal{O}((|\mathcal{E}| + |\mathcal{V}|) \log |\mathcal{V}|)$
- Space Complexity: $\mathcal{O}(|\mathcal{V}|)$ to store scores and parent pointers.
