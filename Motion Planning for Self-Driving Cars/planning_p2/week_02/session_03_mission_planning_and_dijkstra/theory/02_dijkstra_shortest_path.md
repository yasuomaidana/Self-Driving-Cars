# Dijkstra's Shortest Path Algorithm

## 1. Algorithmic Overview

Dijkstra's algorithm finds the shortest paths from a single source node $s \in \mathcal{V}$ to all other reachable nodes (or terminates early upon reaching a specific target node $t$) in a graph with non-negative edge weights ($w(u, v) \ge 0$).

### Core Principle
Maintain for each node $v \in \mathcal{V}$:
- $g(v)$: The exact accumulated cost of the shortest path discovered so far from start $s$ to $v$. Initialized to $\infty$, with $g(s) = 0$.
- $\text{prev}[v]$: The predecessor (parent) pointer of $v$ along the shortest path.
- A **Min-Priority Queue** `open_set` storing active frontier nodes ordered by $g(v)$.
- A **Visited Set (Closed Set)** tracking nodes whose shortest distance from $s$ has been finalized.

---

## 2. Pseudocode: Point-to-Point Dijkstra (With Early Stopping)

In mission planning and point-to-point navigation, we stop exploration the instant the destination node is extracted from the priority queue:

```python
def dijkstra_early_stop(graph, source, target):
    # 1. Initialize data structures
    open_set = MinPriorityQueue()          # Priority queue ordered by g_score (lowest cost first)
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[source] = 0.0
    prev = {node: None for node in graph.nodes}
    visited = set()                        # Closed set to prevent reprocessing finalized nodes
    
    open_set.push(source, priority=0.0)
    
    # 2. Main exploration loop
    while not open_set.is_empty():
        u = open_set.extract_min()         # O(log V) extraction of node with smallest g(u)
        
        # Target reached: early stopping when popped guarantees optimal path
        if u == target:
            return reconstruct_path(prev, u), g_score[u]
            
        # Duplicate check for standard binary min-heaps
        if u in visited:
            continue
        visited.add(u)
        
        # 3. Relax outgoing edges
        for v in graph.neighbors(u):
            edge_weight = graph.distance(u, v)
            tentative_g = g_score[u] + edge_weight
            
            # If a strictly cheaper path to v is found through u
            if tentative_g < g_score[v]:
                prev[v] = u
                g_score[v] = tentative_g
                open_set.push(v, priority=tentative_g)
                
    return None, float('inf')              # Target unreachable
```

---

## 3. Pseudocode: Single-Source All-Destinations Dijkstra (Without Early Stopping)

When computing reachability reach maps, cost-to-go lookup tables, or routing to multiple potential goals, the algorithm runs until the entire connected component is exhausted:

```python
def dijkstra_all_destinations(graph, source):
    # 1. Initialize data structures for complete graph exploration
    open_set = MinPriorityQueue()          # Priority queue ordered by g_score
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[source] = 0.0
    prev = {node: None for node in graph.nodes}
    visited = set()                        # Closed set of permanently finalized nodes
    
    open_set.push(source, priority=0.0)
    
    # 2. Exhaustive expansion loop (no target early stop)
    while not open_set.is_empty():
        u = open_set.extract_min()         # Extract closest unfinalized node
        
        if u in visited:
            continue
        visited.add(u)                     # g_score[u] is now permanently optimal
        
        # 3. Relax all outgoing neighbors of u
        for v in graph.neighbors(u):
            edge_weight = graph.distance(u, v)
            tentative_g = g_score[u] + edge_weight
            
            if tentative_g < g_score[v]:
                prev[v] = u
                g_score[v] = tentative_g
                open_set.push(v, priority=tentative_g)
                
    # Returns full shortest-path tree and distance maps to all reachable nodes
    return g_score, prev
```

---

## 4. Complexity Analysis

| Data Structure                | Extract-Min ($\mathcal{Q}$) | Decrease-Key / Push | Overall Time Complexity | Space Complexity           |               |             |                |               |             |             |        |               |             |               |             |     |
| :---------------------------- | :-------------------------- | :------------------ | :---------------------- | :------------------------- | ------------- | ----------- | -------------- | ------------- | ----------- | ----------- | ------ | ------------- | ----------- | ------------- | ----------- | --- |
| **Array / Unordered List**    | $\mathcal{O}(               | \mathcal{V}         | )$                      | $\mathcal{O}(1)$           | $\mathcal{O}( | \mathcal{V} | ^2)$           | $\mathcal{O}( | \mathcal{V} | )$          |        |               |             |               |             |     |
| **Binary Min-Heap (`heapq`)** | $\mathcal{O}(\log           | \mathcal{V}         | )$                      | $\mathcal{O}(\log          | \mathcal{V}   | )$          | $\mathcal{O}(( | \mathcal{E}   | +           | \mathcal{V} | ) \log | \mathcal{V}   | )$          | $\mathcal{O}( | \mathcal{V} | )$  |
| **Fibonacci Heap**            | $\mathcal{O}(\log           | \mathcal{V}         | )$                      | $\mathcal{O}(1)$ amortized | $\mathcal{O}( | \mathcal{E} | +              | \mathcal{V}   | \log        | \mathcal{V} | )$     | $\mathcal{O}( | \mathcal{V} | )$            |             |     |
