# Priority Queues & Graph Data Structures

## 1. Graph Representations in Memory

For road networks spanning tens of thousands of road segments, efficient memory layout and neighbor lookup speed are critical.

### Comparison of Graph Storage Models
1. **Adjacency Matrix:**
   - Space: $\mathcal{O}(|\mathcal{V}|^2)$.
   - For road networks where average node degree is $3 \sim 4$, adjacency matrices are $>99.9\%$ sparse and waste massive amounts of RAM.
2. **Adjacency List (Dict of Lists/Dicts):**
   - Space: $\mathcal{O}(|\mathcal{V}| + |\mathcal{E}|)$.
   - Ideal for road network graphs where degree is bounded ($d \le 6$).
   - Standard in NetworkX and production C++ autonomous routing engines.

---

## 2. Priority Queue Min-Heap Mechanics

In Python, `heapq` provides a binary heap where elements are tuples `(priority, item)`:
- `heappush(heap, (cost, u))`: $\mathcal{O}(\log N)$
- `heappop(heap)`: Returns minimum element in $\mathcal{O}(\log N)$

### Handling Lazy Deletion vs Decrease-Key
Standard binary heaps in Python do not have an $\mathcal{O}(\log N)$ `decrease_key` method. Instead, we use the **lazy duplicate insertion strategy**:
- When a shorter path to a node $v$ is discovered, push a new `(new_g, v)` tuple onto the heap.
- When popping `(cost, u)` from the heap, check `if cost > g_score[u]` or `if u in visited:` and skip stale entries in $\mathcal{O}(1)$.
