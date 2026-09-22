# A* Search Algorithm in Driving Environments

## 1. The Evaluation Function $f(n)$

While Dijkstra's algorithm searches uniformly in all directions from the origin ($g(n)$ alone), the **A\*** (A-Star) search algorithm focuses its exploration toward the goal by incorporating a heuristic estimation $h(n)$:

$$f(n) = g(n) + h(n)$$

Where:
- $g(n)$: Exact accumulated cost from the start node $s$ to node $n$.
- $h(n)$: Estimated remaining cost from node $n$ to the goal node $g$.
- $f(n)$: Estimated total cost of the cheapest path from $s$ to $g$ passing through node $n$.

---

## 2. A* Search Mechanics

```python
def a_star(Graph G, Node start, Node goal, Heuristic h):
    g_score = {node: infinity for node in G.nodes}
    g_score[start] = 0
    f_score = {node: infinity for node in G.nodes}
    f_score[start] = h(start, goal)
    
    parent = {node: None for node in G.nodes}
    frontier = PriorityQueue() # ordered by f_score
    frontier.push((f_score[start], start))
    visited = set()
    
    while not frontier.is_empty():
        current_f, u = frontier.pop_min()
        
        if u == goal:
            return reconstruct_path(parent, goal)
            
        if u in visited:
            continue
        visited.add(u)
        
        for neighbor, edge_weight in G.neighbors(u):
            tentative_g = g_score[u] + edge_weight
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h(neighbor, goal)
                parent[neighbor] = u
                frontier.push((f_score[neighbor], neighbor))
                
    return None
```
