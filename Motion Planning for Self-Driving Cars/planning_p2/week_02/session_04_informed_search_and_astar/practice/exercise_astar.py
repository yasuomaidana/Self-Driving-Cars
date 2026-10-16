"""
Session 04 Practice: Informed Search with A* Routing & Heuristic Benchmarking
---------------------------------------------------------------------------
In this lab, you will implement the A* search algorithm with an admissible
heuristic, compare its efficiency against Dijkstra, and extend it to real-world
OpenStreetMap (OSMnx) road networks.

Tools used:
- NetworkX: Graph topology modeling
- OSMnx: Real-world OpenStreetMap graph acquisition & preprocessing
- Polars: Tabular route analysis and node data logging
- Plotly: Interactive network and shortest path visualization
"""

import os
import math
import heapq
import time
import networkx as nx
import osmnx as ox
import polars as pl
import plotly.graph_objects as go


# =========================================================================
# Part 1: Synthetic Grid Road Network
# =========================================================================

def create_grid_road_network(rows: int = 15, cols: int = 15, spacing: float = 50.0) -> nx.DiGraph:
    """Creates an NxM grid road network with differentiated speed limits."""
    G = nx.DiGraph()
    for r in range(rows):
        for c in range(cols):
            node_id = r * cols + c
            G.add_node(node_id, pos=(c * spacing, r * spacing))
            
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                v = r * cols + (c + 1)
                speed = 20.0 if (r % 3 == 0) else 10.0
                G.add_edge(u, v, length=spacing, speed=speed, weight=spacing / speed)
                G.add_edge(v, u, length=spacing, speed=speed, weight=spacing / speed)
            if r + 1 < rows:
                v = (r + 1) * cols + c
                speed = 20.0 if (c % 3 == 0) else 10.0
                G.add_edge(u, v, length=spacing, speed=speed, weight=spacing / speed)
                G.add_edge(v, u, length=spacing, speed=speed, weight=spacing / speed)
                
    return G


def euclidean_travel_time_heuristic(graph: nx.DiGraph, node: int, goal: int, max_speed: float = 20.0) -> float:
    """Admissible and consistent Euclidean travel-time heuristic."""
    x1, y1 = graph.nodes[node]['pos']
    x2, y2 = graph.nodes[goal]['pos']
    return math.hypot(x2 - x1, y2 - y1) / max_speed


def astar_search(
    graph: nx.DiGraph, 
    start_node: int, 
    goal_node: int, 
    max_speed: float = 20.0, 
    weight_h: float = 1.0
) -> tuple[list[int], float, int, list[int]]:
    """
    TODO (Student): Implement Forward A* Search on a Directed Graph.
    
    Parameters:
        graph: NetworkX DiGraph with edge attribute 'weight' (travel time in seconds).
        start_node: Origin node ID.
        goal_node: Destination node ID.
        max_speed: Maximum vehicle speed for Euclidean travel time heuristic (m/s).
        weight_h: Heuristic weight (1.0 for standard A*, 0.0 for uniform-cost Dijkstra, >1.0 for Weighted A*).
        
    Returns:
        path: Ordered list of node IDs from start_node to goal_node [start, ..., goal].
        cost: Exact total accumulated travel time cost along the optimal path.
        nodes_explored: Total count of nodes extracted from the priority queue.
        explored_order: Sequence of node IDs in order of extraction.
        
    Algorithm Steps:
    1. Initialize g_score dict with float('inf') for all nodes in graph; set g_score[start_node] = 0.0.
    2. Compute initial heuristic h_start = euclidean_travel_time_heuristic(graph, start_node, goal_node, max_speed).
    3. Initialize min-heap priority queue storing tuples (f_score, g_score, node_id), seeded with (weight_h * h_start, 0.0, start_node).
    4. Maintain parent dict to track optimal predecessors and visited set to track closed nodes.
    5. Loop while pq is not empty:
       a. Pop (f, current_g, u) with lowest f_score.
       b. If u in visited, skip; otherwise add u to visited and record in explored_order.
       c. If u == goal_node, reconstruct and return the optimal path from start to goal.
       d. For each neighbor v of u:
          - tentative_g = current_g + edge weight
          - If tentative_g < g_score[v]: update g_score[v], parent[v] = u, compute h_v, and push (tentative_g + weight_h * h_v, tentative_g, v) to pq.
    6. Return ([], float('inf'), len(explored_order), explored_order) if goal is unreachable.
    """
    # >>> YOUR CODE HERE >>>
    raise NotImplementedError("Implement astar_search!")
    # <<< YOUR CODE HERE <<<


def dijkstra_search(graph: nx.DiGraph, start_node: int, goal_node: int):
    """Uniform-cost search (A* with h=0)."""
    return astar_search(graph, start_node, goal_node, weight_h=0.0)


# =========================================================================
# Part 2: Real-World OSMnx Urban Network Search
# =========================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle Haversine distance between two GPS coordinates in meters."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def osmnx_travel_time_heuristic(graph: nx.MultiDiGraph, u: int, goal: int, max_speed_mps: float) -> float:
    """Admissible and consistent Haversine travel-time heuristic for geographic road maps."""
    u_lat, u_lon = graph.nodes[u]['y'], graph.nodes[u]['x']
    g_lat, g_lon = graph.nodes[goal]['y'], graph.nodes[goal]['x']
    dist_m = haversine_distance(u_lat, u_lon, g_lat, g_lon)
    return dist_m / max_speed_mps


def osmnx_astar(
    graph: nx.MultiDiGraph, 
    start_node: int, 
    goal_node: int, 
    weight: str = 'travel_time',
    weight_h: float = 1.0,
    max_speed_mps: float = 15.65
) -> tuple[list[int], float, int, list[int]]:
    """
    Executes A* search on an OSMnx MultiDiGraph.
    
    TODO (Student):
    1. Initialize g_score dict with float('inf') for all nodes, g_score[start_node] = 0.0
    2. Compute h_start = osmnx_travel_time_heuristic(graph, start_node, goal_node, max_speed_mps)
    3. Maintain priority queue sorted by f_score = g + w_h * h: pq = [(w_h * h_start, 0.0, start_node)]
    4. Pop node with lowest f_score, relax MultiDiGraph edges, and update parent pointers
    5. Terminate and reconstruct path when goal_node is popped
    """
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start_node] = 0.0
    parent = {node: None for node in graph.nodes}
    
    h_start = osmnx_travel_time_heuristic(graph, start_node, goal_node, max_speed_mps) if weight_h > 0 else 0.0
    pq = [(weight_h * h_start, 0.0, start_node)]
    visited = set()
    explored_order = []
    
    while pq:
        f, current_g, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        explored_order.append(u)
        
        if u == goal_node:
            path = []
            curr = goal_node
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            return path, current_g, len(explored_order), explored_order
            
        for v in graph.neighbors(u):
            edge_data = graph[u][v]
            if isinstance(graph, (nx.MultiDiGraph, nx.MultiGraph)):
                w = min(float(d.get(weight, float('inf'))) for d in edge_data.values())
            else:
                w = float(edge_data.get(weight, float('inf')))
                
            tentative_g = current_g + w
            if tentative_g < g_score[v]:
                g_score[v] = tentative_g
                parent[v] = u
                h_v = osmnx_travel_time_heuristic(graph, v, goal_node, max_speed_mps) if weight_h > 0 else 0.0
                heapq.heappush(pq, (tentative_g + weight_h * h_v, tentative_g, v))
                
    return [], float('inf'), len(explored_order), explored_order


# =========================================================================
# Part 3: Bidirectional A* Search
# =========================================================================

def bidirectional_astar_search(
    graph: nx.DiGraph, 
    start_node: int, 
    goal_node: int, 
    max_speed: float = 20.0
) -> tuple[list[int], float, int, tuple[list[int], list[int]]]:
    """
    Bidirectional A* search on a directed graph with Euclidean heuristic.
    
    TODO (Student):
    1. Initialize forward and backward g_scores: g_f[start]=0, g_b[goal]=0.
    2. Maintain forward priority queue pq_f with h(u, goal) and backward pq_b with h(v, start).
    3. Alternate exploration (expand direction with smaller queue size).
    4. For forward search: relax graph.neighbors(u). For backward search: relax graph.predecessors(v).
    5. Check connection and maintain best_cost = min(best_cost, g_f[x] + g_b[x]).
    6. Stop when pq_f[0].f + pq_b[0].f >= best_cost + h(start, goal).
    7. Reconstruct path by concatenating forward path to meeting_node and reversed backward path.
    """
    if start_node == goal_node:
        return [start_node], 0.0, 1, ([start_node], [])
        
    g_f = {node: float('inf') for node in graph.nodes}
    g_f[start_node] = 0.0
    parent_f = {node: None for node in graph.nodes}
    h_start = euclidean_travel_time_heuristic(graph, start_node, goal_node, max_speed)
    pq_f = [(h_start, 0.0, start_node)]
    vis_f = set()
    
    g_b = {node: float('inf') for node in graph.nodes}
    g_b[goal_node] = 0.0
    parent_b = {node: None for node in graph.nodes}
    h_goal = euclidean_travel_time_heuristic(graph, goal_node, start_node, max_speed)
    pq_b = [(h_goal, 0.0, goal_node)]
    vis_b = set()
    
    best_cost = float('inf')
    meeting_node = None
    
    while pq_f and pq_b:
        if pq_f[0][0] + pq_b[0][0] >= best_cost + h_start:
            break
            
        if len(pq_f) <= len(pq_b):
            f_curr, curr_g, u = heapq.heappop(pq_f)
            if u in vis_f:
                continue
            vis_f.add(u)
            
            if u in vis_b and curr_g + g_b[u] < best_cost:
                best_cost = curr_g + g_b[u]
                meeting_node = u
                
            for v in graph.neighbors(u):
                w = graph[u][v]['weight']
                tentative_g = curr_g + w
                if tentative_g < g_f[v]:
                    g_f[v] = tentative_g
                    parent_f[v] = u
                    h_v = euclidean_travel_time_heuristic(graph, v, goal_node, max_speed)
                    heapq.heappush(pq_f, (tentative_g + h_v, tentative_g, v))
                    if v in vis_b and tentative_g + g_b[v] < best_cost:
                        best_cost = tentative_g + g_b[v]
                        meeting_node = v
        else:
            f_curr, curr_g, v = heapq.heappop(pq_b)
            if v in vis_b:
                continue
            vis_b.add(v)
            
            if v in vis_f and g_f[v] + curr_g < best_cost:
                best_cost = g_f[v] + curr_g
                meeting_node = v
                
            for u in graph.predecessors(v):
                w = graph[u][v]['weight']
                tentative_g = curr_g + w
                if tentative_g < g_b[u]:
                    g_b[u] = tentative_g
                    parent_b[u] = v
                    h_u = euclidean_travel_time_heuristic(graph, u, start_node, max_speed)
                    heapq.heappush(pq_b, (tentative_g + h_u, tentative_g, u))
                    if u in vis_f and g_f[u] + tentative_g < best_cost:
                        best_cost = g_f[u] + tentative_g
                        meeting_node = u
                        
    if meeting_node is None:
        return [], float('inf'), len(vis_f) + len(vis_b), (list(vis_f), list(vis_b))
        
    path_f = []
    curr = meeting_node
    while curr is not None:
        path_f.append(curr)
        curr = parent_f[curr]
    path_f.reverse()
    
    path_b = []
    curr = parent_b[meeting_node]
    while curr is not None:
        path_b.append(curr)
        curr = parent_b[curr]
        
    full_path = path_f + path_b
    total_explored = len(vis_f) + len(vis_b)
    return full_path, best_cost, total_explored, (list(vis_f), list(vis_b))


def osmnx_bidirectional_astar(
    graph: nx.MultiDiGraph,
    start_node: int,
    goal_node: int,
    weight: str = 'travel_time',
    max_speed_mps: float = 15.65
) -> tuple[list[int], float, int, tuple[list[int], list[int]]]:
    """
    Bidirectional A* search on OSMnx MultiDiGraph.
    
    TODO (Student):
    1. Initialize forward search from start_node and backward search from goal_node.
    2. Use graph.neighbors(u) for forward expansion and graph.predecessors(v) for backward expansion.
    3. Account for MultiDiGraph parallel edge weights: min(float(d.get(weight, inf)) for d in graph[u][v].values()).
    4. Terminate when forward top f + backward top f exceeds best_cost + h(start, goal).
    5. Reconstruct the end-to-end optimal path across the meeting node.
    """
    if start_node == goal_node:
        return [start_node], 0.0, 1, ([start_node], [])
        
    g_f = {n: float('inf') for n in graph.nodes}
    g_f[start_node] = 0.0
    parent_f = {n: None for n in graph.nodes}
    h_start = osmnx_travel_time_heuristic(graph, start_node, goal_node, max_speed_mps)
    pq_f = [(h_start, 0.0, start_node)]
    vis_f = set()
    
    g_b = {n: float('inf') for n in graph.nodes}
    g_b[goal_node] = 0.0
    parent_b = {n: None for n in graph.nodes}
    h_goal = osmnx_travel_time_heuristic(graph, goal_node, start_node, max_speed_mps)
    pq_b = [(h_goal, 0.0, goal_node)]
    vis_b = set()

    best_cost = float('inf')
    meeting_node = None

    while pq_f and pq_b:
        if pq_f[0][0] + pq_b[0][0] >= best_cost + h_start:
            break
            
        if len(pq_f) <= len(pq_b):
            f_curr, curr_g, u = heapq.heappop(pq_f)
            if u in vis_f:
                continue
            vis_f.add(u)
            
            if u in vis_b and curr_g + g_b[u] < best_cost:
                best_cost = curr_g + g_b[u]
                meeting_node = u
                
            for v in graph.neighbors(u):
                w = min(float(d.get(weight, float('inf'))) for d in graph[u][v].values())
                tentative_g = curr_g + w
                if tentative_g < g_f[v]:
                    g_f[v] = tentative_g
                    parent_f[v] = u
                    h_v = osmnx_travel_time_heuristic(graph, v, goal_node, max_speed_mps)
                    heapq.heappush(pq_f, (tentative_g + h_v, tentative_g, v))
                    if v in vis_b and tentative_g + g_b[v] < best_cost:
                        best_cost = tentative_g + g_b[v]
                        meeting_node = v
        else:
            f_curr, curr_g, v = heapq.heappop(pq_b)
            if v in vis_b:
                continue
            vis_b.add(v)
            
            if v in vis_f and g_f[v] + curr_g < best_cost:
                best_cost = g_f[v] + curr_g
                meeting_node = v
                
            for u in graph.predecessors(v):
                w = min(float(d.get(weight, float('inf'))) for d in graph[u][v].values())
                tentative_g = curr_g + w
                if tentative_g < g_b[u]:
                    g_b[u] = tentative_g
                    parent_b[u] = v
                    h_u = osmnx_travel_time_heuristic(graph, u, start_node, max_speed_mps)
                    heapq.heappush(pq_b, (tentative_g + h_u, tentative_g, u))
                    if u in vis_f and g_f[u] + tentative_g < best_cost:
                        best_cost = g_f[u] + tentative_g
                        meeting_node = u

    if meeting_node is None:
        return [], float('inf'), len(vis_f) + len(vis_b), (list(vis_f), list(vis_b))

    path_f = []
    curr = meeting_node
    while curr is not None:
        path_f.append(curr)
        curr = parent_f[curr]
    path_f.reverse()

    path_b = []
    curr = parent_b[meeting_node]
    while curr is not None:
        path_b.append(curr)
        curr = parent_b[curr]

    full_path = path_f + path_b
    total_explored = len(vis_f) + len(vis_b)
    return full_path, best_cost, total_explored, (list(vis_f), list(vis_b))


# =========================================================================
# Part 4: Iterative Deepening A* (IDA*) Search
# =========================================================================

def ida_star_search(
    graph: nx.DiGraph, 
    start_node: int, 
    goal_node: int, 
    max_speed: float = 20.0
) -> tuple[list[int], float, int, int]:
    """
    Iterative Deepening A* (IDA*) Search implementation.
    
    TODO (Student):
    1. Initialize threshold = euclidean_travel_time_heuristic(graph, start_node, goal_node, max_speed).
    2. Implement recursive search(node, g, current_thresh):
       - If f = g + h > current_thresh, return (f, False).
       - If node == goal_node, return (f, True).
       - For each neighbor not in current path, recurse and track min_exceeded threshold.
    3. Update threshold = min_exceeded across successive iterations until goal is reached.
    """
    threshold = euclidean_travel_time_heuristic(graph, start_node, goal_node, max_speed)
    path = [start_node]
    total_evaluations = 0
    iterations = 0
    
    def search(node: int, g: float, current_thresh: float) -> tuple[float, bool]:
        nonlocal total_evaluations
        total_evaluations += 1
        f = g + euclidean_travel_time_heuristic(graph, node, goal_node, max_speed)
        if f > current_thresh + 1e-9:
            return f, False
        if node == goal_node:
            return f, True
            
        min_exceeded = float('inf')
        for v in graph.neighbors(node):
            if v not in path:
                path.append(v)
                w = graph[node][v]['weight']
                t, found = search(v, g + w, current_thresh)
                if found:
                    return t, True
                if t < min_exceeded:
                    min_exceeded = t
                path.pop()
        return min_exceeded, False

    while True:
        iterations += 1
        t, found = search(start_node, 0.0, threshold)
        if found:
            return list(path), t, total_evaluations, iterations
        if t == float('inf'):
            return [], float('inf'), total_evaluations, iterations
        threshold = t


def osmnx_ida_star(
    graph: nx.MultiDiGraph,
    start_node: int,
    goal_node: int,
    weight: str = 'travel_time',
    max_speed_mps: float = 15.65,
    step_size: float = 5.0
) -> tuple[list[int], float, int, int]:
    """
    Iterative Deepening A* (IDA*) Search on OSMnx MultiDiGraph.
    Memory: O(d) call-stack depth.
    
    TODO (Student):
    1. Set initial threshold = osmnx_travel_time_heuristic(graph, start_node, goal_node, max_speed_mps).
    2. Implement recursive search(node, g, current_thresh, best_g):
       - If f = g + h > current_thresh, return (f, False).
       - If node == goal_node, return (f, True).
       - For each neighbor v not in active path, recurse with minimum edge travel_time.
    3. Update threshold = max(min_exceeded, threshold + step_size) across iterations until goal is found.
    """
    threshold = osmnx_travel_time_heuristic(graph, start_node, goal_node, max_speed_mps)
    path = [start_node]
    total_evals = 0
    iteration = 0

    def search(node: int, g: float, current_thresh: float, best_g: dict) -> tuple[float, bool]:
        nonlocal total_evals
        total_evals += 1
        f = g + osmnx_travel_time_heuristic(graph, node, goal_node, max_speed_mps)
        if f > current_thresh + 1e-4:
            return f, False
        if node == goal_node:
            return f, True

        min_exceeded = float('inf')
        for v in graph.neighbors(node):
            if v not in path:
                if isinstance(graph, (nx.MultiDiGraph, nx.MultiGraph)):
                    w = min(float(d.get(weight, float('inf'))) for d in graph[node][v].values())
                else:
                    w = float(graph[node][v].get(weight, float('inf')))
                tentative_g = g + w
                if v in best_g and tentative_g >= best_g[v] - 1e-6:
                    continue
                best_g[v] = tentative_g
                path.append(v)
                t, found = search(v, tentative_g, current_thresh, best_g)
                if found:
                    return t, True
                if t < min_exceeded:
                    min_exceeded = t
                path.pop()
        return min_exceeded, False

    while iteration < 200:
        iteration += 1
        best_g = {start_node: 0.0}
        t, found = search(start_node, 0.0, threshold, best_g)
        if found:
            return list(path), t, total_evals, iteration
        if t == float('inf'):
            break
        threshold = max(t, threshold + step_size)

    return [], float('inf'), total_evals, iteration


if __name__ == "__main__":
    G_grid = create_grid_road_network(rows=8, cols=8, spacing=40.0)
    grid_start = 0
    grid_goal = len(G_grid.nodes) - 1
    d_path, d_cost, d_exp, _ = dijkstra_search(G_grid, grid_start, grid_goal)
    a_path, a_cost, a_exp, _ = astar_search(G_grid, grid_start, grid_goal, max_speed=20.0, weight_h=1.0)
    bi_path, bi_cost, bi_exp, _ = bidirectional_astar_search(G_grid, grid_start, grid_goal, max_speed=20.0)
    ida_path, ida_cost, ida_evals, ida_iters = ida_star_search(G_grid, grid_start, grid_goal, max_speed=20.0)
    print(f"Grid Dijkstra:         {d_exp:5d} nodes | Cost: {d_cost:.2f}s")
    print(f"Grid Forward A*:       {a_exp:5d} nodes | Cost: {a_cost:.2f}s (Saved {(1 - a_exp/d_exp)*100:.1f}%)")
    print(f"Grid Bidirectional A*: {bi_exp:5d} nodes | Cost: {bi_cost:.2f}s (Saved {(1 - bi_exp/d_exp)*100:.1f}%)")
    print(f"Grid IDA* Search:      {ida_evals:5d} evals | Cost: {ida_cost:.2f}s ({ida_iters} iters, Memory O(d)={len(ida_path)})")



