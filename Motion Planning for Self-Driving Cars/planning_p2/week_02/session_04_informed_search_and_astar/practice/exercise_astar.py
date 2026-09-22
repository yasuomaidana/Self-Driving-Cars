"""
Session 04 Practice: A* Search Algorithm & Heuristic Benchmarking
----------------------------------------------------------------
In this lab, you will implement the A* search algorithm with an admissible
heuristic, compare its efficiency against Dijkstra, and analyze the results.

Tools used:
- Polars: For search algorithm benchmark metrics and comparison
- Plotly: For interactive visualization of search frontiers
"""

import heapq
import math
import time
import networkx as nx
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_grid_road_network(rows: int = 15, cols: int = 15, spacing: float = 50.0) -> nx.DiGraph:
    """
    Creates an NxM grid road network with random speed limits and turn connections.
    """
    G = nx.DiGraph()
    
    # Create grid nodes
    for r in range(rows):
        for c in range(cols):
            node_id = r * cols + c
            x = c * spacing
            y = r * spacing
            G.add_node(node_id, pos=(x, y))
            
    # Connect horizontal and vertical 2-way streets
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            # Right neighbor
            if c + 1 < cols:
                v = r * cols + (c + 1)
                speed = 20.0 if (r % 3 == 0) else 10.0 # Arterial vs residential
                dist = spacing
                G.add_edge(u, v, length=dist, speed=speed, weight=dist/speed)
                G.add_edge(v, u, length=dist, speed=speed, weight=dist/speed)
            # Up neighbor
            if r + 1 < rows:
                v = (r + 1) * cols + c
                speed = 20.0 if (c % 3 == 0) else 10.0
                dist = spacing
                G.add_edge(u, v, length=dist, speed=speed, weight=dist/speed)
                G.add_edge(v, u, length=dist, speed=speed, weight=dist/speed)
                
    return G


def euclidean_travel_time_heuristic(graph: nx.DiGraph, node: int, goal: int, max_speed: float = 20.0) -> float:
    """
    Admissible and consistent heuristic: Euclidean distance / max_speed_in_network.
    """
    x1, y1 = graph.nodes[node]['pos']
    x2, y2 = graph.nodes[goal]['pos']
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist / max_speed


def astar_search(graph: nx.DiGraph, start_node: int, goal_node: int, max_speed: float = 20.0) -> tuple[list[int], float, int, list[int]]:
    """
    A* search implementation.
    
    TODO (Student):
    1. Initialize g_score with inf, g_score[start] = 0.0
    2. Maintain priority queue sorted by f_score = g + h
    3. Track order of explored nodes for visualization
    4. Reconstruct path from start to goal
    """
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start_node] = 0.0
    
    h_start = euclidean_travel_time_heuristic(graph, start_node, goal_node, max_speed)
    pq = [(h_start, 0.0, start_node)] # (f_score, g_score, node)
    parent = {node: None for node in graph.nodes}
    
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
            weight = graph[u][v]['weight']
            tentative_g = current_g + weight
            if tentative_g < g_score[v]:
                g_score[v] = tentative_g
                parent[v] = u
                h_v = euclidean_travel_time_heuristic(graph, v, goal_node, max_speed)
                heapq.heappush(pq, (tentative_g + h_v, tentative_g, v))
                
    return [], float('inf'), len(explored_order), explored_order


def dijkstra_search(graph: nx.DiGraph, start_node: int, goal_node: int) -> tuple[list[int], float, int, list[int]]:
    """Uniform-cost Dijkstra search for comparison."""
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start_node] = 0.0
    pq = [(0.0, start_node)]
    parent = {node: None for node in graph.nodes}
    visited = set()
    explored_order = []
    
    while pq:
        current_g, u = heapq.heappop(pq)
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
            weight = graph[u][v]['weight']
            tentative_g = current_g + weight
            if tentative_g < g_score[v]:
                g_score[v] = tentative_g
                parent[v] = u
                heapq.heappush(pq, (tentative_g, v))
                
    return [], float('inf'), len(explored_order), explored_order


def benchmark_search_algorithms(graph: nx.DiGraph, start: int, goal: int) -> pl.DataFrame:
    """Runs and logs performance comparison using Polars."""
    t0 = time.perf_counter()
    d_path, d_cost, d_exp, _ = dijkstra_search(graph, start, goal)
    t_dijkstra = (time.perf_counter() - t0) * 1000 # ms
    
    t0 = time.perf_counter()
    a_path, a_cost, a_exp, _ = astar_search(graph, start, goal, max_speed=20.0)
    t_astar = (time.perf_counter() - t0) * 1000 # ms
    
    df_benchmark = pl.DataFrame([
        {
            "Algorithm": "Dijkstra (Uniform Cost)",
            "Optimal Cost (s)": round(d_cost, 2),
            "Path Length (nodes)": len(d_path),
            "Nodes Explored": d_exp,
            "Exploration %": round((d_exp / len(graph.nodes)) * 100, 1),
            "Runtime (ms)": round(t_dijkstra, 3)
        },
        {
            "Algorithm": "A* (Euclidean Time Heuristic)",
            "Optimal Cost (s)": round(a_cost, 2),
            "Path Length (nodes)": len(a_path),
            "Nodes Explored": a_exp,
            "Exploration %": round((a_exp / len(graph.nodes)) * 100, 1),
            "Runtime (ms)": round(t_astar, 3)
        }
    ])
    return df_benchmark


if __name__ == "__main__":
    G = create_grid_road_network(rows=15, cols=15, spacing=40.0)
    start = 0
    goal = len(G.nodes) - 1 # bottom-left to top-right
    
    df_results = benchmark_search_algorithms(G, start, goal)
    print("--- Benchmark Results (Polars DataFrame) ---")
    print(df_results)
