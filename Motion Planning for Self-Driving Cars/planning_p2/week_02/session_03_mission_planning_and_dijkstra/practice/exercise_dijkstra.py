"""
Session 03 Practice: Road Network Graph Construction & Dijkstra's Algorithm
---------------------------------------------------------------------------
In this lab, you will construct a directed road network graph and implement
Dijkstra's Shortest Path Algorithm from scratch using a Min-Heap.
Then, you will extend it to real-world OpenStreetMap (OSMnx) urban road networks.

Tools used:
- NetworkX: Graph topology modeling
- OSMnx: Real-world OpenStreetMap graph acquisition & preprocessing
- Polars: Tabular route analysis and node data logging
- Plotly: Interactive network and shortest path visualization
"""

import os
import heapq
import math
import networkx as nx
import osmnx as ox
import polars as pl
import plotly.graph_objects as go


def create_sample_road_network() -> nx.DiGraph:
    """
    Constructs a directed road network graph with node (x, y) coordinates
    and edge attributes (length, speed_limit_mps).
    """
    G = nx.DiGraph()
    
    # Define intersection nodes with (x, y) coordinates in meters
    nodes = {
        0: (0.0, 0.0),
        1: (100.0, 0.0),
        2: (250.0, 0.0),
        3: (0.0, 100.0),
        4: (100.0, 100.0),
        5: (250.0, 100.0),
        6: (0.0, 250.0),
        7: (120.0, 250.0),
        8: (250.0, 250.0),
    }
    for node_id, pos in nodes.items():
        G.add_node(node_id, pos=pos)
        
    # Directed road segments (u, v, speed_limit_m/s)
    # 50 km/h ~ 13.8 m/s, 80 km/h ~ 22.2 m/s
    edges = [
        (0, 1, 13.8), (1, 2, 22.2), (1, 0, 13.8), (2, 1, 22.2),
        (3, 4, 13.8), (4, 5, 13.8), (4, 3, 13.8), (5, 4, 13.8),
        (6, 7, 13.8), (7, 8, 22.2), (7, 6, 13.8), (8, 7, 22.2),
        (0, 3, 13.8), (3, 6, 13.8), (3, 0, 13.8), (6, 3, 13.8),
        (1, 4, 13.8), (4, 7, 13.8), (4, 1, 13.8), (7, 4, 13.8),
        (2, 5, 22.2), (5, 8, 22.2), (5, 2, 22.2), (8, 5, 22.2),
        (4, 8, 13.8), # diagonal avenue
    ]
    
    for u, v, speed in edges:
        pos_u = nodes[u]
        pos_v = nodes[v]
        dist = math.hypot(pos_v[0] - pos_u[0], pos_v[1] - pos_u[1])
        # Traversal time cost = dist / speed
        travel_time = dist / speed
        G.add_edge(u, v, length=dist, speed=speed, weight=travel_time)
        
    return G


def dijkstra_shortest_path(graph: nx.DiGraph, start_node: int, goal_node: int) -> tuple[list[int], float, int, list[int]]:
    """
    Executes Dijkstra's Shortest Path Algorithm from start_node to goal_node (With Early Stopping).
    
    Returns:
        path: List of node IDs from start to goal [start, ..., goal]
        cost: Total shortest path travel time cost
        nodes_explored: Total number of nodes popped from priority queue
        explored_order: Ordered list of popped node IDs
    """
    # 1. Initialize data structures
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start_node] = 0.0
    parent = {node: None for node in graph.nodes}
    
    # Priority Queue of (g_score, node) - Min-heap sorts lowest cost to front
    pq = [(0.0, start_node)]
    visited = set()
    explored_order = []
    
    # 2. Main exploration loop
    while pq:
        current_g, u = heapq.heappop(pq)  # Extract node with smallest tentative distance
        
        # Stale entry check for standard binary min-heap
        if u in visited:
            continue
        visited.add(u)
        explored_order.append(u)
        
        # Early termination: stopping as soon as target is extracted guarantees optimality
        if u == goal_node:
            path = []
            curr = goal_node
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            return path, current_g, len(explored_order), explored_order
            
        # 3. Relax outgoing edges
        for v in graph.neighbors(u):
            weight = graph[u][v]['weight']
            tentative_g = current_g + weight
            if tentative_g < g_score[v]:
                g_score[v] = tentative_g
                parent[v] = u
                heapq.heappush(pq, (tentative_g, v))
                
    return [], float('inf'), len(explored_order), explored_order


def dijkstra_all_destinations(graph: nx.DiGraph, start_node: int) -> tuple[dict[int, float], dict[int, int | None], int, list[int]]:
    """
    Executes Single-Source All-Destinations Dijkstra's Algorithm (Without Early Stopping).
    Explores the entire connected component to compute the full shortest-path tree and distance map.
    
    Returns:
        g_score: Dictionary mapping each node to its exact minimum cost from start_node
        parent: Dictionary mapping each node to its optimal predecessor
        nodes_explored: Total number of nodes popped from priority queue
        explored_order: Ordered list of popped node IDs
    """
    # 1. Initialize data structures for complete graph exploration
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start_node] = 0.0
    parent = {node: None for node in graph.nodes}
    
    pq = [(0.0, start_node)]
    visited = set()
    explored_order = []
    
    # 2. Exhaustive expansion loop (no target early stop)
    while pq:
        current_g, u = heapq.heappop(pq)
        
        if u in visited:
            continue
        visited.add(u)
        explored_order.append(u)
        
        # 3. Relax all outgoing neighbors of u
        for v in graph.neighbors(u):
            weight = graph[u][v]['weight']
            tentative_g = current_g + weight
            if tentative_g < g_score[v]:
                g_score[v] = tentative_g
                parent[v] = u
                heapq.heappush(pq, (tentative_g, v))
                
    return g_score, parent, len(explored_order), explored_order


def osmnx_dijkstra(
    graph: nx.MultiDiGraph, 
    start_node: int, 
    goal_node: int, 
    weight: str = 'travel_time'
) -> tuple[list[int], float, int, list[int]]:
    """
    Executes Dijkstra's Shortest Path Algorithm on an OSMnx MultiDiGraph with arbitrary edge weight.
    
    TODO (Student):
    1. Initialize g_score dict with float('inf') for all nodes in graph, g_score[start_node] = 0.0
    2. Maintain parent dict to track optimal predecessors
    3. Use heapq priority queue with entries (current_cost, u)
    4. Handle MultiDiGraph edge weights: w = min(float(d.get(weight, inf)) for d in graph[u][v].values())
    5. Explore neighbors, relax edge weights, and push updated costs
    6. Terminate and reconstruct path when goal_node is popped
    """
    # =========================================================================
    # TODO: Implement MultiDiGraph Dijkstra
    # =========================================================================
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start_node] = 0.0
    parent = {node: None for node in graph.nodes}
    
    pq = [(0.0, start_node)]
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
            edge_data = graph[u][v]
            if isinstance(graph, (nx.MultiDiGraph, nx.MultiGraph)):
                w = min(float(d.get(weight, float('inf'))) for d in edge_data.values())
            else:
                w = float(edge_data.get(weight, float('inf')))
                
            tentative_g = current_g + w
            if tentative_g < g_score[v]:
                g_score[v] = tentative_g
                parent[v] = u
                heapq.heappush(pq, (tentative_g, v))
                
    return [], float('inf'), len(explored_order), explored_order


if __name__ == "__main__":
    G = create_sample_road_network()
    start_node = 0
    goal_node = 8
    
    path, cost, explored, _ = dijkstra_shortest_path(G, start_node, goal_node)
    print(f"Optimal Path from Node {start_node} to Node {goal_node}: {path}")
    print(f"Estimated Travel Time: {cost:.2f} seconds ({explored} nodes explored)")
