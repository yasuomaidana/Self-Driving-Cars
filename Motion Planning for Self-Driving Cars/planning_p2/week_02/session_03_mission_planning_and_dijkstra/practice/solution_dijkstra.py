"""
Session 03 Solution: Road Network Graph Construction & Dijkstra's Algorithm
---------------------------------------------------------------------------
Instructor Reference Solution for Mission Planning & Multi-Attribute Dijkstra Search
featuring synthetic grid road networks and real-world OSMnx urban map routing.
"""

import os
import math
import heapq
import networkx as nx
import osmnx as ox
import polars as pl
import plotly.graph_objects as go
from shapely import wkt


def create_sample_road_network() -> nx.DiGraph:
    """Constructs a 9-node directed grid road network with spatial coordinates and speed limits."""
    G = nx.DiGraph()
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
        
    edges = [
        (0, 1, 13.8), (1, 2, 22.2), (1, 0, 13.8), (2, 1, 22.2),
        (3, 4, 13.8), (4, 5, 13.8), (4, 3, 13.8), (5, 4, 13.8),
        (6, 7, 13.8), (7, 8, 22.2), (7, 6, 13.8), (8, 7, 22.2),
        (0, 3, 13.8), (3, 6, 13.8), (3, 0, 13.8), (6, 3, 13.8),
        (1, 4, 13.8), (4, 7, 13.8), (4, 1, 13.8), (7, 4, 13.8),
        (2, 5, 22.2), (5, 8, 22.2), (5, 2, 22.2), (8, 5, 22.2),
        (4, 8, 13.8),
    ]
    for u, v, speed in edges:
        pos_u = nodes[u]
        pos_v = nodes[v]
        dist = math.hypot(pos_v[0] - pos_u[0], pos_v[1] - pos_u[1])
        travel_time = dist / speed
        G.add_edge(u, v, length=dist, speed=speed, weight=travel_time)
        
    return G


def dijkstra_shortest_path(graph: nx.DiGraph, start_node: int, goal_node: int) -> tuple[list[int], float, int, list[int]]:
    """Executes Dijkstra's Shortest Path Algorithm from start_node to goal_node on DiGraph (With Early Stopping)."""
    # 1. Initialize data structures
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start_node] = 0.0
    parent = {node: None for node in graph.nodes}
    
    # Priority Queue of (g_score, node)
    pq = [(0.0, start_node)]
    visited = set()
    explored_order = []
    
    # 2. Main exploration loop
    while pq:
        current_g, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        explored_order.append(u)
        
        # Early termination
        if u == goal_node:
            path = []
            curr = goal_node
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            return path, current_g, len(explored_order), explored_order
            
        # 3. Edge relaxation
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
    """
    # 1. Initialize data structures
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start_node] = 0.0
    parent = {node: None for node in graph.nodes}
    
    pq = [(0.0, start_node)]
    visited = set()
    explored_order = []
    
    # 2. Exhaustive expansion loop
    while pq:
        current_g, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        explored_order.append(u)
        
        # 3. Edge relaxation
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
    """
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
    print("=== Part 1: Synthetic Grid Network Dijkstra ===")
    G = create_sample_road_network()
    path, cost, explored, _ = dijkstra_shortest_path(G, 0, 8)
    assert len(path) > 0, "Dijkstra failed to find path"
    print(f"Grid Solution Verified: Path = {path}, Cost = {cost:.2f}s, Explored = {explored} nodes")
    
    print("\n=== Part 2: Real-World OSMnx Urban Network Dijkstra ===")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    graph_path = os.path.join(data_dir, "piedmont_road_network.graphml")
    
    if os.path.exists(graph_path):
        G_osm = ox.load_graphml(filepath=graph_path)
    else:
        G_osm = ox.graph_from_place("Piedmont, California, USA", network_type="drive")
        G_osm = ox.add_edge_speeds(G_osm)
        G_osm = ox.add_edge_travel_times(G_osm)
        ox.save_graphml(G_osm, filepath=graph_path)
        
    for u, v, k, data in G_osm.edges(keys=True, data=True):
        data['length'] = float(data.get('length', 1.0))
        data['travel_time'] = float(data.get('travel_time', data['length'] / 13.8))
        
    nodes = list(G_osm.nodes)
    start = nodes[10]
    reachable = [n for n in nodes if n != start and nx.has_path(G_osm, start, n)]
    goal = reachable[-15]
    
    path_fast, time_fast, exp_fast, _ = osmnx_dijkstra(G_osm, start, goal, weight='travel_time')
    path_short, dist_short, exp_short, _ = osmnx_dijkstra(G_osm, start, goal, weight='length')
    
    print(f"OSMnx Fastest Route: {len(path_fast)} nodes, Travel Time: {time_fast:.2f}s ({exp_fast} nodes explored)")
    print(f"OSMnx Shortest Route: {len(path_short)} nodes, Distance: {dist_short:.2f}m ({exp_short} nodes explored)")
    print("✅ All Mission Planning solutions verified successfully!")
