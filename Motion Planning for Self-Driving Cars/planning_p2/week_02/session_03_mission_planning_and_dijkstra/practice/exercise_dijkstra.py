"""
Session 03 Practice: Road Network Graph Construction & Dijkstra's Algorithm
---------------------------------------------------------------------------
In this lab, you will construct a directed road network graph and implement
Dijkstra's Shortest Path Algorithm from scratch using a Min-Heap.

Tools used:
- NetworkX: Graph topology modeling
- Polars: Tabular route analysis and node data logging
- Plotly: Interactive network and shortest path visualization
"""

import heapq
import math
import networkx as nx
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


def dijkstra_shortest_path(graph: nx.DiGraph, start_node: int, goal_node: int) -> tuple[list[int], float, int]:
    """
    Executes Dijkstra's Shortest Path Algorithm from start_node to goal_node.
    
    Returns:
        path: List of node IDs from start to goal [start, ..., goal]
        cost: Total shortest path travel time cost
        nodes_explored: Total number of nodes popped from priority queue
        
    TODO (Student):
    1. Initialize g_score dict with float('inf'), g_score[start_node] = 0.0
    2. Maintain parent dict to track optimal predecessors
    3. Use heapq priority queue with entries (current_cost, u)
    4. Explore neighbors, relax edge weights, and push updated costs
    5. Terminate and reconstruct path when goal_node is popped
    """
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start_node] = 0.0
    parent = {node: None for node in graph.nodes}
    
    pq = [(0.0, start_node)]
    visited = set()
    nodes_explored = 0
    
    while pq:
        current_g, u = heapq.heappop(pq)
        
        if u in visited:
            continue
        visited.add(u)
        nodes_explored += 1
        
        if u == goal_node:
            # Reconstruct path
            path = []
            curr = goal_node
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            return path, current_g, nodes_explored
            
        for v in graph.neighbors(u):
            weight = graph[u][v]['weight']
            tentative_g = current_g + weight
            if tentative_g < g_score[v]:
                g_score[v] = tentative_g
                parent[v] = u
                heapq.heappush(pq, (tentative_g, v))
                
    return [], float('inf'), nodes_explored


def visualize_route(graph: nx.DiGraph, path: list[int], output_html: str = "dijkstra_route.html"):
    """Visualizes the road network and computed shortest path using Plotly."""
    edge_x = []
    edge_y = []
    for u, v in graph.edges():
        x0, y0 = graph.nodes[u]['pos']
        x1, y1 = graph.nodes[v]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    node_x = [graph.nodes[n]['pos'][0] for n in graph.nodes]
    node_y = [graph.nodes[n]['pos'][1] for n in graph.nodes]
    node_text = [f"Node {n}" for n in graph.nodes]
    
    fig = go.Figure()
    
    # 1. Base Road Network Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        name='Road Network'
    ))
    
    # 2. Network Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=12, color='lightblue', line=dict(width=2, color='darkblue')),
        text=node_text,
        textposition="top center",
        name='Intersections'
    ))
    
    # 3. Shortest Path Highlight
    if path:
        path_x = [graph.nodes[n]['pos'][0] for n in path]
        path_y = [graph.nodes[n]['pos'][1] for n in path]
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode='lines+markers',
            line=dict(width=5, color='crimson'),
            marker=dict(size=14, color='gold'),
            name='Dijkstra Optimal Route'
        ))
        
    fig.update_layout(
        title="Road Network Mission Routing (Dijkstra's Algorithm)",
        xaxis_title="Global X (m)",
        yaxis_title="Global Y (m)",
        showlegend=True,
        width=850,
        height=700
    )
    
    fig.write_html(output_html)
    print(f"Interactive route visualization saved to {output_html}")


if __name__ == "__main__":
    G = create_sample_road_network()
    start_node = 0
    goal_node = 8
    
    path, cost, explored = dijkstra_shortest_path(G, start_node, goal_node)
    print(f"Optimal Path from Node {start_node} to Node {goal_node}: {path}")
    print(f"Estimated Travel Time: {cost:.2f} seconds")
    print(f"Nodes Explored: {explored}")
    
    # Create route telemetry Polars DataFrame
    route_data = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = G[u][v]
        route_data.append({
            "segment_index": i,
            "from_node": u,
            "to_node": v,
            "length_m": edge_data["length"],
            "speed_limit_mps": edge_data["speed"],
            "segment_time_s": edge_data["weight"]
        })
    df_route = pl.DataFrame(route_data)
    print("\n--- Route Segment Breakdown (Polars DataFrame) ---")
    print(df_route)
    
    visualize_route(G, path, "dijkstra_route_plotly.html")
