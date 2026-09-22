"""
Session 03 Solution: Road Network Graph Construction & Dijkstra's Algorithm
"""

import networkx as nx
import polars as pl
from exercise_dijkstra import create_sample_road_network, dijkstra_shortest_path, visualize_route

if __name__ == "__main__":
    G = create_sample_road_network()
    start_node = 0
    goal_node = 8
    
    path, cost, explored = dijkstra_shortest_path(G, start_node, goal_node)
    assert len(path) > 0, "Dijkstra failed to find path"
    print(f"Dijkstra Solution Verified: Path = {path}, Cost = {cost:.2f}s, Explored = {explored} nodes")
