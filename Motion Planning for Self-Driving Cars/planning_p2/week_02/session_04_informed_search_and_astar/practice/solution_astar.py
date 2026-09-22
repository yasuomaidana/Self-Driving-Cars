"""
Session 04 Solution: A* Search Algorithm & Heuristic Benchmarking
"""

from exercise_astar import create_grid_road_network, benchmark_search_algorithms, astar_search, dijkstra_search

if __name__ == "__main__":
    G = create_grid_road_network(rows=20, cols=20, spacing=30.0)
    start = 0
    goal = len(G.nodes) - 1
    
    d_path, d_cost, d_exp, _ = dijkstra_search(G, start, goal)
    a_path, a_cost, a_exp, _ = astar_search(G, start, goal, max_speed=20.0)
    
    assert abs(d_cost - a_cost) < 1e-4, f"A* cost ({a_cost}) differs from Dijkstra ({d_cost})"
    print(f"Verification Successful! Both algorithms found identical cost: {a_cost:.2f}s")
    print(f"Nodes Explored - Dijkstra: {d_exp}, A*: {a_exp} (Reduced by {(1 - a_exp/d_exp)*100:.1f}%)")
