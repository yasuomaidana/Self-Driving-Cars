"""
Dijkstra Search Wavefront Video & Animation Generator
Session 03: Mission Planning on Real-World Urban Networks (OSMnx)
-----------------------------------------------------------------
Generates:
1. Priority Queue Early-Termination Scenario:
   - Mid-range destination where Dijkstra terminates immediately upon reaching the goal
   - Clearly shows ~66.2% of the network unvisited/pruned by the priority queue
2. Full-Network Traversal Scenario:
   - Distant destination at the perimeter demonstrating full-city expansion
"""

import os
import heapq
import subprocess
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from shapely import wkt


def record_dijkstra(graph: nx.MultiDiGraph, start: int, goal: int, weight: str = 'travel_time'):
    """Executes Dijkstra while recording complete step-by-step search state history."""
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0.0
    parent = {node: None for node in graph.nodes}
    pq = [(0.0, start)]
    visited = set()
    history = []
    step = 0
    
    while pq:
        current_g, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        
        frontier = [item[1] for item in pq if item[1] not in visited]
        history.append({
            'step': step,
            'current_node': u,
            'current_g': current_g,
            'visited_nodes': list(visited),
            'parent_map': parent.copy(),
            'frontier_nodes': frontier,
            'frontier_size': len(frontier),
            'status': 'SEARCHING'
        })
        step += 1
        
        if u == goal:
            path = []
            curr = goal
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            history[-1]['status'] = 'GOAL_REACHED'
            history[-1]['path'] = path
            return history, path, current_g
            
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
                
    return history, [], float('inf')


def render_scenario_video(
    G: nx.MultiDiGraph, 
    start_node: int, 
    goal_node: int, 
    title_scenario: str, 
    output_mp4: str, 
    output_gif: str, 
    fps: int = 25
):
    print(f"\n--- Generating Video: {title_scenario} ---")
    print(f"Start: {start_node}, Goal: {goal_node}")
    
    history, path, cost = record_dijkstra(G, start_node, goal_node, weight='travel_time')
    
    step_stride = 1 if len(history) < 150 else 2
    subsampled = history[::step_stride]
    if history[-1] not in subsampled:
        subsampled.append(history[-1])
        
    end_frame = history[-1].copy()
    end_frame['status'] = 'COMPLETE'
    frames_data = subsampled + [end_frame] * 30
    
    print(f"Recorded {len(history)} steps. Rendering {len(frames_data)} frames...")
    
    edge_segments = []
    for u, v, data in G.edges(data=True):
        if 'geometry' in data:
            geom = data['geometry']
            if isinstance(geom, str):
                geom = wkt.loads(geom)
            xs, ys = geom.xy
            edge_segments.append(list(zip(xs, ys)))
        else:
            edge_segments.append([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])

    fig = plt.figure(figsize=(16, 9), facecolor='#0f172a', dpi=120)
    gs = GridSpec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[1.0, 1.0], figure=fig, left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.22, hspace=0.28)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_cost = fig.add_subplot(gs[0, 1])
    ax_front = fig.add_subplot(gs[1, 1])

    for ax in [ax_map, ax_cost, ax_front]:
        ax.set_facecolor('#1e293b')

    fig.suptitle(f"Dijkstra Priority Queue Search: {title_scenario}", 
                 color='#f8fafc', fontsize=16, fontweight='bold')

    node_xs = [G.nodes[n]['x'] for n in G.nodes]
    node_ys = [G.nodes[n]['y'] for n in G.nodes]
    x_min, x_max = min(node_xs), max(node_xs)
    y_min, y_max = min(node_ys), max(node_ys)
    dx, dy = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05

    ax_map.set_xlim(x_min - dx, x_max + dx)
    ax_map.set_ylim(y_min - dy, y_max + dy)
    ax_map.set_xlabel("Longitude (°)", color='#94a3b8', fontsize=11)
    ax_map.set_ylabel("Latitude (°)", color='#94a3b8', fontsize=11)
    ax_map.tick_params(colors='#94a3b8')
    for spine in ax_map.spines.values():
        spine.set_color('#334155')

    lc_base = LineCollection(edge_segments, colors='#334155', linewidths=0.9, zorder=1)
    ax_map.add_collection(lc_base)

    all_nx = [G.nodes[n]['x'] for n in G.nodes]
    all_ny = [G.nodes[n]['y'] for n in G.nodes]
    ax_map.scatter(all_nx, all_ny, c='#475569', s=12, alpha=0.5, zorder=2, label='Unvisited Network Nodes')

    sx, sy = G.nodes[start_node]['x'], G.nodes[start_node]['y']
    gx, gy = G.nodes[goal_node]['x'], G.nodes[goal_node]['y']
    ax_map.scatter(sx, sy, c='#22c55e', s=180, edgecolors='#ffffff', linewidths=2.0, zorder=10, label='Origin (Start)')
    ax_map.scatter(gx, gy, c='#eab308', s=240, marker='*', edgecolors='#ffffff', linewidths=1.5, zorder=10, label='Destination (Goal)')
    ax_map.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc', loc='upper left', fontsize=9.5)

    visited_scatter = ax_map.scatter([], [], c=[], cmap='plasma', s=35, alpha=0.9, zorder=4, vmin=0, vmax=cost)
    frontier_scatter = ax_map.scatter([], [], c='#38bdf8', s=45, alpha=0.9, edgecolors='#ffffff', linewidths=0.8, zorder=5)
    curr_scatter = ax_map.scatter([], [], c='#f43f5e', s=140, edgecolors='#ffffff', linewidths=2.0, zorder=6)
    tree_lines = LineCollection([], colors='#64748b', linewidths=0.9, alpha=0.5, zorder=3)
    ax_map.add_collection(tree_lines)
    route_line, = ax_map.plot([], [], color='#ef4444', linewidth=4.5, zorder=8, solid_capstyle='round')

    status_text = ax_map.text(
        0.03, 0.05, "", transform=ax_map.transAxes,
        color='#f8fafc', fontsize=10.5, family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#0f172a', edgecolor='#38bdf8', alpha=0.9)
    )

    ax_cost.set_title("Cost-to-Come g(u) Progression (seconds)", color='#f8fafc', fontsize=12, fontweight='bold')
    ax_cost.set_xlim(0, max(len(history), 50))
    ax_cost.set_ylim(0, cost * 1.15)
    ax_cost.set_xlabel("Nodes Explored (Step k)", color='#94a3b8', fontsize=10)
    ax_cost.set_ylabel("Travel Time g(u) (s)", color='#94a3b8', fontsize=10)
    ax_cost.tick_params(colors='#94a3b8')
    ax_cost.grid(True, color='#334155', linestyle='--', alpha=0.6)
    for spine in ax_cost.spines.values():
        spine.set_color('#334155')

    cost_line, = ax_cost.plot([], [], color='#38bdf8', linewidth=2.0)
    cost_curr_point, = ax_cost.plot([], [], marker='o', color='#f43f5e', markersize=6)

    ax_front.set_title("Min-Heap Frontier Size (|Open Set|) vs Step", color='#f8fafc', fontsize=12, fontweight='bold')
    ax_front.set_xlim(0, max(len(history), 50))
    max_frontier = max(h['frontier_size'] for h in history)
    ax_front.set_ylim(0, max_frontier * 1.25)
    ax_front.set_xlabel("Nodes Explored (Step k)", color='#94a3b8', fontsize=10)
    ax_front.set_ylabel("Nodes in Min-Heap", color='#94a3b8', fontsize=10)
    ax_front.tick_params(colors='#94a3b8')
    ax_front.grid(True, color='#334155', linestyle='--', alpha=0.6)
    for spine in ax_front.spines.values():
        spine.set_color('#334155')

    front_line, = ax_front.plot([], [], color='#eab308', linewidth=2.0)
    front_curr_point, = ax_front.plot([], [], marker='o', color='#f43f5e', markersize=6)

    path_xs, path_ys = [], []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_dict = G[u][v]
        best_data = min(edge_dict.values(), key=lambda d: float(d.get('length', float('inf'))))
        if 'geometry' in best_data:
            geom = best_data['geometry']
            if isinstance(geom, str):
                geom = wkt.loads(geom)
            xs, ys = geom.xy
            path_xs.extend(list(xs))
            path_ys.extend(list(ys))
        else:
            path_xs.extend([G.nodes[u]['x'], G.nodes[v]['x']])
            path_ys.extend([G.nodes[u]['y'], G.nodes[v]['y']])

    all_steps = [h['step'] for h in history]
    all_costs = [h['current_g'] for h in history]
    all_fronts = [h['frontier_size'] for h in history]
    total_network_nodes = len(G.nodes)

    def update(frame_idx):
        data = frames_data[frame_idx]
        k = data['step']
        curr_u = data['current_node']
        vis_nodes = data['visited_nodes']
        front_nodes = data['frontier_nodes']
        
        if vis_nodes:
            vx = [G.nodes[n]['x'] for n in vis_nodes]
            vy = [G.nodes[n]['y'] for n in vis_nodes]
            visited_scatter.set_offsets(np.column_stack([vx, vy]))
            visited_scatter.set_array(np.linspace(0, data['current_g'], len(vis_nodes)))
        
        if front_nodes:
            fx = [G.nodes[n]['x'] for n in front_nodes]
            fy = [G.nodes[n]['y'] for n in front_nodes]
            frontier_scatter.set_offsets(np.column_stack([fx, fy]))
        else:
            frontier_scatter.set_offsets(np.empty((0, 2)))
            
        cx, cy = G.nodes[curr_u]['x'], G.nodes[curr_u]['y']
        curr_scatter.set_offsets([[cx, cy]])
        
        tree_segs = []
        for node, parent in data['parent_map'].items():
            if parent is not None and node in vis_nodes:
                tree_segs.append([(G.nodes[parent]['x'], G.nodes[parent]['y']), (G.nodes[node]['x'], G.nodes[node]['y'])])
        tree_lines.set_segments(tree_segs)
        
        unvisited_count = total_network_nodes - len(vis_nodes)
        unvisited_pct = (unvisited_count / total_network_nodes) * 100
        
        if data['status'] in ('GOAL_REACHED', 'COMPLETE'):
            status_str = "EARLY TERMINATION: GOAL REACHED!"
            status_text.set_text(
                f"STATUS: {status_str}\n"
                f"Explored Nodes:  {len(vis_nodes)} / {total_network_nodes} ({len(vis_nodes)/total_network_nodes*100:.1f}%)\n"
                f"Unvisited Nodes: {unvisited_count} ({unvisited_pct:.1f}% PRUNED)\n"
                f"Frontier in PQ:  {data['frontier_size']} nodes remaining\n"
                f"Optimal Time:    {cost:.2f} s"
            )
            status_text.set_bbox(dict(boxstyle='round,pad=0.6', facecolor='#064e3b', edgecolor='#22c55e', alpha=0.95))
            route_line.set_data(path_xs, path_ys)
        else:
            status_str = "SEARCHING (Dijkstra Min-Heap)"
            status_text.set_text(
                f"STATUS: {status_str}\n"
                f"Explored Nodes:  {len(vis_nodes)} / {total_network_nodes} ({len(vis_nodes)/total_network_nodes*100:.1f}%)\n"
                f"Unvisited Nodes: {unvisited_count} ({unvisited_pct:.1f}%)\n"
                f"Frontier in PQ:  {data['frontier_size']} nodes queued\n"
                f"Min Cost g(u):   {data['current_g']:.2f} s"
            )
            status_text.set_bbox(dict(boxstyle='round,pad=0.6', facecolor='#0f172a', edgecolor='#38bdf8', alpha=0.9))
            route_line.set_data([], [])
            
        curr_history_idx = min(k, len(all_steps) - 1)
        cost_line.set_data(all_steps[:curr_history_idx + 1], all_costs[:curr_history_idx + 1])
        cost_curr_point.set_data([all_steps[curr_history_idx]], [all_costs[curr_history_idx]])
        
        front_line.set_data(all_steps[:curr_history_idx + 1], all_fronts[:curr_history_idx + 1])
        front_curr_point.set_data([all_steps[curr_history_idx]], [all_fronts[curr_history_idx]])
        
        return [visited_scatter, frontier_scatter, curr_scatter, tree_lines, route_line, status_text, cost_line, cost_curr_point, front_line, front_curr_point]

    anim = animation.FuncAnimation(fig, update, frames=len(frames_data), interval=40, blit=False)

    print("Encoding MP4 video...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=2500, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    anim.save(output_mp4, writer=writer)
    print(f"MP4 successfully generated: {output_mp4}")

    print("Generating optimized GIF...")
    cmd = [
        "ffmpeg", "-y", "-i", output_mp4,
        "-vf", "fps=15,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        output_gif
    ]
    subprocess.run(cmd, check=True)
    print(f"GIF successfully generated: {output_gif}")
    plt.close(fig)


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "piedmont_road_network.graphml")
    G = ox.load_graphml(filepath=data_path)

    for u, v, k, data in G.edges(keys=True, data=True):
        data['length'] = float(data.get('length', 1.0))
        data['travel_time'] = float(data.get('travel_time', data['length'] / 13.8))

    nodes = list(G.nodes)
    start_node = nodes[10]
    
    # Scenario 1: Priority Queue Early-Termination (Mid-Range Target)
    mid_goal_node = 53041443
    render_scenario_video(
        G, start_node, mid_goal_node,
        "Priority Queue Early-Termination (Mid-Range Target)",
        os.path.join(base_dir, "dijkstra_pq_early_termination.mp4"),
        os.path.join(base_dir, "dijkstra_pq_early_termination.gif"),
        fps=25
    )
    
    # Scenario 2: Full-Network Traversal (Boundary Target)
    far_goal_node = [n for n in nodes if n != start_node and nx.has_path(G, start_node, n)][-15]
    render_scenario_video(
        G, start_node, far_goal_node,
        "Full-Network Traversal (Boundary Target)",
        os.path.join(base_dir, "dijkstra_search_evolution.mp4"),
        os.path.join(base_dir, "dijkstra_search_evolution.gif"),
        fps=25
    )
