"""
Map Evolution Video & Interactive Animation Generator
Week 01 - Session 02: 2D LiDAR Occupancy Grid & Costmap Inflation
----------------------------------------------------------------
Generates:
1. Interactive HTML animation (Plotly frames + Play/Pause button + Slider)
2. Standalone MP4 video (Matplotlib + ffmpeg)
3. Standalone Animated GIF (Matplotlib + Pillow/ffmpeg)
"""

import math
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from exercise_occupancy_grid import (
    OccupancyGrid2D,
    generate_synthetic_lidar_dataset,
    create_ground_truth_map,
    get_default_obstacle_boxes
)


def generate_map_evolution_frames(num_scans: int = 16):
    """
    Simulates vehicle moving through the environment and saves
    snapshots of grid, probability map, costmap, robot poses, and rays at each step.
    """
    boxes = get_default_obstacle_boxes()
    df_lidar = generate_synthetic_lidar_dataset(boxes=boxes, num_scans=num_scans)
    grid = OccupancyGrid2D(x_min=-5.0, x_max=28.0, y_min=-15.0, y_max=15.0, resolution=0.2)
    gt_map = create_ground_truth_map(grid, boxes)
    
    unique_scans = df_lidar["scan_id"].unique().sort().to_list()
    history = []
    
    # Store initial unmapped state (t = -1 or prior)
    history.append({
        "scan_id": -1,
        "robot_x": 0.0,
        "robot_y": 0.0,
        "robot_theta": 0.0,
        "prob_map": grid.get_probability_map().copy(),
        "costmap": grid.compute_costmap().copy(),
        "ranges": np.array([]),
        "angles": np.array([]),
        "hit_points": (np.array([]), np.array([]))
    })
    
    for s_id in unique_scans:
        scan_df = df_lidar.filter(pl.col("scan_id") == s_id)
        rx = scan_df["robot_x"][0]
        ry = scan_df["robot_y"][0]
        rtheta = scan_df["robot_theta"][0]
        ranges = scan_df["measured_range"].to_numpy()
        angles = scan_df["beam_angle"].to_numpy()
        
        # Calculate hit coordinates
        valid = ranges < 14.8
        beam_angles = rtheta + angles[valid]
        hit_x = rx + ranges[valid] * np.cos(beam_angles)
        hit_y = ry + ranges[valid] * np.sin(beam_angles)
        
        # Update map
        grid.update_scan(rx, ry, rtheta, ranges, angles)
        
        history.append({
            "scan_id": s_id,
            "robot_x": rx,
            "robot_y": ry,
            "robot_theta": rtheta,
            "prob_map": grid.get_probability_map().copy(),
            "costmap": grid.compute_costmap().copy(),
            "ranges": ranges,
            "angles": angles,
            "hit_points": (hit_x, hit_y)
        })
        
    return grid, history, df_lidar, gt_map, boxes


def create_plotly_animated_html(grid: OccupancyGrid2D, history: list, gt_map: np.ndarray, boxes: list, output_html: str = "map_evolution_interactive.html"):
    """
    Creates an interactive animated HTML using Plotly with play/pause controls and a time slider.
    """
    x_coords = np.linspace(grid.x_min, grid.x_max, grid.width)
    y_coords = np.linspace(grid.y_min, grid.y_max, grid.height)
    
    # Base figure with 3 subplots: True Map, Occupancy Grid, Inflated Costmap
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("1. True Ground Truth Map",
                        "2. Occupancy Probability Grid $P(m)$ & LiDAR Beams",
                        "3. Dynamic Inflated Safety Costmap"),
        horizontal_spacing=0.08
    )
    
    initial = history[0]
    
    # Trace 0: Ground Truth Map (static background reference)
    fig.add_trace(
        go.Heatmap(
            z=gt_map,
            x=x_coords,
            y=y_coords,
            colorscale="Greys",
            zmin=0,
            zmax=1,
            colorbar=dict(title="True Map", x=0.27),
            name="Ground Truth"
        ),
        row=1, col=1
    )
    # Robot on True map
    fig.add_trace(
        go.Scatter(
            x=[initial["robot_x"]],
            y=[initial["robot_y"]],
            mode="markers+text",
            marker=dict(size=12, color="blue", symbol="triangle-right"),
            text=["Ego"],
            textposition="top center",
            name="Ego (True Map)"
        ),
        row=1, col=1
    )
    
    # Trace 2: Probability heatmap
    fig.add_trace(
        go.Heatmap(
            z=initial["prob_map"],
            x=x_coords,
            y=y_coords,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="P(Occ)", x=0.62),
            name="P(Occ)"
        ),
        row=1, col=2
    )
    # Trace 3: LiDAR hit points on Prob map
    fig.add_trace(
        go.Scatter(
            x=initial["hit_points"][0],
            y=initial["hit_points"][1],
            mode="markers",
            marker=dict(size=4, color="crimson", symbol="circle"),
            name="LiDAR Hits"
        ),
        row=1, col=2
    )
    # Trace 4: Robot pose on Prob map
    fig.add_trace(
        go.Scatter(
            x=[initial["robot_x"]],
            y=[initial["robot_y"]],
            mode="markers+text",
            marker=dict(size=12, color="magenta", symbol="triangle-right"),
            text=["Ego"],
            textposition="top center",
            name="Ego Vehicle"
        ),
        row=1, col=2
    )
    
    # Trace 5: Costmap heatmap
    fig.add_trace(
        go.Heatmap(
            z=initial["costmap"],
            x=x_coords,
            y=y_coords,
            colorscale="Hot_r",
            zmin=0,
            zmax=254,
            colorbar=dict(title="Cost", x=1.0),
            name="Costmap"
        ),
        row=1, col=3
    )
    # Trace 6: Robot pose on Costmap
    fig.add_trace(
        go.Scatter(
            x=[initial["robot_x"]],
            y=[initial["robot_y"]],
            mode="markers",
            marker=dict(size=12, color="cyan", symbol="triangle-right"),
            name="Vehicle (Costmap)"
        ),
        row=1, col=3
    )
    
    # Build animation frames
    frames = []
    for idx, h in enumerate(history):
        scan_id = h["scan_id"]
        frame_name = f"Scan {scan_id}" if scan_id >= 0 else "Initial Prior"
        
        frame = go.Frame(
            name=frame_name,
            data=[
                # Ground truth map (fixed)
                go.Heatmap(z=gt_map, x=x_coords, y=y_coords),
                # Robot on GT
                go.Scatter(x=[h["robot_x"]], y=[h["robot_y"]]),
                # Prob heatmap
                go.Heatmap(z=h["prob_map"], x=x_coords, y=y_coords),
                # LiDAR hits
                go.Scatter(x=h["hit_points"][0], y=h["hit_points"][1]),
                # Robot pose on prob map
                go.Scatter(x=[h["robot_x"]], y=[h["robot_y"]]),
                # Costmap heatmap
                go.Heatmap(z=h["costmap"], x=x_coords, y=y_coords),
                # Robot on costmap
                go.Scatter(x=[h["robot_x"]], y=[h["robot_y"]]),
            ]
        )
        frames.append(frame)
        
    fig.frames = frames
    
    # Sliders and Play/Pause buttons
    slider_steps = []
    for f in frames:
        step = dict(
            method="animate",
            label=f.name,
            args=[
                [f.name],
                dict(
                    mode="immediate",
                    frame=dict(duration=500, redraw=True),
                    transition=dict(duration=150)
                )
            ]
        )
        slider_steps.append(step)
        
    fig.update_layout(
        title={
            "text": "<b>LiDAR Occupancy Grid & Costmap Evolution Over Time</b><br><sup>Multi-turn reactive exploration (4 direction changes) resolving foreground occlusion of hidden obstacle</sup>",
            "y": 0.96
        },
        width=1500,
        height=620,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.08,
                y=-0.15,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play Video",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=550, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                                transition=dict(duration=150)
                            )
                        ]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0)
                            )
                        ]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.10,
                y=-0.15,
                len=0.88,
                xanchor="left",
                yanchor="top",
                steps=slider_steps,
                currentvalue=dict(prefix="Telemetry Step: ", visible=True, xanchor="right")
            )
        ]
    )
    
    fig.update_xaxes(title_text="Global X (m)", scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="Global Y (m)", row=1, col=1)
    fig.update_xaxes(title_text="Global X (m)", scaleanchor="y2", scaleratio=1, row=1, col=2)
    fig.update_yaxes(title_text="Global Y (m)", row=1, col=2)
    fig.update_xaxes(title_text="Global X (m)", scaleanchor="y3", scaleratio=1, row=1, col=3)
    fig.update_yaxes(title_text="Global Y (m)", row=1, col=3)
    
    fig.write_html(output_html)
    print(f"Plotly animated HTML video successfully generated at: {output_html}")


def create_mp4_and_gif_video(grid: OccupancyGrid2D, history: list, gt_map: np.ndarray, boxes: list, output_mp4: str = "map_evolution.mp4", output_gif: str = "map_evolution.gif"):
    """
    Renders high-quality MP4 video and animated GIF using Matplotlib animation.
    """
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6), dpi=120)
    fig.patch.set_facecolor('#1e1e2f')
    
    extent = [grid.x_min, grid.x_max, grid.y_min, grid.y_max]
    
    for ax in (ax0, ax1, ax2):
        ax.set_facecolor('#12121c')
        ax.set_xlim(grid.x_min, grid.x_max)
        ax.set_ylim(grid.y_min, grid.y_max)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#555577')
            
    ax0.set_xlabel("Global X [m]", color="white", fontsize=10)
    ax0.set_ylabel("Global Y [m]", color="white", fontsize=10)
    ax1.set_xlabel("Global X [m]", color="white", fontsize=10)
    ax1.set_ylabel("Global Y [m]", color="white", fontsize=10)
    ax2.set_xlabel("Global X [m]", color="white", fontsize=10)
    ax2.set_ylabel("Global Y [m]", color="white", fontsize=10)
    
    im0 = ax0.imshow(1 - gt_map, origin='lower', extent=extent, cmap='gray', vmin=0.0, vmax=1.0)
    ax0.set_title("1. Ground Truth Obstacle Map", color='white', fontsize=11)
    
    init_prob = history[0]["prob_map"]
    init_cost = history[0]["costmap"]
    
    im1 = ax1.imshow(init_prob, origin='lower', extent=extent, cmap='viridis', vmin=0.0, vmax=1.0, animated=True)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors='white')
    cbar1.set_label("P(Occupied)", color='white', fontsize=9)
    
    im2 = ax2.imshow(init_cost, origin='lower', extent=extent, cmap='hot_r', vmin=0, vmax=254, animated=True)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors='white')
    cbar2.set_label("Planning Cost [0-254]", color='white', fontsize=9)
    
    # Scatter objects
    hits_scatter = ax1.scatter([], [], c='cyan', s=10, alpha=0.8, edgecolors='none', label='LiDAR Hits')
    
    robot_marker0, = ax0.plot([], [], marker='^', color='red', markersize=10, label='Ego Vehicle')
    traj_line0, = ax0.plot([], [], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    robot_marker1, = ax1.plot([], [], marker='^', color='red', markersize=10, label='Ego Vehicle')
    traj_line1, = ax1.plot([], [], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    robot_marker2, = ax2.plot([], [], marker='^', color='cyan', markersize=10, label='Ego Vehicle')
    traj_line2, = ax2.plot([], [], color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Ground truth box outlines
    for (x0, x1, y0, y1) in boxes:
        for ax in (ax0, ax1, ax2):
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='lime', linestyle=':', linewidth=1.2, alpha=0.7)
            ax.add_patch(rect)
        
    ax0.legend(loc='upper right', facecolor='#2a2a3e', edgecolor='#555577', labelcolor='white', fontsize=8)
    ax1.legend(loc='upper right', facecolor='#2a2a3e', edgecolor='#555577', labelcolor='white', fontsize=8)
    ax2.legend(loc='upper right', facecolor='#2a2a3e', edgecolor='#555577', labelcolor='white', fontsize=8)
    
    title_text = fig.suptitle("", color='white', fontsize=13, weight='bold', y=0.98)
    
    def update(frame_idx):
        h = history[frame_idx]
        scan_id = h["scan_id"]
        rx, ry, rtheta = h["robot_x"], h["robot_y"], h["robot_theta"]
        
        im1.set_data(h["prob_map"])
        im2.set_data(h["costmap"])
        
        hit_x, hit_y = h["hit_points"]
        hits_scatter.set_offsets(np.c_[hit_x, hit_y] if len(hit_x) > 0 else np.empty((0, 2)))
        
        cur_traj_x = [history[i]["robot_x"] for i in range(frame_idx + 1) if history[i]["scan_id"] >= 0]
        cur_traj_y = [history[i]["robot_y"] for i in range(frame_idx + 1) if history[i]["scan_id"] >= 0]
        
        if scan_id >= 0:
            for rm in (robot_marker0, robot_marker1, robot_marker2):
                rm.set_data([rx], [ry])
            for tl in (traj_line0, traj_line1, traj_line2):
                tl.set_data(cur_traj_x, cur_traj_y)
            status = f"Scan #{scan_id:02d} | Pose: ({rx:.1f} m, {ry:.1f} m, {math.degrees(rtheta):.0f}°) | Hits: {len(hit_x)}"
        else:
            for rm in (robot_marker0, robot_marker1, robot_marker2):
                rm.set_data([], [])
            for tl in (traj_line0, traj_line1, traj_line2):
                tl.set_data([], [])
            status = "Initial Prior State: Log-Odds = 0.0, P(Occ) = 0.50 (Unexplored)"
            
        ax1.set_title(f"2. Occupancy Probability $P(m)$\n{status}", color='white', fontsize=10)
        ax2.set_title("3. Inflated Costmap (Safety Buffer: 2.5m)\nLethal=254, Inscribed=253, Decay=1..252", color='white', fontsize=10)
        title_text.set_text("Autonomous Vehicle Environmental Mapping Evolution (Occlusion Discovery)")
        
        return im1, im2, hits_scatter, robot_marker0, robot_marker1, robot_marker2, traj_line0, traj_line1, traj_line2
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(history),
        interval=700, blit=False, repeat=True
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save MP4
    try:
        writermp4 = animation.FFMpegWriter(fps=1.5, bitrate=2000)
        ani.save(output_mp4, writer=writermp4)
        print(f"MP4 Video saved to: {output_mp4}")
    except Exception as e:
        print(f"Could not export MP4: {e}")
        
    # Save GIF
    try:
        writergif = animation.PillowWriter(fps=1.5)
        ani.save(output_gif, writer=writergif)
        print(f"Animated GIF saved to: {output_gif}")
    except Exception as e:
        print(f"Could not export GIF: {e}")
        
    plt.close(fig)


if __name__ == "__main__":
    import os
    target_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Generating simulation trajectory and map evolution...")
    grid, history, df_lidar, gt_map, boxes = generate_map_evolution_frames(num_scans=16)
    
    html_path = os.path.join(target_dir, "map_evolution_interactive.html")
    mp4_path = os.path.join(target_dir, "map_evolution.mp4")
    gif_path = os.path.join(target_dir, "map_evolution.gif")
    
    print("1. Creating interactive Plotly video/animation HTML...")
    create_plotly_animated_html(grid, history, gt_map, boxes, html_path)
    
    print("2. Creating MP4 video and animated GIF with Matplotlib & ffmpeg...")
    create_mp4_and_gif_video(grid, history, gt_map, boxes, mp4_path, gif_path)
    
    print("\n✅ All video & animation artifacts successfully regenerated!")
