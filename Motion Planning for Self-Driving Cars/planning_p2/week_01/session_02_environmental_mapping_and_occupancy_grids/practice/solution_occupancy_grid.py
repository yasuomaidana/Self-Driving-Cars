"""
Session 02 Reference Solution: 2D LiDAR Occupancy Grid Mapping & Costmap Inflation
----------------------------------------------------------------------------------
Runs the complete mapping and costmap inflation pipeline and outputs verification statistics.
"""

import numpy as np
import polars as pl
from exercise_occupancy_grid import (
    OccupancyGrid2D,
    generate_synthetic_lidar_dataset,
    visualize_mapping_and_costmap,
    get_default_obstacle_boxes,
    create_ground_truth_map
)

if __name__ == "__main__":
    print("=== Running Session 02 Reference Solution (Occlusion Discovery & Multi-Turn) ===")
    boxes = get_default_obstacle_boxes()
    df_lidar = generate_synthetic_lidar_dataset(boxes=boxes, num_scans=16)
    grid = OccupancyGrid2D(x_min=-5.0, x_max=28.0, y_min=-15.0, y_max=15.0, resolution=0.2)
    gt_map = create_ground_truth_map(grid, boxes)
    
    unique_scans = df_lidar["scan_id"].unique().sort().to_list()
    for s_id in unique_scans:
        scan_df = df_lidar.filter(pl.col("scan_id") == s_id)
        rx = scan_df["robot_x"][0]
        ry = scan_df["robot_y"][0]
        rtheta = scan_df["robot_theta"][0]
        ranges = scan_df["measured_range"].to_numpy()
        angles = scan_df["beam_angle"].to_numpy()
        
        grid.update_scan(rx, ry, rtheta, ranges, angles)
        
    p_map = grid.get_probability_map()
    costmap = grid.compute_costmap()
    
    print("\n--- Map Statistics ---")
    print(f"Occupancy Grid Dimensions: {p_map.shape} (Width: {grid.width}, Height: {grid.height})")
    print(f"Ground Truth Obstacle Cells: {np.sum(gt_map == 1)}")
    print(f"High Probability Obstacle Cells (P > 0.7): {np.sum(p_map > 0.7)}")
    print(f"Free Drivable Cells (P < 0.3): {np.sum(p_map < 0.3)}")
    print(f"Lethal Cost Cells (Cost == 254): {np.sum(costmap == 254)}")
    print(f"Inscribed Safety Inflation Cells (Cost == 253): {np.sum(costmap == 253)}")
    print(f"Decay Buffer Cells (1 <= Cost <= 252): {np.sum((costmap >= 1) & (costmap <= 252))}")
    print(f"Zero Cost Free Space Cells: {np.sum(costmap == 0)}")
    
    visualize_mapping_and_costmap(grid, "occupancy_and_costmap_solution.html", boxes=boxes)
