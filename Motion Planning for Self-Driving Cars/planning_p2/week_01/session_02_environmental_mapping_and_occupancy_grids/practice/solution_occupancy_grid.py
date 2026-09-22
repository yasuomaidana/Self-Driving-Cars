"""
Session 02 Solution: 2D LiDAR Occupancy Grid Mapping
"""

import math
import numpy as np
import polars as pl
import plotly.graph_objects as go
from exercise_occupancy_grid import OccupancyGrid2D, generate_synthetic_lidar_dataset, visualize_occupancy_grid

if __name__ == "__main__":
    df_lidar = generate_synthetic_lidar_dataset(num_scans=6)
    grid = OccupancyGrid2D(x_min=-5.0, x_max=25.0, y_min=-15.0, y_max=15.0, resolution=0.2)
    
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
    print("Occupancy grid successfully generated. Grid shape:", p_map.shape)
    print("Occupied cells (P > 0.7):", np.sum(p_map > 0.7))
    print("Free cells (P < 0.3):", np.sum(p_map < 0.3))
