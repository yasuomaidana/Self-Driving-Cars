"""
Session 02 Practice: 2D LiDAR Occupancy Grid Mapping
---------------------------------------------------
In this lab, you will build a 2D Log-Odds Occupancy Grid mapping pipeline that processes
simulated 2D LiDAR point clouds and produces a probabilistic occupancy map.

Tools used:
- Polars: For high-speed LiDAR beam telemetry handling
- Plotly: For 2D spatial heatmap rendering of the generated map
"""

import math
import numpy as np
import polars as pl
import plotly.graph_objects as go


class OccupancyGrid2D:
    def __init__(self, x_min: float = -20.0, x_max: float = 20.0,
                 y_min: float = -20.0, y_max: float = 20.0,
                 resolution: float = 0.2):
        """
        resolution: Grid cell size (meters per cell)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.resolution = resolution
        
        self.width = int((x_max - x_min) / resolution)
        self.height = int((y_max - y_min) / resolution)
        
        # Log-odds probability array initialized to 0 (P=0.5 prior)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Log-odds parameters
        self.l_occ = math.log(0.85 / (1.0 - 0.85))    # ~ +1.73
        self.l_free = math.log(0.30 / (1.0 - 0.30))   # ~ -0.85
        self.l_min = -4.0
        self.l_max = 4.0
        self.thickness = 0.3 # meters

    def world_to_grid(self, x: float, y: float) -> tuple:
        """Converts continuous (x, y) coordinates to discrete (row, col) indices."""
        col = int((x - self.x_min) / self.resolution)
        row = int((y - self.y_min) / self.resolution)
        if 0 <= col < self.width and 0 <= row < self.height:
            return row, col
        return None

    def update_scan(self, robot_x: float, robot_y: float, robot_theta: float,
                    ranges: np.ndarray, angles: np.ndarray, max_range: float = 15.0):
        """
        Update occupancy grid with one full LiDAR scan using Inverse Sensor Model.
        
        TODO (Student):
        For each beam (angle, range):
        1. Raycast from robot pose to hit point at steps of self.resolution/2
        2. Apply l_free for cells along the ray (r < measured_range - thickness/2)
        3. Apply l_occ at the measured obstacle hit cell
        4. Clamp log_odds into [self.l_min, self.l_max]
        """
        for angle, r in zip(angles, ranges):
            if r <= 0.1:
                continue
            
            beam_angle = robot_theta + angle
            is_valid_hit = (r < max_range)
            ray_len = min(r, max_range)
            
            # Step along the beam raycast
            num_steps = int(ray_len / (self.resolution * 0.5))
            for s in range(num_steps):
                dist = s * (self.resolution * 0.5)
                px = robot_x + dist * math.cos(beam_angle)
                py = robot_y + dist * math.sin(beam_angle)
                
                idx = self.world_to_grid(px, py)
                if idx is not None:
                    row, col = idx
                    # If strictly before hit, decrease occupancy (free space)
                    if dist < (r - self.thickness / 2.0):
                        self.log_odds[row, col] += self.l_free
            
            # Update obstacle cell
            if is_valid_hit:
                hit_x = robot_x + r * math.cos(beam_angle)
                hit_y = robot_y + r * math.sin(beam_angle)
                idx = self.world_to_grid(hit_x, hit_y)
                if idx is not None:
                    row, col = idx
                    self.log_odds[row, col] += self.l_occ
                    
        # Clamp log odds
        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)

    def get_probability_map(self) -> np.ndarray:
        """Converts log-odds grid into probability values [0.0, 1.0]."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))


def generate_synthetic_lidar_dataset(num_scans: int = 5) -> pl.DataFrame:
    """
    Generates synthetic LiDAR telemetry data using Polars DataFrame.
    Vehicle drives forward along X-axis observing rectangular obstacles.
    """
    records = []
    num_beams = 180
    angles = np.linspace(-math.pi, math.pi, num_beams)
    
    # Synthetic obstacle boxes [(x_min, x_max, y_min, y_max), ...]
    boxes = [
        (5.0, 8.0, 2.0, 5.0),
        (8.0, 11.0, -6.0, -3.0),
        (12.0, 14.0, -1.0, 3.0),
    ]
    
    for scan_id in range(num_scans):
        rx = scan_id * 2.5
        ry = 0.0
        rtheta = 0.0
        
        for a in angles:
            beam_dir = rtheta + a
            # Compute intersection with boxes
            measured_r = 15.0 # default max range
            for x0, x1, y0, y1 in boxes:
                for step_r in np.linspace(0.5, 15.0, 150):
                    px = rx + step_r * math.cos(beam_dir)
                    py = ry + step_r * math.sin(beam_dir)
                    if x0 <= px <= x1 and y0 <= py <= y1:
                        measured_r = min(measured_r, step_r)
                        break
                        
            # Add Gaussian noise
            if measured_r < 15.0:
                measured_r += float(np.random.normal(0, 0.05))
                
            records.append({
                "scan_id": scan_id,
                "robot_x": rx,
                "robot_y": ry,
                "robot_theta": rtheta,
                "beam_angle": float(a),
                "measured_range": float(measured_r)
            })
            
    return pl.DataFrame(records)


def visualize_occupancy_grid(grid: OccupancyGrid2D, output_html: str = "occupancy_grid.html"):
    """Visualizes the 2D occupancy probability heatmap using Plotly."""
    prob_map = grid.get_probability_map()
    
    x_coords = np.linspace(grid.x_min, grid.x_max, grid.width)
    y_coords = np.linspace(grid.y_min, grid.y_max, grid.height)
    
    fig = go.Figure(data=go.Heatmap(
        z=prob_map,
        x=x_coords,
        y=y_coords,
        colorscale="Viridis",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="P(Occupied)")
    ))
    
    fig.update_layout(
        title="2D LiDAR Occupancy Grid Probability Map",
        xaxis_title="Global X (meters)",
        yaxis_title="Global Y (meters)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=800,
        height=800
    )
    
    fig.write_html(output_html)
    print(f"Occupancy grid map saved to {output_html}")


if __name__ == "__main__":
    print("Generating synthetic LiDAR pointcloud dataset with Polars...")
    df_lidar = generate_synthetic_lidar_dataset(num_scans=6)
    print("Dataset shape:", df_lidar.shape)
    
    grid = OccupancyGrid2D(x_min=-5.0, x_max=25.0, y_min=-15.0, y_max=15.0, resolution=0.2)
    
    # Process scan-by-scan
    unique_scans = df_lidar["scan_id"].unique().sort().to_list()
    for s_id in unique_scans:
        scan_df = df_lidar.filter(pl.col("scan_id") == s_id)
        rx = scan_df["robot_x"][0]
        ry = scan_df["robot_y"][0]
        rtheta = scan_df["robot_theta"][0]
        ranges = scan_df["measured_range"].to_numpy()
        angles = scan_df["beam_angle"].to_numpy()
        
        grid.update_scan(rx, ry, rtheta, ranges, angles)
        print(f"Processed Scan {s_id} at Pose ({rx:.1f}, {ry:.1f})")
        
    visualize_occupancy_grid(grid, "occupancy_grid_plotly.html")
