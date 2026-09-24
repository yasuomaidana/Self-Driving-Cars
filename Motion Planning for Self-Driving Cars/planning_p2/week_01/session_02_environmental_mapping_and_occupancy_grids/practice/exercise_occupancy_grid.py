"""
Session 02 Practice: 2D LiDAR Occupancy Grid Mapping & Costmap Inflation
-------------------------------------------------------------------------
In this lab, you will:
1. Build a 2D Log-Odds Occupancy Grid mapping pipeline from LiDAR point clouds.
2. Formulate and apply the Inverse Sensor Model with sub-grid ray-casting.
3. Maintain a Ground Truth obstacle matrix to verify map convergence and collision safety.
4. Demonstrate occlusion handling where an object hidden in the shadow of a foreground obstacle
   is discovered as the vehicle navigates around obstacles with 4 direction changes.
5. Generate an inflated Safety Costmap using Euclidean Distance Transforms (EDT)
   to bridge perception to downstream trajectory planners.

Tools used:
- Polars: For high-speed LiDAR beam telemetry handling
- SciPy: For Euclidean Distance Transform (distance_transform_edt)
- Plotly: For 2D interactive spatial visualization
"""

import math
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import distance_transform_edt


class OccupancyGrid2D:
    def __init__(self, x_min: float = -5.0, x_max: float = 28.0,
                 y_min: float = -15.0, y_max: float = 15.0,
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
        
        # Log-odds update values
        self.l_occ = math.log(0.85 / (1.0 - 0.85))    # ~ +1.73
        self.l_free = math.log(0.30 / (1.0 - 0.30))   # ~ -0.85
        self.l_min = -4.0
        self.l_max = 4.0
        self.thickness = 0.3  # meters

    def world_to_grid(self, x: float, y: float):
        """Converts continuous (x, y) coordinates to discrete (row, col) indices."""
        col = int((x - self.x_min) / self.resolution)
        row = int((y - self.y_min) / self.resolution)
        if 0 <= col < self.width and 0 <= row < self.height:
            return row, col
        return None

    def grid_to_world(self, row: int, col: int):
        """Converts discrete (row, col) indices back to continuous world (x, y) coordinates."""
        x = self.x_min + (col + 0.5) * self.resolution
        y = self.y_min + (row + 0.5) * self.resolution
        return x, y

    def update_scan(self, robot_x: float, robot_y: float, robot_theta: float,
                    ranges: np.ndarray, angles: np.ndarray, max_range: float = 15.0):
        """
        Update occupancy grid with one full LiDAR scan using Inverse Sensor Model.
        
        For each beam (angle, range):
        1. Raycast from robot pose to hit point at sub-cell steps (resolution * 0.4)
        2. Apply l_free for cells along the ray (r < measured_range - thickness/2)
        3. Apply l_occ at the measured obstacle hit cell
        4. Clamp log_odds into [self.l_min, self.l_max]
        """
        step_size = self.resolution * 0.4
        for angle, r in zip(angles, ranges):
            if r <= 0.1:
                continue
            
            beam_angle = robot_theta + angle
            is_valid_hit = (r < max_range)
            ray_len = min(r, max_range)
            
            # Step along the beam raycast with sub-grid resolution
            num_steps = int(ray_len / step_size)
            for s in range(num_steps):
                dist = s * step_size
                px = robot_x + dist * math.cos(beam_angle)
                py = robot_y + dist * math.sin(beam_angle)
                
                idx = self.world_to_grid(px, py)
                if idx is not None:
                    row, col = idx
                    # If strictly before hit point, decrease occupancy (free space)
                    if dist < (r - self.thickness / 2.0):
                        self.log_odds[row, col] += self.l_free
            
            # Update obstacle hit cell
            if is_valid_hit:
                hit_x = robot_x + r * math.cos(beam_angle)
                hit_y = robot_y + r * math.sin(beam_angle)
                idx = self.world_to_grid(hit_x, hit_y)
                if idx is not None:
                    row, col = idx
                    self.log_odds[row, col] += self.l_occ
                    
        # Clamp log-odds to allow map recovery
        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)

    def get_probability_map(self) -> np.ndarray:
        """Converts log-odds grid into probability values [0.0, 1.0]."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def compute_costmap(self, prob_threshold: float = 0.65,
                        r_inscribed: float = 1.0,
                        r_inflation: float = 2.5,
                        decay_factor: float = 2.0) -> np.ndarray:
        """
        Computes an inflated safety costmap from the occupancy grid.
        
        Cost Values:
        - 254: Lethal Obstacle (P >= prob_threshold)
        - 253: Inscribed Vehicle Radius (Guaranteed collision)
        - 1..252: Exponential Safety Decay Zone
        - 0: Free / Safe Drivable Space
        """
        prob_map = self.get_probability_map()
        lethal_mask = prob_map >= prob_threshold
        
        # Euclidean distance from every cell to nearest lethal obstacle in meters
        distance_grid = distance_transform_edt(~lethal_mask) * self.resolution
        
        costmap = np.zeros_like(prob_map, dtype=np.uint8)
        
        # 1. Lethal zone
        costmap[lethal_mask] = 254
        
        # 2. Inscribed vehicle zone
        inscribed_mask = (~lethal_mask) & (distance_grid <= r_inscribed)
        costmap[inscribed_mask] = 253
        
        # 3. Inflation decay zone
        decay_mask = (~lethal_mask) & (distance_grid > r_inscribed) & (distance_grid <= r_inflation)
        dist_diff = distance_grid[decay_mask] - r_inscribed
        costmap[decay_mask] = np.uint8(253 * np.exp(-decay_factor * dist_diff))
        
        return costmap


def create_ground_truth_map(grid: OccupancyGrid2D, boxes: list) -> np.ndarray:
    """
    Creates a discrete binary Ground Truth matrix (height x width)
    where obstacle cells inside boxes are 1 and free space cells are 0.
    """
    gt_map = np.zeros((grid.height, grid.width), dtype=np.uint8)
    for (x0, x1, y0, y1) in boxes:
        col_min = max(0, int((x0 - grid.x_min) / grid.resolution))
        col_max = min(grid.width, int((x1 - grid.x_min) / grid.resolution) + 1)
        row_min = max(0, int((y0 - grid.y_min) / grid.resolution))
        row_max = min(grid.height, int((y1 - grid.y_min) / grid.resolution) + 1)
        gt_map[row_min:row_max, col_min:col_max] = 1
    return gt_map


def get_default_obstacle_boxes():
    """
    Standard obstacle box definitions [(x_min, x_max, y_min, y_max), ...]
    Demonstrates occlusion: Obstacle 2 is directly behind Foreground Obstacle 1 from the initial pose.
    """
    return [
        (7.0, 11.0, -2.5, 2.5),     # Foreground obstacle (blocks central path)
        (16.0, 19.5, -2.0, 2.0),    # Occluded obstacle (shadowed until vehicle navigates around)
        (6.0, 10.0, 6.0, 9.5),      # North obstacle
        (6.0, 10.0, -9.5, -6.0),    # South obstacle
        (21.0, 25.0, 2.5, 6.5),     # Far East obstacle
    ]


def generate_synthetic_lidar_dataset(boxes: list = None, num_scans: int = 16) -> pl.DataFrame:
    """
    Generates synthetic LiDAR telemetry data using Polars DataFrame.
    Vehicle drives towards obstacles and changes direction 4 times:
    - Approaching central obstacle -> Turn 1 (North)
    - Reaching North corridor -> Turn 2 (East) past the foreground obstacle
    - Reaching East corridor -> Turn 3 (South) discovering the occluded obstacle
    - Reaching South corridor -> Turn 4 (West) completing the survey loop
    """
    if boxes is None:
        boxes = get_default_obstacle_boxes()
        
    # Multi-turn exploration trajectory (rx, ry, rtheta)
    waypoints = [
        # 1. Drive East towards Foreground Obstacle (0 deg)
        (0.0, 0.0, 0.0),
        (2.5, 0.0, 0.0),
        (5.0, 0.0, 0.0),            # Turn 1: Steer North (+90 deg)
        
        # 2. Drive North along obstacle face
        (5.0, 2.2, math.pi / 2.0),
        (5.0, 4.5, math.pi / 2.0),  # Turn 2: Steer East (0 deg)
        
        # 3. Drive East past Foreground Obstacle (Occluded obstacle starts revealing from North)
        (8.0, 4.5, 0.0),
        (11.0, 4.5, 0.0),
        (13.5, 4.5, 0.0),           # Turn 3: Steer South (-90 deg)
        
        # 4. Drive South between Foreground and Occluded Obstacles
        (13.5, 2.0, -math.pi / 2.0),
        (13.5, 0.0, -math.pi / 2.0), # Directly between both obstacles (fully unoccluded)
        (13.5, -2.0, -math.pi / 2.0),
        (13.5, -4.5, -math.pi / 2.0),# Turn 4: Steer West (180 deg)
        
        # 5. Drive West along southern corridor
        (11.0, -4.5, math.pi),
        (8.0, -4.5, math.pi),
        (5.0, -4.5, math.pi),
        (2.5, -4.5, math.pi),
    ]
    
    waypoints = waypoints[:num_scans]
    records = []
    num_beams = 180
    angles = np.linspace(-math.pi, math.pi, num_beams)
    
    for scan_id, (rx, ry, rtheta) in enumerate(waypoints):
        # Verify ego vehicle is strictly outside all obstacles
        for x0, x1, y0, y1 in boxes:
            assert not (x0 <= rx <= x1 and y0 <= ry <= y1), (
                f"Ego vehicle pose ({rx}, {ry}) collided with obstacle box [{x0}, {x1}, {y0}, {y1}]"
            )
        
        for a in angles:
            beam_dir = rtheta + a
            measured_r = 15.0  # default max range
            for x0, x1, y0, y1 in boxes:
                for step_r in np.linspace(0.1, 15.0, 300):
                    px = rx + step_r * math.cos(beam_dir)
                    py = ry + step_r * math.sin(beam_dir)
                    if x0 <= px <= x1 and y0 <= py <= y1:
                        measured_r = min(measured_r, step_r)
                        break
                        
            # Gaussian sensor noise
            if measured_r < 15.0:
                measured_r += float(np.random.normal(0, 0.03))
                
            records.append({
                "scan_id": scan_id,
                "robot_x": rx,
                "robot_y": ry,
                "robot_theta": rtheta,
                "beam_angle": float(a),
                "measured_range": float(measured_r)
            })
            
    return pl.DataFrame(records)


def visualize_mapping_and_costmap(grid: OccupancyGrid2D, output_html: str = "occupancy_and_costmap.html", boxes: list = None):
    """Visualizes ground truth, probability grid, and inflated costmap using Plotly subplots."""
    if boxes is None:
        boxes = get_default_obstacle_boxes()
        
    prob_map = grid.get_probability_map()
    costmap = grid.compute_costmap()
    gt_map = create_ground_truth_map(grid, boxes)
    
    x_coords = np.linspace(grid.x_min, grid.x_max, grid.width)
    y_coords = np.linspace(grid.y_min, grid.y_max, grid.height)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("1. True Map (Ground Truth Matrix)", "2. Estimated Occupancy Grid P(m)", "3. Inflated Costmap (Planning)")
    )
    
    # 1. Ground truth map
    fig.add_trace(
        go.Heatmap(
            z=gt_map,
            x=x_coords,
            y=y_coords,
            colorscale="Greys",
            zmin=0,
            zmax=1,
            colorbar=dict(title="True Map", x=0.28)
        ),
        row=1, col=1
    )
    
    # 2. Probability Heatmap
    fig.add_trace(
        go.Heatmap(
            z=prob_map,
            x=x_coords,
            y=y_coords,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="P(Occ)", x=0.62)
        ),
        row=1, col=2
    )
    
    # 3. Costmap Heatmap
    fig.add_trace(
        go.Heatmap(
            z=costmap,
            x=x_coords,
            y=y_coords,
            colorscale="Hot_r",
            zmin=0,
            zmax=254,
            colorbar=dict(title="Cost", x=1.0)
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        title="Session 02: 2D LiDAR Occupancy Grid & Costmap Inflation Pipeline (Multi-Turn Exploration)",
        width=1400,
        height=550,
        showlegend=False
    )
    fig.update_xaxes(title_text="Global X (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Global Y (m)")
    
    fig.write_html(output_html)
    print(f"Interactive mapping & costmap visualization saved to {output_html}")


if __name__ == "__main__":
    print("Generating synthetic LiDAR point cloud dataset with Polars (4 direction changes)...")
    boxes = get_default_obstacle_boxes()
    df_lidar = generate_synthetic_lidar_dataset(boxes=boxes, num_scans=16)
    print("Telemetry shape:", df_lidar.shape)
    
    grid = OccupancyGrid2D(x_min=-5.0, x_max=28.0, y_min=-15.0, y_max=15.0, resolution=0.2)
    
    unique_scans = df_lidar["scan_id"].unique().sort().to_list()
    for s_id in unique_scans:
        scan_df = df_lidar.filter(pl.col("scan_id") == s_id)
        rx = scan_df["robot_x"][0]
        ry = scan_df["robot_y"][0]
        rtheta = scan_df["robot_theta"][0]
        ranges = scan_df["measured_range"].to_numpy()
        angles = scan_df["beam_angle"].to_numpy()
        
        grid.update_scan(rx, ry, rtheta, ranges, angles)
        print(f"Processed Scan {s_id:02d} at Pose ({rx:.1f}, {ry:.1f}, theta={math.degrees(rtheta):.0f} deg)")
        
    visualize_mapping_and_costmap(grid, "occupancy_and_costmap.html", boxes=boxes)
