# Session 02: Environmental Representation & Occupancy Grid Mapping

**Week:** 1 | **Session:** 2  
**Time Allocation:** 3 Hours Total
- ⏱️ **Theory (2 Hours):** HD Maps, Lanelet2, LiDAR Point Cloud Filtering, 2D Occupancy Grids & Log-Odds, Inverse Sensor Models, Costmaps & Frenet Frames
- 💻 **Practice (1 Hour):** 2D Occupancy Grid Builder & Costmap Inflation from LiDAR Point Clouds with `Polars`, `SciPy`, and `Plotly`

---

## 🎯 Learning Objectives
By the end of this session, students will be able to:
1. Explain the differences between High-Definition (HD) Vector Maps (Lanelet2 / OpenDRIVE) and Raster Occupancy Grids.
2. Implement LiDAR point cloud filtering pipelines: voxel downsampling, ground plane extraction, overhanging clipping, and dynamic obstacle removal.
3. Formulate the binary Bayesian filter and Log-Odds representation to avoid numerical floating-point underflow.
4. Formulate and implement the Inverse Sensor Model for range-bearing LiDAR beams (free-space ray-casting vs obstacle detection).
5. Generate an inflated, multi-layered **Costmap** using Euclidean Distance Transforms (EDT) for collision checking.
6. Formulate the Frenet-Serret road-aligned coordinate frame $(s, d)$ and compute bidirectional Frenet $\leftrightarrow$ Cartesian coordinate transformations.
7. Understand why decoupling motion into longitudinal progress $s(t)$ and lateral cross-track offset $d(s)$ simplifies motion planning.

---

## 📚 Session Outline

### 1. Theory (2 Hours)
- [01. HD Maps, Lanelet2 & Road Network Representations](theory/01_hd_maps_and_lanelets.md): Vector road representation, regulatory elements, and lane connectivity graphs.
- [02. LiDAR Point Cloud Preprocessing & Filtering](theory/02_lidar_preprocessing_and_filtering.md): Voxel downsampling, RANSAC ground removal, ceiling cropping, and dynamic vehicle filtering.
- [03. Occupancy Grids & Log-Odds Formulation](theory/03_occupancy_grids_and_log_odds.md): Mathematical derivation of the binary Bayes filter, Log-odds ratio, and recursive cell probability updates.
- [04. LiDAR Inverse Sensor Model (2D & 3D)](theory/04_lidar_inverse_sensor_model.md): 2D polar & 3D spherical beam geometry, volumetric voxel log-odds, and 3D Amanatides-Woo ray traversal.
- [05. Costmaps, Inflation Layers & Drivability Grids](theory/05_costmaps_and_obstacle_inflation.md): Converting probability grids into inflated obstacle costmaps via Distance Transforms.
- [06. Frenet Coordinate Frames](theory/06_frenet_coordinate_frames.md): Road-aligned curvilinear coordinates $(s, d)$, Cartesian $\leftrightarrow$ Frenet transformations, and longitudinal/lateral decoupling.

### 2. Practice Lab (1 Hour)
- Lab Exercise: [`practice/exercise_occupancy_grid.py`](practice/exercise_occupancy_grid.py)
- Reference Solution: [`practice/solution_occupancy_grid.py`](practice/solution_occupancy_grid.py)
- Run the lab:
  ```bash
  source ../../.venv/bin/activate
  python practice/exercise_occupancy_grid.py
  ```
