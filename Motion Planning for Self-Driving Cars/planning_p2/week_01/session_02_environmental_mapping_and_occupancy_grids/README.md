# Session 02: Environmental Representation & Occupancy Grid Mapping

**Week:** 1 | **Session:** 2  
**Time Allocation:** 3 Hours Total
- ⏱️ **Theory (2 Hours):** HD Maps, Lanelets, Frenet Coordinate Frames, 2D Occupancy Grids, Inverse Sensor Models & Bayesian Log-Odds
- 💻 **Practice (1 Hour):** 2D Occupancy Grid Builder from LiDAR Point Clouds with `Polars` and `Plotly`

---

## 🎯 Learning Objectives
By the end of this session, students will be able to:
1. Explain the differences between High-Definition (HD) Vector Maps (Lanelets) and Raster Occupancy Grids.
2. Formulate the Frenet-Serret road-aligned coordinate frame $(s, d)$ and compute bidirectional Frenet $\leftrightarrow$ Cartesian coordinate transformations.
3. Understand why decoupling motion into longitudinal progress $s(t)$ and lateral cross-track offset $d(s)$ simplifies planning.
4. Formulate the binary Bayesian filter and Log-Odds representation to avoid numerical floating-point underflow.
5. Formulate the Inverse Sensor Model for range-bearing LiDAR beams (free space clearing vs obstacle detection).
6. Process raw LiDAR polar range/bearing measurements into cartesian coordinate frames using `Polars`.
7. Construct and incrementally update an interactive 2D Occupancy Grid map rendered with `Plotly`.

---

## 📚 Session Outline

### 1. Theory (2 Hours)
- [01. HD Maps and Lanelets](theory/01_hd_maps_and_lanelets.md): Vector road representation, regulatory elements, and lane connectivity.
- [02. Occupancy Grids & Log-Odds Formulation](theory/02_occupancy_grids_and_log_odds.md): Binary Bayes filter, Log-odds ratio, and recursive cell probability updates.
- [03. LiDAR Inverse Sensor Model](theory/03_lidar_inverse_sensor_model.md): Beam sensor geometry, free-space dilation, and ray casting.
- [04. Frenet Coordinate Frames](theory/04_frenet_coordinate_frames.md): Road-aligned curvilinear coordinates $(s, d)$, Cartesian $\leftrightarrow$ Frenet transformations, and longitudinal/lateral decoupling.

### 2. Practice Lab (1 Hour)
- Lab Exercise: [`practice/exercise_occupancy_grid.py`](practice/exercise_occupancy_grid.py)
- Reference Solution: [`practice/solution_occupancy_grid.py`](practice/solution_occupancy_grid.py)
- Run the lab:
  ```bash
  source ../../.venv/bin/activate
  python practice/exercise_occupancy_grid.py
  ```
