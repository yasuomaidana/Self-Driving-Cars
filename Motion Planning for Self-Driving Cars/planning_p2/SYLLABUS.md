# Course Syllabus: Motion Planning for Autonomous Vehicles (Planning P2)

**Term:** Fall / Spring  
**Duration:** 5 Weeks (10 Sessions)  
**Format:** 2 Sessions per Week, 3 Hours per Session (2 Hours Theory + 1 Hour Hands-On Practice)  
**Total Hours:** 30 Hours (20h Theory, 10h Lab/Coding)  
**Prerequisites:** Python 3.10+, Linear Algebra, Calculus, Basic Probability, Data Structures (Graphs, Trees, Priority Queues).

---

## 🎯 Course Description & Objectives
This course covers the theoretical foundations, algorithmic methods, and software engineering implementations of modern motion planning systems for Level 3–4 self-driving vehicles. 

By the end of the course, students will be able to:
1. Formulate and decompose autonomous motion planning into a structured architectural hierarchy (Mission, Behaviour, Local/Motion Planning).
2. Represent static and dynamic driving environments using HD Vector Lanelet maps, Occupancy Grids, and Frenet Coordinate Frames.
3. Implement graph search algorithms (Dijkstra, A*) on large-scale road network topologies.
4. Design robust, deterministic Hierarchical State Machines (HSMs) for complex traffic scenarios (intersections, stop signs, lane changes).
5. Predict dynamic obstacle behaviors and evaluate spatial-temporal collision risks using Time-to-Collision (TTC) metrics.
6. Generate smooth, kinematically-feasible trajectories using parametric curves (cubic spirals, splines) and conformal lattices.
7. Optimize non-linear trajectory paths subject to lateral acceleration, friction ellipses, and obstacle clearance.
8. Generate dynamic velocity profiles obeying comfort and physical vehicle dynamics limits.
9. Integrate and benchmark a complete closed-loop autonomous driving stack.

---

## 📊 Detailed Weekly Schedule (10 Sessions)

### Week 1: Foundations of Motion Planning & Environmental Perception
#### Session 01: The Motion Planning Problem, Hierarchy & Vehicle Constraints
- **Theory (2 Hours):**
  - Decomposition of autonomous motion planning: Mission (Global Route) Planner, Behavior Planner, Local / Trajectory Planner, and Execution / Control.
  - Motion planning constraints: Non-holonomic kinematic constraints (Ackermann steering & bicycle model), dynamic constraints (lateral acceleration limits, friction ellipse / Kamm's circle), actuation limits, regulatory constraints, and geometric obstacle collision constraints.
  - Multi-objective cost formulation: Safety margin, passenger comfort (jerk, lateral acceleration minimization), efficiency, and rule compliance.
- **Practice (1 Hour):**
  - Implementation of a discrete-time non-holonomic Kinematic Bicycle Model in Python.
  - Verification of acceleration, steering angle, and curvature constraints.
  - Interactive trajectory rollout visualization and state-space plotting using **Plotly** and tabular state logging using **Polars**.

#### Session 02: Environmental Representation & Occupancy Grid Mapping
- **Theory (2 Hours):**
  - Road representations: High-Definition (HD) Vector maps, Lanelets, Road Graph representations.
  - 2D Occupancy Grid Maps: Binary state space, probabilistic cell representation, and Log-Odds formulation.
  - LiDAR and Sonar Inverse Measurement Models: Beam range-bearing sensor models, free space ray-casting, and occupied cell probability updates.
  - Efficient grid update pipelines and handling sensor noise/uncertainty.
- **Practice (1 Hour):**
  - Implementation of a 2D Log-Odds Occupancy Grid mapping pipeline in Python.
  - Processing simulated 2D LiDAR range and bearing point clouds using **Polars**.
  - Interactive heat-map visualization of probability grids using **Plotly**.

---

### Week 2: Mission Planning & Route Finding on Road Networks
#### Session 03: Topological Road Networks & Dijkstra's Algorithm
- **Theory (2 Hours):**
  - Road network graph representation: Directed graphs, intersection nodes, lane segment edges, geometric centerline interpolation.
  - Edge cost calculation: Physical distance, speed limit, road class weighting, lane count, and historical traffic factors.
  - Dijkstra’s Shortest Path Algorithm: Mathematical proof of optimality, graph exploration frontiers, priority queues (Min-Heaps), and path reconstruction.
- **Practice (1 Hour):**
  - Constructing a weighted directed road network graph using **NetworkX**.
  - Writing an optimized Dijkstra shortest path finder from scratch.
  - Visualizing the graph topology and explored paths on an interactive map using **Plotly**.

#### Session 04: Informed Search & A* Route Planning
- **Theory (2 Hours):**
  - Informed search principles: Evaluation function $f(n) = g(n) + h(n)$.
  - Heuristic function design: Euclidean distance, Manhattan distance, Dubins distance; Admissibility ($h(n) \le h^*(n)$) and Consistency ($h(n) \le c(n, a, n') + h(n')$).
  - Maneuver-aware routing: Incorporating turn penalties (left turns vs right turns), U-turn restrictions, lane-change costs, and traffic light delays.
- **Practice (1 Hour):**
  - Implementing A* search with custom admissible heuristics in Python.
  - Performance benchmarking: Exploring node expansions and computational runtime between Dijkstra and A* using **Polars**.
  - Interactive visual comparison of explored frontiers with **Plotly**.

---

### Week 3: Behaviour Planning & Dynamic Object Prediction
#### Session 05: Behaviour Planning & Hierarchical State Machines (HSM)
- **Theory (2 Hours):**
  - Role of behaviour planning: High-level tactical decision making without continuous control execution.
  - Finite State Machines (FSMs) vs Hierarchical State Machines (HSMs): State transitions, entry/exit actions, guard conditions, and hierarchical decomposition (e.g., cruising, intersection negotiation, lane following, emergency stop).
  - Rule-based vs. learning-based behaviour systems (Fuzzy logic, Decision trees, Reinforcement Learning, POMDPs).
- **Practice (1 Hour):**
  - Building a deterministic Hierarchical State Machine (HSM) Behaviour Planner in Python.
  - Handling dynamic scenarios: Stop-sign 3-second wait protocol, following a lead vehicle, and initiating left-turn maneuvers.
  - State transition timeline generation and verification with **Polars** and **Plotly**.

#### Session 06: Dynamic Environments & Object Motion Prediction
- **Theory (2 Hours):**
  - Dynamic object tracking and perception input formats (bounding boxes, velocity vectors, covariances).
  - Prediction modeling paradigms:
    - *Physics-based:* Constant Velocity (CV), Constant Acceleration (CA), Constant Turn Rate and Velocity (CTRV).
    - *Maneuver-based:* Lane keeping vs lane change classifiers.
    - *Interaction-aware:* Game-theoretic and potential field interaction models.
  - Risk assessment & Safety Metrics: Time-to-Collision (TTC), Time Headway (TH), and reachability sets.
- **Practice (1 Hour):**
  - Implementing multi-hypothesis motion prediction (CV and CTRV) for surrounding vehicles.
  - Calculating dynamic Time-to-Collision (TTC) and detecting impending collision polygons using **Shapely**.
  - Visualizing spatial-temporal obstacle trajectories and safety buffers with **Plotly**.

---

### Week 4: Local Planning & Trajectory Generation in Static/Dynamic Environments
#### Session 07: Reactive Planning & Trajectory Rollouts
- **Theory (2 Hours):**
  - Reactive planning fundamentals: Dynamic Window Approach (DWA) in velocity space ($v, \omega$).
  - Trajectory rollout algorithm: Forward simulation across a discretized lateral and longitudinal action space.
  - Collision checking in trajectory rollouts: Vehicle bounding box representations, multi-circle conservative approximations, swath generation along path segments.
  - Trajectory evaluation & selection: Multi-term cost function (distance to goal, clearance to obstacles, alignment to lane center).
- **Practice (1 Hour):**
  - Implementing the Trajectory Rollout generation algorithm with kinematic propagation.
  - Efficient circle-based collision checking and swath calculation.
  - Visualizing candidate rollout bundles and selected paths with **Plotly**.

#### Session 08: Parametric Curves & Conformal Spatiotemporal Lattices
- **Theory (2 Hours):**
  - Continuous curve generation for smooth driving: Polynomial paths (Cubic, Quintic), clothoids / Euler spirals, and Cubic Spirals ($\kappa(s) = a s^3 + b s^2 + c s + d$).
  - Frenet-Serret Road-Aligned Coordinate System: Curvilinear coordinates $(s, d)$ along reference path centerlines.
  - Conformal Lattice Planning: Discretizing lateral offsets and longitudinal horizons to form road-aligned lattices.
- **Practice (1 Hour):**
  - Implementing Frenet-to-Cartesian and Cartesian-to-Frenet coordinate transformations.
  - Generating candidate cubic spiral path lattices in Python.
  - Interactive lattice visualization overlaid on road boundaries with **Plotly**.

---

### Week 5: Trajectory Optimization, Velocity Profiling & Capstone Integration
#### Session 09: Numerical Path Optimization & Velocity Profiling
- **Theory (2 Hours):**
  - Non-linear path optimization: Formulating boundary value problems (BVP) with boundary conditions ($x, y, \theta, \kappa$).
  - Objective formulation: Minimizing bending energy $\int \kappa(s)^2 ds$, lateral acceleration, and obstacle repulsion potentials.
  - Velocity profile generation: Trapezoidal profiles, S-curve (jerk-limited) acceleration, curvature-speed limits ($v_{max} = \sqrt{a_{lat}^{max} / \kappa}$).
- **Practice (1 Hour):**
  - Implementing a continuous non-linear path optimizer using `scipy.optimize`.
  - Constructing a curvature-constrained trapezoidal velocity profiler in Python.
  - Analyzing speed, acceleration, and jerk profiles with **Polars** and **Plotly**.

#### Session 10: End-to-End Autonomous Driving Planning Capstone
- **Theory (2 Hours):**
  - Full-stack architectural integration: Mission Planner $\to$ Behaviour Planner $\to$ Local Path Optimizer $\to$ Velocity Planner $\to$ Low-Level Tracking Controller.
  - Fail-safe strategies, replanning triggers, emergency braking envelopes, and safety verification.
  - Simulation benchmarking in CARLA and automated validation pipelines.
- **Practice (1 Hour):**
  - Integrating all subsystems into an end-to-end autonomous motion planning pipeline.
  - Running a closed-loop scenario benchmark (handling parked cars, stop signs, pedestrian crossings).
  - Generating execution telemetry reports and interactive trajectory dashboards with **Polars** and **Plotly**.

---

## 🛠️ Software Stack & Tooling
- **Language:** Python 3.10+
- **Data Engineering:** `Polars` (DataFrame queries, waypoint interpolation, time-series telemetry)
- **Visualization:** `Plotly` (Interactive 2D spatial maps, 3D state charts, kinematic curves)
- **Scientific Computing:** `NumPy`, `SciPy` (Optimization, Splines, Numerical Integration)
- **Geometric / Graph Tools:** `NetworkX`, `Shapely`
- **Environment Management:** `uv` (Fast virtual environment and package resolver)
