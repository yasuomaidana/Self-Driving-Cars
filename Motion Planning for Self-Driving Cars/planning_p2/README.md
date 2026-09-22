# Motion Planning for Autonomous Vehicles (Planning P2)
## 5-Week Intensive Course (10 Sessions × 3 Hours)

Welcome to the **Motion Planning for Self-Driving Cars (Planning P2)** course repository. This course provides an in-depth, hands-on journey through modern autonomous vehicle motion planning—from high-level mission routing and behaviour decision-making down to reactive trajectory rollouts, non-linear spline optimization, velocity profiling, and CARLA simulator integration.

---

## 📅 Course Structure

The course spans **5 weeks**, with **2 sessions per week**. Each session is **3 hours** (30 hours total):
- **Theory (2 Hours):** Mathematical formulations, algorithmic principles, vehicle dynamics, state-space modeling, and architectural patterns.
- **Practice & Coding (1 Hour):** Hands-on implementation in Python using **`Polars`** for high-performance data/telemetry processing and **`Plotly`** for interactive spatial-temporal visualization.

```
planning_p2/
├── .venv/                                                      # uv virtual environment
├── README.md                                                   # Course overview & environment setup
├── SYLLABUS.md                                                 # Full 10-session syllabus & grading policy
├── pyproject.toml / requirements.txt                           # Project dependencies
├── week_01/
│   ├── session_01_introduction_and_kinematic_constraints/     # Planning hierarchy & Vehicle kinematics
│   └── session_02_environmental_mapping_and_occupancy_grids/  # HD maps & LiDAR Occupancy Grids
├── week_02/
│   ├── session_03_mission_planning_and_dijkstra/              # Road networks & Dijkstra's Algorithm
│   └── session_04_informed_search_and_astar/                  # Heuristic design & A* routing
├── week_03/
│   ├── session_05_behaviour_planning_and_fsm/                 # Hierarchical State Machines (HSMs)
│   └── session_06_dynamic_prediction_and_collision_risk/      # Motion prediction & TTC risk metrics
├── week_04/
│   ├── session_07_reactive_planning_and_trajectory_rollouts/  # DWA, Trajectory rollouts & Swaths
│   └── session_08_smooth_parametric_curves_and_conformal_lattices/ # Cubic spirals & Frenet lattices
└── week_05/
    ├── session_09_path_optimization_and_velocity_profiling/   # Nonlinear optimization & Velocity profiles
    └── session_10_end_to_end_integrated_planning_capstone/    # Full-stack autonomous planner capstone
```

---

## 🛠️ Environment Setup with `uv`

This repository uses [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package and environment manager.

### 1. Prerequisites
Ensure `uv` is installed on your system:
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or via Homebrew (macOS)
brew install uv
```

### 2. Activate Virtual Environment
The virtual environment is already initialized in `planning_p2/.venv`. Activate it using:
```bash
# From within planning_p2 directory:
source .venv/bin/activate
```

### 3. Dependencies
All core libraries are installed:
- **`polars`**: High-performance data processing for waypoints, sensor telemetry, and dynamic obstacle trajectories.
- **`plotly`**: Interactive web-based 2D and 3D visualizers for search graphs, lattices, and trajectories.
- **`numpy` & `scipy`**: Numerical computation, polynomial fitting, numerical integration, and optimization (`scipy.optimize`).
- **`networkx` & `shapely`**: Graph representations for road networks and geometric collision checking.
- **`jupyterlab`**: Interactive notebook environments for lab exploration.

To install or sync dependencies:
```bash
uv pip install -r requirements.txt
```

---

## 🧭 Course Roadmap at a Glance

| Week | Session | Topic | Theory (2h) Focus | Practice (1h) Coding Lab |
| :--- | :--- | :--- | :--- | :--- |
| **W1** | **S01** | Planning Hierarchy & Constraints | Vehicle Kinematics, Dynamics, Actuation Limits & Cost Formulation | Bicycle Kinematic Simulator & Constraint Validation (`Polars`/`Plotly`) |
| **W1** | **S02** | Occupancy Grids & Map Representation | HD Maps, Lanelets, Log-Odds Formulation & Inverse LiDAR Model | 2D Occupancy Grid Builder from LiDAR scans (`Polars`/`Plotly`) |
| **W2** | **S03** | Mission Planning & Dijkstra | Road Graph Topology, Edge Costs, Dijkstra Algorithm & Heaps | Graph Construction & Dijkstra Route Solver (`NetworkX`/`Plotly`) |
| **W2** | **S04** | Informed Search & A* Routing | Heuristic Admissibility, Consistent Heuristics, Lane Rules | A* Search vs Dijkstra Benchmark & Lane Cost Modifiers |
| **W3** | **S05** | Behaviour Planning & HSMs | Finite & Hierarchical State Machines, Transition Guards | HSM Rule-Based Behaviour Planner for Intersections & Stop Signs |
| **W3** | **S06** | Dynamic Prediction & Collision Risk | Physics/Maneuver-based Prediction, Time-to-Collision (TTC) | Multi-Hypothesis Obstacle Prediction & TTC Collision Checker |
| **W4** | **S07** | Reactive Planning & Rollouts | Dynamic Window Approach (DWA), Trajectory Propagation & Swaths | Trajectory Rollout Generator & Circle Collision Checking |
| **W4** | **S08** | Parametric Curves & Conformal Lattices | Cubic Spirals, Quintic Splines, Frenet Frames ($s, d$) | Conformal Lattice Planner in Frenet Coordinates |
| **W5** | **S09** | Path Optimization & Velocity Profiling | SciPy Non-linear Optimization, Trapezoidal & Curvature Profiles | Optimization-based Path Smoother & Trapezoidal Velocity Profiler |
| **W5** | **S10** | End-to-End Planning Capstone | Closed-Loop Integration, Fallback Strategies & Benchmarking | Full Stack Autonomous Driving Planner & Performance Telemetry Dashboard |

---

## 📖 Recommended References & Textbooks
1. **S. M. LaValle**, *Planning Algorithms*, Cambridge University Press, 2006. [Online access](https://lavalle.pl/planning/).
2. **S. Thrun, W. Burgard, and D. Fox**, *Probabilistic Robotics*, MIT Press, 2005.
3. **P. Bender et al.**, *Lanelets: Efficient Map Representation for Autonomous Driving*, IEEE Intelligent Vehicles Symposium (IV), 2014.
4. **M. McNaughton**, *Parallel Algorithms for Real-Time Motion Planning*, Ph.D. Thesis, Carnegie Mellon University, 2011.
