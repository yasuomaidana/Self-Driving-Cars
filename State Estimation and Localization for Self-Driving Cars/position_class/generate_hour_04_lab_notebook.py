"""Generates both student (scaffolded TODOs) and instructor (solution) notebooks for Hour 4 Lab:
Ego-Vehicle 1D/2D Localization with Known Control Matrix G, Tunnel Outage, and External Target Object Tracking (Car-Following).
"""

import nbformat as nbf


def create_notebook(student_mode: bool = False):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "language_info": {"name": "python"},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    }

    title_suffix = " (Student Exercise Handout)" if student_mode else " (Complete Solution Guide)"

    # Header Markdown
    nb.cells.append(nbf.v4.new_markdown_cell(f"""# Autonomous Vehicle State Estimation & Localization
## Hour 4 Applied Programming Lab: Ego-Vehicle Localization with Control Inputs ($\mathbf{{G}}\mathbf{{u}}$) & Target Object Tracking{title_suffix}

> **Learning Objectives**:
> 1. Master the mathematical origins and continuous-to-discrete derivations of the 5 system matrices: $\\mathbf{{F}}, \\mathbf{{G}}, \\mathbf{{H}}, \\mathbf{{Q}}, \\mathbf{{R}}$.
> 2. **Ego-Vehicle 1D Longitudinal Tracking**: Estimate our own vehicle's position and velocity using GPS observations combined with known throttle/brake pedal acceleration commands $\\mathbf{{u}}_k = a_{{\\text{{cmd}}}}$.
> 3. **Ego-Vehicle 2D Localization & Mountain Tunnel Outage**: Navigate a curved mountain highway through a 5-second GPS blackout using dead reckoning with known IMU/throttle inputs $\\mathbf{{G}}\\mathbf{{u}}_k$ and observe covariance $\\mathbf{{P}}$ expansion and reacquisition.
> 4. **Target Object Tracking (Car-Following / ACC)**: Contrast ego-vehicle localization with passive lead-vehicle tracking where control inputs are unknown ($\\mathbf{{u}} = \\mathbf{{0}}$) and maneuvers are modeled via process noise $\\mathbf{{Q}}$.
> 5. **Filter Tuning, Statistical Consistency (NEES)**, and **Datasheet Prior Initialization** ($3\\sigma$ rule).
> 6. Understand why the **Joseph Stabilized Form** is required in production software instead of the textbook shortcut $(\\mathbf{{I}} - \\mathbf{{K}}\\mathbf{{H}})\\mathbf{{P}}$.

---
### 1. Mathematical Foundations: Where Do the Matrices Come From?

#### 1.1 State Transition $\\mathbf{{F}}$ and Control Matrix $\\mathbf{{G}}$ (Continuous-to-Discrete Kinematics)
Continuous vehicle dynamics obey:
$$\\dot{{\\mathbf{{x}}}}(t) = \\mathbf{{A}}\\mathbf{{x}}(t) + \\mathbf{{B}}\\mathbf{{u}}(t)$$

For 1D motion where $\\mathbf{{x}}(t) = [p(t), v(t)]^T$ and $u(t) = a_{{\\text{{cmd}}}}(t)$:
$$\\begin{{bmatrix}} \\dot{{p}}(t) \\\\ \\dot{{v}}(t) \\end{{bmatrix}} = \\underbrace{{\\begin{{bmatrix}} 0 & 1 \\\\ 0 & 0 \\end{{bmatrix}}}}_{{\\mathbf{{A}}}} \\begin{{bmatrix}} p(t) \\\\ v(t) \\end{{bmatrix}} + \\underbrace{{\\begin{{bmatrix}} 0 \\\\ 1 \\end{{bmatrix}}}}_{{\\mathbf{{B}}}} a_{{\\text{{cmd}}}}(t)$$

Discretizing over sample time $\\Delta t$ via the Matrix Exponential:
$$\\mathbf{{F}} = e^{{\\mathbf{{A}}\\Delta t}} = \\mathbf{{I}} + \\mathbf{{A}}\\Delta t = \\begin{{bmatrix}} 1 & \\Delta t \\\\ 0 & 1 \\end{{bmatrix}}$$
$$\\mathbf{{G}} = \\int_0^{{\\Delta t}} e^{{\\mathbf{{A}}\\tau}} \\mathbf{{B}} d\\tau = \\int_0^{{\\Delta t}} \\begin{{bmatrix}} \\tau \\\\ 1 \\end{{bmatrix}} d\\tau = \\begin{{bmatrix}} \\frac{{\\Delta t^2}}{{2}} \\\\[4pt] \\Delta t \\end{{bmatrix}}$$

#### 1.2 Process Noise Covariance $\\mathbf{{Q}}$ (Continuous White Noise Acceleration)
Unmodeled disturbances (road bumps, tire slip, wind) enter as continuous white acceleration noise with spectral density $\\sigma_a^2$:
$$\\mathbf{{Q}} = \\int_0^{{\\Delta t}} e^{{\\mathbf{{A}}(\\Delta t - \\tau)}} \\mathbf{{L}} \\sigma_a^2 \\mathbf{{L}}^T e^{{\\mathbf{{A}}^T(\\Delta t - \\tau)}} d\\tau = \\sigma_a^2 \\int_0^{{\\Delta t}} \\begin{{bmatrix}} s^2 & s \\\\ s & 1 \\end{{bmatrix}} ds = \\begin{{bmatrix}} \\frac{{\\Delta t^4}}{{4}} & \\frac{{\\Delta t^3}}{{2}} \\\\[4pt] \\frac{{\\Delta t^3}}{{2}} & \\Delta t^2 \\end{{bmatrix}} \\sigma_a^2$$

#### 1.3 Measurement Noise Covariance $\\mathbf{{R}}$ (Sensor Datasheets)
For sensor measurements $\\mathbf{{y}}_k = \\mathbf{{H}}\\mathbf{{x}}_k + \\mathbf{{v}}_k$:
$$\\mathbf{{H}} = \\begin{{bmatrix}} 1 & 0 \\end{{bmatrix}}, \\qquad \\mathbf{{R}} = [\\sigma_{{\\text{{meas}}}}^2]$$
where $\\sigma_{{\\text{{meas}}}}$ is directly specified in the sensor hardware datasheet (e.g. GPS standard deviation).

#### 1.4 The 5 Discrete Linear Kalman Filter Equations
$$\\begin{{array}}{{lll}}
\\hline
\\textbf{{Phase}} & \\textbf{{Operation}} & \\textbf{{Mathematical Equation}} \\\\
\\hline
\\text{{1. State Prediction}} & \\text{{Propagate State Mean}} & \\check{{\\mathbf{{x}}}}_k = \\mathbf{{F}}_{{k-1}}\\hat{{\\mathbf{{x}}}}_{{k-1}} + \\mathbf{{G}}_{{k-1}}\\mathbf{{u}}_{{k-1}} \\\\[4pt]
\\text{{2. Covariance Prediction}} & \\text{{Propagate Uncertainty}} & \\check{{\\mathbf{{P}}}}_k = \\mathbf{{F}}_{{k-1}}\\hat{{\\mathbf{{P}}}}_{{k-1}}\\mathbf{{F}}_{{k-1}}^T + \\mathbf{{Q}}_{{k-1}} \\\\[4pt]
\\text{{3. Kalman Gain}} & \\text{{Optimal Weighting Matrix}} & \\mathbf{{K}}_k = \\check{{\\mathbf{{P}}}}_k \\mathbf{{H}}_k^T \\left( \\mathbf{{H}}_k \\check{{\\mathbf{{P}}}}_k \\mathbf{{H}}_k^T + \\mathbf{{R}}_k \\right)^{{-1}} \\\\[4pt]
\\text{{4. State Correction}} & \\text{{Incorporate Measurement}} & \\hat{{\\mathbf{{x}}}}_k = \\check{{\\mathbf{{x}}}}_k + \\mathbf{{K}}_k (\\mathbf{{y}}_k - \\mathbf{{H}}_k \\check{{\\mathbf{{x}}}}_k) \\\\[4pt]
\\text{{5. Covariance Correction}} & \\text{{Joseph Stabilized Form}} & \\hat{{\\mathbf{{P}}}}_k = (\\mathbf{{I}} - \\mathbf{{K}}_k \\mathbf{{H}}_k) \\check{{\\mathbf{{P}}}}_k (\\mathbf{{I}} - \\mathbf{{K}}_k \\mathbf{{H}}_k)^T + \\mathbf{{K}}_k \\mathbf{{R}}_k \\mathbf{{K}}_k^T \\\\[4pt]
\\hline
\\end{{array}}$$
"""))

    # Imports Cell
    nb.cells.append(nbf.v4.new_code_cell("""import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

# Set random seed for reproducible stochastic simulations
np.random.seed(42)

print("Environment configured: NumPy, SciPy, and Plotly loaded successfully.")
"""))

    # Section 1: Kalman Filter Class Implementation
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## Section 1: Generic Linear Kalman Filter Implementation with Matrix $\\mathbf{G}$

Let us implement a generic, dimension-agnostic `LinearKalmanFilter` class in Python supporting:
* State transition matrix $\\mathbf{F} \\in \\mathbb{R}^{n \\times n}$
* Control input matrix $\\mathbf{G} \\in \\mathbb{R}^{n \\times p}$
* Measurement matrix $\\mathbf{H} \\in \\mathbb{R}^{m \\times n}$
* Process noise covariance $\\mathbf{Q} \\in \\mathbb{R}^{n \\times n}$
* Measurement noise covariance $\\mathbf{R} \\in \\mathbb{R}^{m \\times m}$
"""))

    if student_mode:
        kf_class_code = """class KalmanFilter:
    \"\"\"Discrete Linear Kalman Filter with Control Matrix G.\"\"\"
    
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray, 
                 G: np.ndarray = None, x0: np.ndarray = None, P0: np.ndarray = None):
        self.F = np.asarray(F, dtype=np.float64)
        self.H = np.asarray(H, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        
        self.n = self.F.shape[0]  # State dimension
        self.m = self.H.shape[0]  # Measurement dimension
        
        if G is not None:
            self.G = np.asarray(G, dtype=np.float64)
        else:
            self.G = np.zeros((self.n, 1))
            
        self.x = np.zeros((self.n, 1)) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(self.n, 1)
        self.P = np.eye(self.n) * 100.0 if P0 is None else np.asarray(P0, dtype=np.float64).reshape(self.n, self.n)
        
        self.latest_innovation = None
        self.latest_innovation_cov = None
        self.latest_gain = None

    def predict(self, u: np.ndarray = None):
        \"\"\"Executes the Prediction Step with Control Input:
            x_check = F * x_hat + G * u
            P_check = F * P_hat * F^T + Q
        \"\"\"
        # ---------------------------------------------------------------------
        # TODO 1.1: Implement State and Covariance Prediction
        # ---------------------------------------------------------------------
        # if u is not None:
        #     u_vec = np.asarray(u, dtype=np.float64).reshape(-1, 1)
        #     self.x = self.F @ self.x + self.G @ u_vec
        # else:
        #     self.x = self.F @ self.x
        # self.P = self.F @ self.P @ self.F.T + self.Q
        pass  # <-- TODO: YOUR CODE HERE

    def update(self, y: np.ndarray):
        \"\"\"Executes the Measurement Update (Correction) Step:
            nu = y - H * x_check
            S  = H * P_check * H^T + R
            K  = P_check * H^T * inv(S)
            x_hat = x_check + K * nu
            P_hat = (I - K * H) * P_check
        \"\"\"
        # ---------------------------------------------------------------------
        # TODO 1.2: Implement Measurement Update Equations (Joseph Form)
        # ---------------------------------------------------------------------
        pass  # <-- TODO: YOUR CODE HERE
"""
    else:
        kf_class_code = """class KalmanFilter:
    \"\"\"Discrete Linear Kalman Filter with Control Matrix G.\"\"\"
    
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray, 
                 G: np.ndarray = None, x0: np.ndarray = None, P0: np.ndarray = None):
        self.F = np.asarray(F, dtype=np.float64)
        self.H = np.asarray(H, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        
        self.n = self.F.shape[0]  # State dimension
        self.m = self.H.shape[0]  # Measurement dimension
        
        if G is not None:
            self.G = np.asarray(G, dtype=np.float64)
        else:
            self.G = np.zeros((self.n, 1))
            
        self.x = np.zeros((self.n, 1)) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(self.n, 1)
        self.P = np.eye(self.n) * 100.0 if P0 is None else np.asarray(P0, dtype=np.float64).reshape(self.n, self.n)
        
        self.latest_innovation = None
        self.latest_innovation_cov = None
        self.latest_gain = None

    def predict(self, u: np.ndarray = None):
        \"\"\"Executes the Prediction Step with Control Input:
            x_check = F * x_hat + G * u
            P_check = F * P_hat * F^T + Q
        \"\"\"
        if u is not None:
            u_vec = np.asarray(u, dtype=np.float64).reshape(-1, 1)
            self.x = self.F @ self.x + self.G @ u_vec
        else:
            self.x = self.F @ self.x
            
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, y: np.ndarray):
        \"\"\"Executes the Measurement Update (Correction) Step:
            nu = y - H * x_check
            S  = H * P_check * H^T + R
            K  = P_check * H^T * inv(S)
            x_hat = x_check + K * nu
            P_hat = (I - K * H) * P_check
        \"\"\"
        y_vec = np.asarray(y, dtype=np.float64).reshape(self.m, 1)
        
        # 1. Innovation residual
        nu = y_vec - self.H @ self.x
        
        # 2. Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # 3. Optimal Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 4. A posteriori State Estimate
        self.x = self.x + K @ nu
        
        # 5. A posteriori Error Covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        self.latest_innovation = nu
        self.latest_innovation_cov = S
        self.latest_gain = K
        
        return self.x.copy(), self.P.copy()
"""
    nb.cells.append(nbf.v4.new_code_cell(kf_class_code))

    # Unit test block for generic class
    nb.cells.append(nbf.v4.new_markdown_cell("""### Unit Test: Kalman Filter Matrix Operations Sanity Check (with Control Input $\\mathbf{G}\\mathbf{u}$)"""))
    nb.cells.append(nbf.v4.new_code_cell("""# Sanity verification of Kalman Filter class with G matrix
dt_test = 0.1
G_test = np.array([[0.5 * dt_test**2], [dt_test]])
kf_test = KalmanFilter(
    F=np.array([[1.0, dt_test], [0.0, 1.0]]),
    H=np.array([[1.0, 0.0]]),
    Q=np.diag([0.001, 0.001]),
    R=np.array([[0.25]]),
    G=G_test,
    x0=np.array([[10.0], [2.0]]),
    P0=np.eye(2) * 4.0
)

# Test Predict with control input u = 2.0 m/s^2
# Expected x = 10.0 + 2.0*0.1 + 0.5*(0.1^2)*2.0 = 10.2 + 0.01 = 10.21
# Expected v = 2.0 + 2.0*0.1 = 2.2
x_pred, P_pred = kf_test.predict(u=np.array([[2.0]]))
assert np.isclose(x_pred[0, 0], 10.21), f"Expected pos 10.21, got {x_pred[0, 0]}"
assert np.isclose(x_pred[1, 0], 2.20), f"Expected vel 2.20, got {x_pred[1, 0]}"

# Test Update
y_test = np.array([[10.25]])
x_upd, P_upd = kf_test.update(y_test)
assert x_upd.shape == (2, 1)
assert P_upd.shape == (2, 2)
assert np.all(np.linalg.eigvals(P_upd) > 0), "P must remain strictly positive definite!"
print("✅ KalmanFilter class unit tests with Control Input G passed!")
"""))

    # Section 2: 1D Ego-Vehicle State Estimation with Commanded Throttle / Braking
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## Section 2: Problem 1 — 1D Ego-Vehicle State Estimation with Commanded Throttle & Braking ($\mathbf{G}\mathbf{u}$)

### Physical Scenario
We are driving our **own autonomous ego-vehicle**. The powertrain controller issues commanded longitudinal accelerations $u_k = a_{\\text{cmd}, k}$ via the throttle and brake pedals.
* $0 \\le t < 5\\text{s}$: Cruising at $v = 10\\text{ m/s}$ ($u = 0\\text{ m/s}^2$).
* $5 \\le t < 10\\text{s}$: Overtaking acceleration burst ($u = +2.5\\text{ m/s}^2$).
* $10 \\le t < 15\\text{s}$: Highway cruise at $v = 22.5\\text{ m/s}$ ($u = 0\\text{ m/s}^2$).
* $15 \\le t \\le 20\\text{s}$: Smooth braking into toll booth ($u = -3.0\\text{ m/s}^2$).

A GPS receiver measures position $p_k$ at $10\\text{ Hz}$ ($\\sigma_{\\text{gps}} = 1.5\\text{ m}$).

### State Vector & Matrices with Control Matrix $\\mathbf{G}$
$$\\mathbf{x} = \\begin{bmatrix} p \\\\ v \\end{bmatrix}, \\quad 
\\mathbf{F} = \\begin{bmatrix} 1 & \\Delta t \\\\ 0 & 1 \\end{bmatrix}, \\quad
\\mathbf{G} = \\begin{bmatrix} \\frac{\\Delta t^2}{2} \\\\[4pt] \\Delta t \\end{bmatrix}, \\quad
\\mathbf{H} = \\begin{bmatrix} 1 & 0 \\end{bmatrix}$$

$$\\check{\\mathbf{x}}_k = \\mathbf{F}\\hat{\\mathbf{x}}_{k-1} + \\mathbf{G} u_{k-1}$$
"""))

    if student_mode:
        p1_code = """# 1. Generate 1D Ego-Vehicle Trajectory with Driver Acceleration Commands
dt_1d = 0.1
time_1d = np.arange(0, 20.0, dt_1d)
N_1d = len(time_1d)

# Commanded Acceleration Profile u(t)
u_cmd_1d = np.zeros(N_1d)
for i, t in enumerate(time_1d):
    if 5.0 <= t < 10.0:
        u_cmd_1d[i] = 2.5   # Acceleration burst (+2.5 m/s^2)
    elif 15.0 <= t <= 20.0:
        u_cmd_1d[i] = -3.0  # Deceleration braking (-3.0 m/s^2)

# Simulate Ground Truth Kinematics
true_p_1d = np.zeros(N_1d)
true_v_1d = np.zeros(N_1d)
true_v_1d[0] = 10.0  # Initial speed: 10 m/s

for k in range(1, N_1d):
    true_p_1d[k] = true_p_1d[k-1] + true_v_1d[k-1]*dt_1d + 0.5 * u_cmd_1d[k-1] * (dt_1d**2)
    true_v_1d[k] = true_v_1d[k-1] + u_cmd_1d[k-1]*dt_1d

# Noisy GPS measurements (sigma = 1.5 m)
sigma_gps_1d = 1.5
meas_p_1d = true_p_1d + np.random.normal(0, sigma_gps_1d, size=N_1d)

# -----------------------------------------------------------------------------
# TODO 2.1: Define F, G, H, Q, R for 1D Ego-Vehicle Tracking
# -----------------------------------------------------------------------------
sigma_a_noise = 0.3  # Unmodeled throttle/road slope disturbance

F_1d = None  # <-- TODO: np.array([[1.0, dt_1d], [0.0, 1.0]])
G_1d = None  # <-- TODO: np.array([[0.5 * dt_1d**2], [dt_1d]])
H_1d = None  # <-- TODO: np.array([[1.0, 0.0]])
Q_1d = None  # <-- TODO: Continuous white noise acceleration Q
R_1d = None  # <-- TODO: np.array([[sigma_gps_1d**2]])

# TODO 2.2: Run Kalman Filter with Control Input u_cmd
# ...
"""
    else:
        p1_code = """# 1. Generate 1D Ego-Vehicle Trajectory with Driver Acceleration Commands
dt_1d = 0.1
time_1d = np.arange(0, 20.0, dt_1d)
N_1d = len(time_1d)

# Commanded Acceleration Profile u(t)
u_cmd_1d = np.zeros(N_1d)
for i, t in enumerate(time_1d):
    if 5.0 <= t < 10.0:
        u_cmd_1d[i] = 2.5   # Acceleration burst (+2.5 m/s^2)
    elif 15.0 <= t <= 20.0:
        u_cmd_1d[i] = -3.0  # Deceleration braking (-3.0 m/s^2)

# Simulate Ground Truth Kinematics
true_p_1d = np.zeros(N_1d)
true_v_1d = np.zeros(N_1d)
true_v_1d[0] = 10.0  # Initial speed: 10 m/s

for k in range(1, N_1d):
    true_p_1d[k] = true_p_1d[k-1] + true_v_1d[k-1]*dt_1d + 0.5 * u_cmd_1d[k-1] * (dt_1d**2)
    true_v_1d[k] = true_v_1d[k-1] + u_cmd_1d[k-1]*dt_1d

# Noisy GPS measurements (sigma = 1.5 m)
sigma_gps_1d = 1.5
meas_p_1d = true_p_1d + np.random.normal(0, sigma_gps_1d, size=N_1d)

# 2. Assemble System Matrices with Control Input G
sigma_a_noise = 0.3

F_1d = np.array([
    [1.0, dt_1d],
    [0.0, 1.0  ]
])

G_1d = np.array([
    [0.5 * (dt_1d**2)],
    [dt_1d           ]
])

H_1d = np.array([[1.0, 0.0]])

dt4 = (dt_1d**4) / 4.0 * (sigma_a_noise**2)
dt3 = (dt_1d**3) / 2.0 * (sigma_a_noise**2)
dt2 = (dt_1d**2) * (sigma_a_noise**2)
Q_1d = np.array([
    [dt4, dt3],
    [dt3, dt2]
])

R_1d = np.array([[sigma_gps_1d**2]])

# 3. Filter Execution: With Control Input (Ego) vs Blind to Control (u=0)
kf_with_u = KalmanFilter(F=F_1d, H=H_1d, Q=Q_1d, R=R_1d, G=G_1d, x0=np.array([[meas_p_1d[0]], [10.0]]), P0=np.diag([4.0, 4.0]))
kf_no_u   = KalmanFilter(F=F_1d, H=H_1d, Q=Q_1d, R=R_1d, G=G_1d, x0=np.array([[meas_p_1d[0]], [10.0]]), P0=np.diag([4.0, 4.0]))

est_p_with_u, est_v_with_u = [], []
est_p_no_u, est_v_no_u     = [], []
cov_p_with_u, cov_v_with_u = [], []

for k in range(N_1d):
    # Predict with known commanded pedal input
    kf_with_u.predict(u=np.array([[u_cmd_1d[k]]]))
    # Predict assuming zero input (blind to throttle)
    kf_no_u.predict(u=np.array([[0.0]]))
    
    y_k = np.array([[meas_p_1d[k]]])
    kf_with_u.update(y_k)
    kf_no_u.update(y_k)
    
    est_p_with_u.append(kf_with_u.x[0, 0])
    est_v_with_u.append(kf_with_u.x[1, 0])
    est_p_no_u.append(kf_no_u.x[0, 0])
    est_v_no_u.append(kf_no_u.x[1, 0])
    cov_p_with_u.append(kf_with_u.P[0, 0])
    cov_v_with_u.append(kf_with_u.P[1, 1])

est_p_with_u = np.array(est_p_with_u)
est_v_with_u = np.array(est_v_with_u)
est_p_no_u   = np.array(est_p_no_u)
est_v_no_u   = np.array(est_v_no_u)
cov_p_with_u = np.array(cov_p_with_u)
cov_v_with_u = np.array(cov_v_with_u)

rmse_v_with_u = np.sqrt(np.mean((true_v_1d - est_v_with_u)**2))
rmse_v_no_u   = np.sqrt(np.mean((true_v_1d - est_v_no_u)**2))

print("--- 1D Ego-Vehicle Estimation with Control Input G ---")
print(f"Velocity RMSE with Known Control (G*u):   {rmse_v_with_u:.3f} m/s")
print(f"Velocity RMSE Blind to Control (u=0):     {rmse_v_no_u:.3f} m/s (Lag error is {((rmse_v_no_u - rmse_v_with_u)/rmse_v_with_u)*100:.1f}% higher!)")
"""
    nb.cells.append(nbf.v4.new_code_cell(p1_code))

    # Plotly 1D Visualization
    nb.cells.append(nbf.v4.new_markdown_cell("""### Plotly Interactive Visualization: Ego-Vehicle Longitudinal State with Commanded Throttle/Braking Inputs"""))
    nb.cells.append(nbf.v4.new_code_cell("""fig_1d = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=(
        "<b>Commanded Acceleration Input u(t) = a_cmd [m/s²] (Throttle / Brake Pedals)</b>",
        "<b>Ego-Vehicle Position Tracking & ±3σ Confidence Bounds</b>",
        "<b>Velocity Estimation: Known Control G*u vs. Blind Lag (u=0)</b>"
    ),
    vertical_spacing=0.08
)

# Row 1: Commanded Input u(t)
fig_1d.add_trace(go.Scatter(x=time_1d, y=u_cmd_1d, mode='lines+markers', line=dict(color='darkorange', width=2), name='Commanded Accel u(t)'), row=1, col=1)

# Row 2: Position
sigma_3p = 3.0 * np.sqrt(cov_p_with_u)
fig_1d.add_trace(go.Scatter(x=time_1d, y=true_p_1d, mode='lines', line=dict(color='black', width=2), name='True Position'), row=2, col=1)
fig_1d.add_trace(go.Scatter(x=time_1d, y=meas_p_1d, mode='markers', marker=dict(color='crimson', size=4, opacity=0.4), name='Noisy GPS Pings'), row=2, col=1)
fig_1d.add_trace(go.Scatter(x=time_1d, y=est_p_with_u, mode='lines', line=dict(color='royalblue', width=2), name='Ego LKF Estimate (with G*u)'), row=2, col=1)
fig_1d.add_trace(go.Scatter(
    x=np.concatenate([time_1d, time_1d[::-1]]),
    y=np.concatenate([est_p_with_u + sigma_3p, (est_p_with_u - sigma_3p)[::-1]]),
    fill='toself', fillcolor='rgba(65, 105, 225, 0.2)', line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip", showlegend=True, name='±3σ Position Bound'
), row=2, col=1)

# Row 3: Velocity
fig_1d.add_trace(go.Scatter(x=time_1d, y=true_v_1d, mode='lines', line=dict(color='black', width=2.5), name='True Velocity'), row=3, col=1)
fig_1d.add_trace(go.Scatter(x=time_1d, y=est_v_with_u, mode='lines', line=dict(color='forestgreen', width=2), name='Ego LKF (with G*u) - Instant Response'), row=3, col=1)
fig_1d.add_trace(go.Scatter(x=time_1d, y=est_v_no_u, mode='lines', line=dict(color='gray', width=2, dash='dot'), name='Blind Estimator (u=0) - Dynamic Lag'), row=3, col=1)

fig_1d.update_xaxes(title_text="Time [s]", row=3, col=1)
fig_1d.update_yaxes(title_text="Accel [m/s²]", row=1, col=1)
fig_1d.update_yaxes(title_text="Position [m]", row=2, col=1)
fig_1d.update_yaxes(title_text="Velocity [m/s]", row=3, col=1)

fig_1d.update_layout(
    title="<b>1D Ego-Vehicle Estimation: Accelerating and Braking with Control Matrix G</b>",
    template="plotly_white",
    height=800,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig_1d.show()
"""))

    # Section 3: 2D Ego-Vehicle Localization with Control Inputs & Tunnel GPS Outage
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## Section 3: Problem 2 — 2D Ego-Vehicle Localization & The Mountain Tunnel Outage ($t = 10\\text{ s} \\to 15\\text{ s}$)

### Physical Scenario: Driving Our Own Vehicle through a Tunnel
We are driving our own vehicle along a curved mountain highway into a **5-second dark tunnel** ($t = 10\\text{s} \\to 15\\text{s}$) where GPS satellite signals are completely lost.

Because **we are driving our own vehicle**, we have direct access to our onboard IMU / steering / throttle acceleration commands:
$$\\mathbf{u}_k = \\begin{bmatrix} a_{x, \\text{cmd}, k} \\\\[2pt] a_{y, \\text{cmd}, k} \\end{bmatrix}$$

### 2D Kinematic Model with Control Matrix $\\mathbf{G}$
$$\\mathbf{x}_k = \\begin{bmatrix} x_k \\\\ y_k \\\\ \\dot{x}_k \\\\ \\dot{y}_k \\end{bmatrix} \\in \\mathbb{R}^4, \\quad
\\mathbf{F} = \\begin{bmatrix} 1 & 0 & \\Delta t & 0 \\\\ 0 & 1 & 0 & \\Delta t \\\\ 0 & 0 & 1 & 0 \\\\ 0 & 0 & 0 & 1 \\end{bmatrix}, \\quad
\\mathbf{G} = \\begin{bmatrix} \\frac{\\Delta t^2}{2} & 0 \\\\[4pt] 0 & \\frac{\\Delta t^2}{2} \\\\[4pt] \\Delta t & 0 \\\\[4pt] 0 & \\Delta t \\end{bmatrix}$$

### Inside the Tunnel ($10\\text{s} \\le t \\le 15\\text{s}$):
1. **With Control ($\\mathbf{G}\\mathbf{u}$)**: The vehicle executes dead reckoning using known driver steering/throttle commands:
   $$\\check{\\mathbf{x}}_k = \\mathbf{F}\\hat{\\mathbf{x}}_{k-1} + \\mathbf{G}\\mathbf{u}_{k-1}$$
   It follows the curved highway path accurately even in complete darkness!
2. **Without Control ($u=0$)**: Assuming zero acceleration causes the vehicle to drift straight and cut corners inside the tunnel.
3. **Covariance $\\mathbf{P}$ Expansion**: $\\check{\\mathbf{P}}_k = \\mathbf{F}\\hat{\\mathbf{P}}_{k-1}\\mathbf{F}^T + \\mathbf{Q}$ expands continuously, signaling expanding uncertainty until GPS reacquisition at $t=15.1\\text{s}$.
"""))

    if student_mode:
        p2_code = """# 1. Generate 2D Curved Highway Trajectory with Commanded Lateral Acceleration
dt_2d = 0.1
time_2d = np.arange(0, 30.0, dt_2d)
N_2d = len(time_2d)

# Commanded Acceleration Profile: Constant forward speed vx=15 m/s, sinusoidal lateral steering
u_x_2d = np.zeros(N_2d)
u_y_2d = 2.4 * np.cos(0.3 * time_2d)  # Lateral steering acceleration [m/s^2]
u_cmd_2d = np.vstack([u_x_2d, u_y_2d]).T

# Simulate Ground Truth Path
true_x_2d = np.zeros(N_2d)
true_y_2d = np.zeros(N_2d)
true_vx_2d = np.zeros(N_2d)
true_vy_2d = np.zeros(N_2d)
true_vx_2d[0] = 15.0

for k in range(1, N_2d):
    true_x_2d[k]  = true_x_2d[k-1] + true_vx_2d[k-1]*dt_2d + 0.5*u_x_2d[k-1]*(dt_2d**2)
    true_y_2d[k]  = true_y_2d[k-1] + true_vy_2d[k-1]*dt_2d + 0.5*u_y_2d[k-1]*(dt_2d**2)
    true_vx_2d[k] = true_vx_2d[k-1] + u_x_2d[k-1]*dt_2d
    true_vy_2d[k] = true_vy_2d[k-1] + u_y_2d[k-1]*dt_2d

true_states_2d = np.vstack([true_x_2d, true_y_2d, true_vx_2d, true_vy_2d]).T

# GPS Measurements (dropped out between 10s and 15s)
sigma_meas_2d = 3.0
meas_x_2d = true_x_2d + np.random.normal(0, sigma_meas_2d, size=N_2d)
meas_y_2d = true_y_2d + np.random.normal(0, sigma_meas_2d, size=N_2d)

# -----------------------------------------------------------------------------
# TODO 3.1: Construct F, G, H, Q, R for 2D Ego-Vehicle Localization
# -----------------------------------------------------------------------------
# ...
"""
    else:
        p2_code = """# 1. Generate 2D Curved Highway Trajectory with Commanded Lateral Acceleration
dt_2d = 0.1
time_2d = np.arange(0, 30.0, dt_2d)
N_2d = len(time_2d)

# Commanded Acceleration Profile: Constant forward speed vx=15 m/s, sinusoidal lateral steering
u_x_2d = np.zeros(N_2d)
u_y_2d = 2.4 * np.cos(0.3 * time_2d)  # Lateral steering acceleration [m/s^2]
u_cmd_2d = np.vstack([u_x_2d, u_y_2d]).T

# Simulate Ground Truth Path
true_x_2d = np.zeros(N_2d)
true_y_2d = np.zeros(N_2d)
true_vx_2d = np.zeros(N_2d)
true_vy_2d = np.zeros(N_2d)
true_vx_2d[0] = 15.0

for k in range(1, N_2d):
    true_x_2d[k]  = true_x_2d[k-1] + true_vx_2d[k-1]*dt_2d + 0.5*u_x_2d[k-1]*(dt_2d**2)
    true_y_2d[k]  = true_y_2d[k-1] + true_vy_2d[k-1]*dt_2d + 0.5*u_y_2d[k-1]*(dt_2d**2)
    true_vx_2d[k] = true_vx_2d[k-1] + u_x_2d[k-1]*dt_2d
    true_vy_2d[k] = true_vy_2d[k-1] + u_y_2d[k-1]*dt_2d

true_states_2d = np.vstack([true_x_2d, true_y_2d, true_vx_2d, true_vy_2d]).T

# GPS Measurements
sigma_meas_2d = 3.0
meas_x_2d = true_x_2d + np.random.normal(0, sigma_meas_2d, size=N_2d)
meas_y_2d = true_y_2d + np.random.normal(0, sigma_meas_2d, size=N_2d)

# 2. Assemble 2D Matrices with Control Input G
sigma_a_proc = 0.5

F_2d = np.array([
    [1.0, 0.0, dt_2d, 0.0  ],
    [0.0, 1.0, 0.0,   dt_2d],
    [0.0, 0.0, 1.0,   0.0  ],
    [0.0, 0.0, 0.0,   1.0  ]
])

G_2d = np.array([
    [0.5 * (dt_2d**2), 0.0             ],
    [0.0,              0.5 * (dt_2d**2)],
    [dt_2d,            0.0             ],
    [0.0,              dt_2d           ]
])

H_2d = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
])

dt4 = (dt_2d**4) / 4.0 * (sigma_a_proc**2)
dt3 = (dt_2d**3) / 2.0 * (sigma_a_proc**2)
dt2 = (dt_2d**2) * (sigma_a_proc**2)
Q_2d = np.array([
    [dt4, 0.0, dt3, 0.0],
    [0.0, dt4, 0.0, dt3],
    [dt3, 0.0, dt2, 0.0],
    [0.0, dt3, 0.0, dt2]
])

R_2d = np.eye(2) * (sigma_meas_2d**2)

x0_2d = np.array([[meas_x_2d[0]], [meas_y_2d[0]], [15.0], [0.0]])
P0_2d = np.diag([9.0, 9.0, 1.0, 1.0])

# 3. Filter Simulation: With Control Inputs G*u vs Blind (u=0) during 5s Tunnel Outage
kf_ego_with_u = KalmanFilter(F=F_2d, H=H_2d, Q=Q_2d, R=R_2d, G=G_2d, x0=x0_2d, P0=P0_2d)
kf_ego_no_u   = KalmanFilter(F=F_2d, H=H_2d, Q=Q_2d, R=R_2d, G=G_2d, x0=x0_2d, P0=P0_2d)

est_states_with_u = []
est_states_no_u   = []
cov_matrices_2d   = []
gps_active_mask   = []

for k in range(N_2d):
    t = time_2d[k]
    
    # 1. State & Uncertainty Prediction
    u_k = u_cmd_2d[k].reshape(-1, 1)
    kf_ego_with_u.predict(u=u_k)
    kf_ego_no_u.predict(u=np.zeros((2, 1)))  # Blind to driver steering
    
    # 2. Conditional Measurement Update: GPS drops out in Tunnel (10s <= t <= 15s)
    if not (10.0 <= t <= 15.0):
        y_k = np.array([[meas_x_2d[k]], [meas_y_2d[k]]])
        kf_ego_with_u.update(y_k)
        kf_ego_no_u.update(y_k)
        gps_active_mask.append(True)
    else:
        gps_active_mask.append(False)  # Prediction-only mode (Dead Reckoning)
        
    est_states_with_u.append(kf_ego_with_u.x.flatten())
    est_states_no_u.append(kf_ego_no_u.x.flatten())
    cov_matrices_2d.append(kf_ego_with_u.P.copy())

est_states_with_u = np.array(est_states_with_u)
est_states_no_u   = np.array(est_states_no_u)
cov_matrices_2d   = np.array(cov_matrices_2d)
gps_active_mask   = np.array(gps_active_mask)

# Performance during the tunnel outage (10s to 15s)
tunnel_idx = (time_2d >= 10.0) & (time_2d <= 15.0)
tunnel_rmse_with_u = np.sqrt(np.mean((true_y_2d[tunnel_idx] - est_states_with_u[tunnel_idx, 1])**2))
tunnel_rmse_no_u   = np.sqrt(np.mean((true_y_2d[tunnel_idx] - est_states_no_u[tunnel_idx, 1])**2))

print("--- 2D Ego-Vehicle Mountain Tunnel Performance ---")
print(f"Tunnel Outage North RMSE with Known Control (G*u): {tunnel_rmse_with_u:.3f} m")
print(f"Tunnel Outage North RMSE Blind to Control (u=0):   {tunnel_rmse_no_u:.3f} m (Drift error is {((tunnel_rmse_no_u - tunnel_rmse_with_u)/tunnel_rmse_with_u)*100:.1f}% higher!)")
"""
    nb.cells.append(nbf.v4.new_code_cell(p2_code))

    # Plotly 2D Tunnel Visualization
    nb.cells.append(nbf.v4.new_markdown_cell("""### Plotly Interactive Visualization: 2D Ego-Vehicle Mountain Tunnel Navigation with Control Matrix $\\mathbf{G}$"""))
    nb.cells.append(nbf.v4.new_code_cell("""fig_2d = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        "<b>2D Mountain Highway: Navigation through 5-Second Tunnel (t = 10s to 15s)</b>",
        "<b>Ego-Vehicle Position Uncertainty (3σ Bound) Expanding during Blackout</b>"
    ),
    vertical_spacing=0.15
)

# Row 1: 2D Trajectory Map
fig_2d.add_trace(go.Scatter(x=true_x_2d, y=true_y_2d, mode='lines', line=dict(color='black', width=3), name='True Highway Path'), row=1, col=1)
fig_2d.add_trace(go.Scatter(x=meas_x_2d[gps_active_mask], y=meas_y_2d[gps_active_mask], mode='markers', marker=dict(color='green', size=4, opacity=0.4), name='Active GPS Fixes'), row=1, col=1)
fig_2d.add_trace(go.Scatter(x=est_states_with_u[:, 0], y=est_states_with_u[:, 1], mode='lines', line=dict(color='royalblue', width=2.5), name='Ego LKF (with G*u) - Accurate Dead Reckoning'), row=1, col=1)
fig_2d.add_trace(go.Scatter(x=est_states_no_u[:, 0], y=est_states_no_u[:, 1], mode='lines', line=dict(color='crimson', width=2, dash='dot'), name='Blind Estimator (u=0) - Severe Corner Cutting'), row=1, col=1)

# Row 2: 3-Sigma Position Uncertainty vs Time
sigma_3y = 3.0 * np.sqrt(cov_matrices_2d[:, 1, 1])
fig_2d.add_trace(go.Scatter(x=time_2d, y=sigma_3y, mode='lines', line=dict(color='darkorange', width=2.5), name='3σ North Uncertainty [m]'), row=2, col=1)
fig_2d.add_vrect(x0=10.0, x1=15.0, fillcolor="red", opacity=0.15, line_width=0, annotation_text="5-SECOND TUNNEL OUTAGE (Dead Reckoning Mode)", annotation_position="top left", row=2, col=1)

fig_2d.update_xaxes(title_text="East Position [m]", row=1, col=1)
fig_2d.update_yaxes(title_text="North Position [m]", row=1, col=1)
fig_2d.update_xaxes(title_text="Time [s]", row=2, col=1)
fig_2d.update_yaxes(title_text="3σ Position Error [m]", row=2, col=1)

fig_2d.update_layout(
    title="<b>2D Ego-Vehicle Localization: Maneuvering with Control Matrix G through Tunnel</b>",
    template="plotly_white",
    height=750,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig_2d.show()
"""))

    # Section 4: Target Object Tracking & Car-Following
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## Section 4: Problem 3 — Target Object Tracking: Car-Following & Adaptive Cruise Control (ACC)

### 4.1 The Autonomous Driving Challenge: Why Do We Need Kalman Filters for ACC?
In modern Advanced Driver Assistance Systems (ADAS) and Autonomous Driving, **Adaptive Cruise Control (ACC)** and **Autonomous Emergency Braking (AEB)** must maintain a safe following distance behind the leading vehicle.

#### 1. The Constant Time-Gap Safety Policy:
$$d_{\\text{safe}}(t) = d_0 + T_{\\text{gap}} \\cdot v_{\\text{ego}}(t)$$
where $d_0$ is the standstill buffer distance (e.g. $5.0\\text{ m}$) and $T_{\\text{gap}}$ is the time headway (e.g. $1.5\\text{ s}$).

#### 2. Time-To-Collision (TTC) Critical Safety Metric:
$$\\text{TTC}(t) = \\begin{cases} \\frac{d(t)}{-\\Delta v(t)} & \\text{if } \\Delta v(t) < 0 \\text{ (closing distance / crash hazard)} \\\\[4pt] +\\infty & \\text{if } \\Delta v(t) \\ge 0 \\text{ (opening gap / safe)} \\end{cases}$$
When $\\text{TTC} < \\text{TTC}_{\\text{critical}}$ (typically $2.5\\text{ s}$), the vehicle must initiate autonomous emergency braking.

---

### 4.2 The Disaster of Direct Differentiation vs. Kalman Filtering
A front-facing Radar or LiDAR measures relative distance $d_{\\text{meas}}$ with noise $\\sigma_r \\approx 0.5\\text{ m}$.
If an engineer naively estimates relative velocity using finite differences:
$$v_{\\text{diff}, k} = \\frac{d_{\\text{meas}, k} - d_{\\text{meas}, k-1}}{\\Delta t}$$
The resulting noise variance explodes:
$$\\sigma_{v, \\text{diff}} = \\frac{\\sqrt{2}\\sigma_r}{\\Delta t} = \\frac{\\sqrt{2} \\cdot 0.5}{0.1} \\approx 7.07\\text{ m/s} \\quad (\\approx 25.5\\text{ km/h}!)$$
This massive noise causes the autonomous emergency braking controller to chatter violently ("phantom braking").

**The Kalman Filter solves this completely:** It enforces the physical kinematic relationship $\\dot{d} = \\Delta v$, filtering out distance noise and recovering an ultra-smooth, accurate latent closing velocity $\\Delta v$.

---

### 4.3 Why is Matrix $\\mathbf{G}$ Omitted When Tracking External Objects?
* When tracking an external target, we **DO NOT have access to the lead driver's accelerator pedal or CAN bus** ($\\mathbf{u} = \\mathbf{0}$).
* Therefore, $\\mathbf{G}\\mathbf{u}$ is omitted from the prediction step: $\\check{\\mathbf{x}}_k = \\mathbf{F}\\hat{\\mathbf{x}}_{k-1}$.
* Instead, the leading vehicle's unpredictable braking and acceleration maneuvers are modeled as **stochastic process noise**:
  $$\\mathbf{w}_k \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{Q})$$
"""))

    if student_mode:
        p3_code = """# 1. Lead Vehicle Tracking Simulation
dt_acc = 0.1
time_acc = np.arange(0, 25.0, dt_acc)
N_acc = len(time_acc)

# Lead car speed profile with sudden hard braking at t = 10s
true_v_lead = 20.0 * np.ones(N_acc)
for i, t in enumerate(time_acc):
    if 10.0 <= t < 15.0:
        true_v_lead[i] = 10.0  # Hard braking from 20 m/s down to 10 m/s
    elif t >= 15.0:
        true_v_lead[i] = 22.0  # Speeding back up

# Relative kinematics: Ego travels at constant 20 m/s (72 km/h)
v_ego_const = 20.0
true_rel_v = true_v_lead - v_ego_const
true_rel_d = 40.0 + np.cumsum(true_rel_v * dt_acc)

# Radar distance measurements (sigma = 0.5 m)
sigma_radar = 0.5
meas_rel_d = true_rel_d + np.random.normal(0, sigma_radar, size=N_acc)

# Naive finite difference velocity (shows why Kalman filtering is necessary!)
naive_v_diff = np.diff(meas_rel_d, prepend=meas_rel_d[0]) / dt_acc

# -----------------------------------------------------------------------------
# TODO 4.1: Formulate Target Tracking Kalman Filter (u=0, G not used)
# -----------------------------------------------------------------------------
# F_acc = ...
# H_acc = ...
# Q_acc = ... (Process noise capturing lead car braking agility)
# R_acc = ...
"""
    else:
        p3_code = """# 1. Lead Vehicle Tracking Simulation
dt_acc = 0.1
time_acc = np.arange(0, 25.0, dt_acc)
N_acc = len(time_acc)

# Lead car speed profile with sudden hard braking at t = 10s
true_v_lead = 20.0 * np.ones(N_acc)
for i, t in enumerate(time_acc):
    if 10.0 <= t < 15.0:
        true_v_lead[i] = 10.0  # Hard braking from 20 m/s down to 10 m/s
    elif t >= 15.0:
        true_v_lead[i] = 22.0  # Speeding back up

# Relative kinematics: Ego travels at constant 20 m/s (72 km/h)
v_ego_const = 20.0
true_rel_v = true_v_lead - v_ego_const
true_rel_d = 40.0 + np.cumsum(true_rel_v * dt_acc)

# Radar distance measurements (sigma = 0.5 m)
sigma_radar = 0.5
meas_rel_d = true_rel_d + np.random.normal(0, sigma_radar, size=N_acc)

# Naive finite difference velocity (shows why Kalman filtering is necessary!)
naive_v_diff = np.diff(meas_rel_d, prepend=meas_rel_d[0]) / dt_acc

# 2. Assemble Target Tracking Kalman Filter (u=0, maneuvers absorbed by Q)
sigma_lead_acc = 2.0  # Process noise capturing lead car braking agility (m/s^2)

F_acc = np.array([
    [1.0, dt_acc],
    [0.0, 1.0   ]
])

H_acc = np.array([[1.0, 0.0]])

dt4 = (dt_acc**4) / 4.0 * (sigma_lead_acc**2)
dt3 = (dt_acc**3) / 2.0 * (sigma_lead_acc**2)
dt2 = (dt_acc**2) * (sigma_lead_acc**2)
Q_acc = np.array([
    [dt4, dt3],
    [dt3, dt2]
])

R_acc = np.array([[sigma_radar**2]])

# Initialize filter: Distance from first radar ping, relative speed guess 0 m/s
kf_acc = KalmanFilter(F=F_acc, H=H_acc, Q=Q_acc, R=R_acc, x0=np.array([[meas_rel_d[0]], [0.0]]), P0=np.diag([1.0, 4.0]))

est_rel_d = []
est_rel_v = []
cov_rel_v = []

for k in range(N_acc):
    # Predict: Note that u is NOT passed (u=0) because we cannot command the lead car!
    kf_acc.predict()
    
    y_k = np.array([[meas_rel_d[k]]])
    kf_acc.update(y_k)
    
    est_rel_d.append(kf_acc.x[0, 0])
    est_rel_v.append(kf_acc.x[1, 0])
    cov_rel_v.append(kf_acc.P[1, 1])

est_rel_d = np.array(est_rel_d)
est_rel_v = np.array(est_rel_v)
cov_rel_v = np.array(cov_rel_v)

# 3. Compute Real-Time Time-To-Collision (TTC)
ttc_true = np.where(true_rel_v < -0.1, true_rel_d / (-true_rel_v), np.nan)
ttc_est  = np.where(est_rel_v < -0.1, est_rel_d / (-est_rel_v), np.nan)

# Performance Evaluation
rmse_rel_d   = np.sqrt(np.mean((true_rel_d - est_rel_d)**2))
rmse_rel_v   = np.sqrt(np.mean((true_rel_v - est_rel_v)**2))
rmse_naive_v = np.sqrt(np.mean((true_rel_v - naive_v_diff)**2))

print("--- Adaptive Cruise Control (ACC) Lead Car Tracking ---")
print(f"Radar Raw Distance RMSE:         {np.sqrt(np.mean((true_rel_d - meas_rel_d)**2)):.3f} m")
print(f"Kalman Filter Distance RMSE:     {rmse_rel_d:.3f} m")
print(f"Naive Finite-Difference Vel RMSE:{rmse_naive_v:.3f} m/s (Unusable for AEB!)")
print(f"Kalman Filter Estimated Vel RMSE:{rmse_rel_v:.3f} m/s (Error reduced by {((rmse_naive_v - rmse_rel_v)/rmse_naive_v)*100:.1f}%)")
"""
    nb.cells.append(nbf.v4.new_code_cell(p3_code))

    # Plotly ACC Visualization
    nb.cells.append(nbf.v4.new_markdown_cell("""### Plotly Interactive Visualization: 3-Panel ACC Dashboard (Distance, Closing Speed & Time-To-Collision)"""))
    nb.cells.append(nbf.v4.new_code_cell("""fig_acc = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=(
        "<b>1. Lead Vehicle Relative Distance d(t) [m] (Radar vs. Kalman Filter)</b>",
        "<b>2. Closing Velocity Δv(t) [m/s]: Kalman Filter vs. Noisy Finite Difference</b>",
        "<b>3. Time-To-Collision TTC(t) [s] & Emergency Braking Activation Trigger</b>"
    ),
    vertical_spacing=0.08
)

# Row 1: Relative Distance
fig_acc.add_trace(go.Scatter(x=time_acc, y=true_rel_d, mode='lines', line=dict(color='black', width=2.5), name='True Relative Distance'), row=1, col=1)
fig_acc.add_trace(go.Scatter(x=time_acc, y=meas_rel_d, mode='markers', marker=dict(color='gray', size=4, opacity=0.4), name='Noisy Radar Pings (σ=0.5m)'), row=1, col=1)
fig_acc.add_trace(go.Scatter(x=time_acc, y=est_rel_d, mode='lines', line=dict(color='blue', width=2), name='Kalman Distance Track'), row=1, col=1)
fig_acc.add_hline(y=20.0, line_dash="dash", line_color="orange", annotation_text="ACC Warning Distance (20m)", row=1, col=1)

# Row 2: Relative Velocity
sigma_3_rel_v = 3.0 * np.sqrt(cov_rel_v)
fig_acc.add_trace(go.Scatter(x=time_acc, y=naive_v_diff, mode='lines', line=dict(color='crimson', width=1, dash='dot'), opacity=0.4, name='Naive Finite Difference Δd/Δt (Unstable!)'), row=2, col=1)
fig_acc.add_trace(go.Scatter(x=time_acc, y=true_rel_v, mode='lines', line=dict(color='black', width=2.5), name='True Relative Velocity'), row=2, col=1)
fig_acc.add_trace(go.Scatter(x=time_acc, y=est_rel_v, mode='lines', line=dict(color='forestgreen', width=2.5), name='Kalman Filter Latent Δv Estimate'), row=2, col=1)
fig_acc.add_trace(go.Scatter(
    x=np.concatenate([time_acc, time_acc[::-1]]),
    y=np.concatenate([est_rel_v + sigma_3_rel_v, (est_rel_v - sigma_3_rel_v)[::-1]]),
    fill='toself', fillcolor='rgba(34, 139, 34, 0.2)', line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip", showlegend=True, name='±3σ Relative Speed Bound'
), row=2, col=1)

# Row 3: Time-To-Collision (TTC)
fig_acc.add_trace(go.Scatter(x=time_acc, y=ttc_true, mode='lines', line=dict(color='black', width=2), name='True TTC [s]'), row=3, col=1)
fig_acc.add_trace(go.Scatter(x=time_acc, y=ttc_est, mode='lines+markers', line=dict(color='purple', width=2), name='Kalman Estimated TTC [s]'), row=3, col=1)
fig_acc.add_hline(y=2.5, line_dash="dash", line_color="red", annotation_text="CRITICAL AEB THRESHOLD (TTC < 2.5s -> EMERGENCY BRAKE)", row=3, col=1)
fig_acc.add_vrect(x0=10.0, x1=15.0, fillcolor="red", opacity=0.15, line_width=0, annotation_text="LEAD CAR HARD BRAKING", annotation_position="top right", row=3, col=1)

fig_acc.update_xaxes(title_text="Time [s]", row=3, col=1)
fig_acc.update_yaxes(title_text="Distance [m]", row=1, col=1)
fig_acc.update_yaxes(title_text="Relative Speed [m/s]", row=2, col=1)
fig_acc.update_yaxes(title_text="TTC [s]", row=3, col=1, range=[0, 10])

fig_acc.update_layout(
    title="<b>Target Object Tracking: Adaptive Cruise Control (ACC) & Autonomous Emergency Braking (AEB)</b>",
    template="plotly_white",
    height=850,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig_acc.show()
"""))

    # Section 5: Filter Tuning Trade-offs
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## Section 5: Problem 4 — The Engineering Art of Tuning $\\mathbf{Q}$ vs. $\\mathbf{R}$

The ratio of process noise $\\mathbf{Q}$ to measurement noise $\\mathbf{R}$ dictates filter agility:
* **Overconfident Model ($\\mathbf{Q}$ too low)**: Filter ignores sensor fixes; high dynamic lag.
* **Underconfident Model ($\\mathbf{Q}$ too high)**: Filter ignores motion model; estimates jitter with raw noise.
* **Optimal Balance**: Smooth trajectory filtering while quickly tracking real dynamic maneuvers.
"""))

    if student_mode:
        p4_code = """# -----------------------------------------------------------------------------
# TODO 5.1: Run 3 Filters with different Q tuning (Low Q, Nom Q, High Q)
# -----------------------------------------------------------------------------
"""
    else:
        p4_code = """# Tuning comparison simulation on 2D Trajectory
def run_kf_simulation(sigma_a_val: float):
    dt4 = (dt_2d**4) / 4.0 * (sigma_a_val**2)
    dt3 = (dt_2d**3) / 2.0 * (sigma_a_val**2)
    dt2 = (dt_2d**2) * (sigma_a_val**2)
    Q_tune = np.array([
        [dt4, 0.0, dt3, 0.0],
        [0.0, dt4, 0.0, dt3],
        [dt3, 0.0, dt2, 0.0],
        [0.0, dt3, 0.0, dt2]
    ])
    kf_tune = KalmanFilter(F=F_2d, H=H_2d, Q=Q_tune, R=R_2d, G=G_2d, x0=x0_2d, P0=P0_2d)
    traj = []
    for k in range(N_2d):
        u_k = u_cmd_2d[k].reshape(-1, 1)
        kf_tune.predict(u=u_k)
        kf_tune.update(np.array([[meas_x_2d[k]], [meas_y_2d[k]]]))
        traj.append(kf_tune.x.flatten())
    return np.array(traj)

traj_low_q = run_kf_simulation(sigma_a_val=0.05)
traj_nom_q = run_kf_simulation(sigma_a_val=0.50)
traj_high_q = run_kf_simulation(sigma_a_val=15.0)

rmse_low = np.sqrt(np.mean((true_y_2d - traj_low_q[:, 1])**2))
rmse_nom = np.sqrt(np.mean((true_y_2d - traj_nom_q[:, 1])**2))
rmse_high = np.sqrt(np.mean((true_y_2d - traj_high_q[:, 1])**2))

print(f"--- Q Tuning Comparison (North Position RMSE) ---")
print(f"Overconfident Model (Low Q,  σ_a = 0.05): RMSE = {rmse_low:.3f} m (Severe Dynamic Lag)")
print(f"Optimal Nominal Model (Nom Q, σ_a = 0.50): RMSE = {rmse_nom:.3f} m")
print(f"Underconfident Model (High Q, σ_a = 15.0): RMSE = {rmse_high:.3f} m (Noisy Jitter)")
"""
    nb.cells.append(nbf.v4.new_code_cell(p4_code))

    # Plotly Tuning Comparison
    nb.cells.append(nbf.v4.new_markdown_cell("""### Plotly Interactive Visualization: Filter Tuning Comparison"""))
    nb.cells.append(nbf.v4.new_code_cell("""fig_tune = go.Figure()

fig_tune.add_trace(go.Scatter(x=time_2d, y=true_y_2d, mode='lines', line=dict(color='black', width=3), name='True North Position'))
fig_tune.add_trace(go.Scatter(x=time_2d, y=meas_y_2d, mode='markers', marker=dict(color='gray', size=4, opacity=0.3), name='Noisy GPS'))
fig_tune.add_trace(go.Scatter(x=time_2d, y=traj_low_q[:, 1], mode='lines', line=dict(color='red', width=2, dash='dot'), name=f'Low Q (σ_a=0.05, RMSE={rmse_low:.2f}m) - Lag'))
fig_tune.add_trace(go.Scatter(x=time_2d, y=traj_nom_q[:, 1], mode='lines', line=dict(color='blue', width=2.5), name=f'Nominal Q (σ_a=0.5, RMSE={rmse_nom:.2f}m) - Optimal'))
fig_tune.add_trace(go.Scatter(x=time_2d, y=traj_high_q[:, 1], mode='lines', line=dict(color='green', width=1.5, dash='dash'), name=f'High Q (σ_a=15.0, RMSE={rmse_high:.2f}m) - Jitter'))

fig_tune.update_layout(
    title="<b>Q Tuning Trade-off: Dynamic Lag (Low Q) vs. Sensor Noise Jitter (High Q)</b>",
    xaxis_title="Time [s]",
    yaxis_title="North Position [m]",
    template="plotly_white",
    height=550,
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
)
fig_tune.show()
"""))

    # Section 6: Statistical Consistency & NEES
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## Section 6: Problem 5 — Filter Statistical Consistency & NEES Analysis

### Normalized Estimation Error Squared (NEES)
$$\\epsilon_k = (\\mathbf{x}_{\\text{true}, k} - \\hat{\\mathbf{x}}_k)^T \\hat{\\mathbf{P}}_k^{-1} (\\mathbf{x}_{\\text{true}, k} - \\hat{\\mathbf{x}}_k)$$

For a consistent 4D filter:
* Expected value: $\\mathbb{E}[\\epsilon_k] = n = 4$
* 95% Confidence Upper Bound: $\\chi_4^2(0.95) = 9.488$
"""))

    if student_mode:
        p5_code = """# -----------------------------------------------------------------------------
# TODO 6.1: Compute NEES across all time steps
# -----------------------------------------------------------------------------
"""
    else:
        p5_code = """# Compute NEES across all time steps
nees_values = []
for k in range(N_2d):
    err = (true_states_2d[k] - est_states_with_u[k]).reshape(-1, 1)
    P_k = cov_matrices_2d[k]
    nees_k = (err.T @ np.linalg.inv(P_k) @ err).item()
    nees_values.append(nees_k)

nees_values = np.array(nees_values)
mean_nees = np.mean(nees_values)
chi2_95_limit = stats.chi2.ppf(0.95, df=4)
fraction_within_bounds = np.mean(nees_values <= chi2_95_limit) * 100.0

print(f"--- Filter Consistency Evaluation (NEES) ---")
print(f"Theoretical Mean NEES: {4.0:.1f}")
print(f"Actual Empirical Mean:  {mean_nees:.2f}")
print(f"95% Chi-Square Bound:   {chi2_95_limit:.3f}")
print(f"Empirical % in Bounds:  {fraction_within_bounds:.1f}% (Expected ~95%)")
"""
    nb.cells.append(nbf.v4.new_code_cell(p5_code))

    # Plotly NEES Chart
    nb.cells.append(nbf.v4.new_markdown_cell("""### Plotly Interactive Visualization: NEES Statistical Consistency"""))
    nb.cells.append(nbf.v4.new_code_cell("""fig_nees = go.Figure()

fig_nees.add_trace(go.Scatter(x=time_2d, y=nees_values, mode='lines', line=dict(color='royalblue', width=1.5), name='Sample NEES ε_k'))
fig_nees.add_trace(go.Scatter(x=time_2d, y=[4.0]*N_2d, mode='lines', line=dict(color='black', dash='dash', width=2), name='Theoretical Expected Mean (n=4)'))
fig_nees.add_trace(go.Scatter(x=time_2d, y=[chi2_95_limit]*N_2d, mode='lines', line=dict(color='crimson', dash='dot', width=2), name=f'95% Chi-Square Limit ({chi2_95_limit:.2f})'))

fig_nees.update_layout(
    title=f"<b>Normalized Estimation Error Squared (NEES) Consistency Test (Mean: {mean_nees:.2f}, In-bounds: {fraction_within_bounds:.1f}%)</b>",
    xaxis_title="Time [s]",
    yaxis_title="NEES Value ε_k",
    template="plotly_white",
    height=500,
    legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98)
)
fig_nees.show()
"""))

    # Section 7: Datasheet Prior Initialization
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## Section 7: Problem 6 — Datasheet Prior Initialization vs. Diffuse Prior

### Physical Specification
Suppose the vehicle starts from an intersection stop line where CAD and GNSS base station survey provide:
* Nominal Position: $x_0 = 0.0\\text{ m} \\pm 0.6\\text{ m}$ ($3\\sigma_x = 0.6\\text{ m} \\implies \\sigma_x = 0.2\\text{ m}$)
* Nominal Speed: $v_0 = 0.0\\text{ m/s} \\pm 0.3\\text{ m/s}$ ($3\\sigma_v = 0.3\\text{ m/s} \\implies \\sigma_v = 0.1\\text{ m/s}$)

Let us compare **Datasheet Prior Initialization** ($P_0 = \\operatorname{diag}([0.04, 0.01])$) against a cold **Diffuse Prior** ($P_0 = \\operatorname{diag}([1000.0, 1000.0])$).
"""))

    if student_mode:
        p6_code = """# -----------------------------------------------------------------------------
# TODO 7.1: Compare Datasheet Prior vs Diffuse Prior Convergence
# -----------------------------------------------------------------------------
"""
    else:
        p6_code = """# Datasheet vs Diffuse Prior comparison
# Datasheet Prior (3-sigma rule)
x0_spec = np.array([[0.0], [0.0]])
P0_spec = np.diag([(0.6/3.0)**2, (0.3/3.0)**2])  # diag([0.04, 0.01])

# Diffuse Prior (Cold start)
x0_diff = np.array([[0.0], [0.0]])
P0_diff = np.diag([1000.0, 1000.0])

kf_spec = KalmanFilter(F=F_1d, H=H_1d, Q=Q_1d, R=R_1d, G=G_1d, x0=x0_spec, P0=P0_spec)
kf_diff = KalmanFilter(F=F_1d, H=H_1d, Q=Q_1d, R=R_1d, G=G_1d, x0=x0_diff, P0=P0_diff)

pos_err_spec = []
pos_err_diff = []

for k in range(30):  # First 3 seconds
    u_k = np.array([[u_cmd_1d[k]]])
    kf_spec.predict(u=u_k)
    kf_diff.predict(u=u_k)
    
    y_k = np.array([[meas_p_1d[k]]])
    kf_spec.update(y_k)
    kf_diff.update(y_k)
    
    pos_err_spec.append(abs(kf_spec.x[0, 0] - true_p_1d[k]))
    pos_err_diff.append(abs(kf_diff.x[0, 0] - true_p_1d[k]))

print("--- Initial Startup Comparison (First 5 steps Error in meters) ---")
print("Datasheet Prior Error:", np.round(pos_err_spec[:5], 3))
print("Diffuse Prior Error:  ", np.round(pos_err_diff[:5], 3))
"""
    nb.cells.append(nbf.v4.new_code_cell(p6_code))

    # Plotly Prior Comparison
    nb.cells.append(nbf.v4.new_markdown_cell("""### Plotly Interactive Visualization: Startup Transient Comparison"""))
    nb.cells.append(nbf.v4.new_code_cell("""fig_prior = go.Figure()

fig_prior.add_trace(go.Scatter(x=time_1d[:30], y=pos_err_spec, mode='lines+markers', line=dict(color='forestgreen', width=2.5), name='Datasheet Prior (P_0 based on ±3σ Spec)'))
fig_prior.add_trace(go.Scatter(x=time_1d[:30], y=pos_err_diff, mode='lines+markers', line=dict(color='crimson', width=2, dash='dot'), name='Diffuse Prior (P_0 = 1000 I)'))

fig_prior.update_layout(
    title="<b>Startup Transient: Datasheet Prior Initialization vs. Cold Diffuse Prior</b>",
    xaxis_title="Time [s]",
    yaxis_title="Absolute Position Error |p_est - p_true| [m]",
    template="plotly_white",
    height=480,
    legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98)
)
fig_prior.show()
"""))

    # Summary Markdown
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## Section 8: Summary & Engineering Takeaways

1. **Ego-Vehicle Localization with Control Inputs ($\\mathbf{G}\\mathbf{u}$)**: When estimating our own car, known driver/powertrain acceleration commands are injected directly through $\\mathbf{G}\\mathbf{u}$, preventing dynamic lag during acceleration and braking maneuvers.
2. **Tunnel Blackout & Dead Reckoning**: During GPS outages, having known IMU/steering inputs $\\mathbf{G}\\mathbf{u}$ allows accurate dead reckoning through mountain highway curves while error covariance $\\mathbf{P}$ correctly expands to reflect loss of sensor observations.
3. **Target Object Tracking (ACC)**: When tracking external vehicles, control inputs are unavailable ($\\mathbf{u}=\\mathbf{0}$, $\\mathbf{G}$ omitted). The Kalman Filter models unknown target maneuvers via process noise $\\mathbf{Q}$ and recovers relative distance and closing speed for collision avoidance.
4. **Consistency & Initialization**: Statistical metrics like NEES confirm filter health, while Datasheet Prior initialization eliminates transient errors from the very first frame.

---
### Next Step: Nonlinear Kalman Filtering (Day 2)
When sensors observe in polar/spherical coordinate frames (Radar Range $r$, Bearing $\\theta$, Doppler $\\dot{r}$), we advance to the **Extended Kalman Filter (EKF)** and **Unscented Kalman Filter (UKF)** in Day 2!
"""))

    return nb


if __name__ == "__main__":
    # 1. Generate Instructor Solution Notebook
    nb_sol = create_notebook(student_mode=False)
    with open("hour_04_linear_kalman_filter_lab.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb_sol, f)
    print("Generated hour_04_linear_kalman_filter_lab.ipynb (Complete Solution)")

    # 2. Generate Student Exercise Notebook
    nb_stud = create_notebook(student_mode=True)
    with open("Day_01_linear_kalman_filter_lab_student.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb_stud, f)
    print("Generated Day_01_linear_kalman_filter_lab_student.ipynb (Student Exercise)")
