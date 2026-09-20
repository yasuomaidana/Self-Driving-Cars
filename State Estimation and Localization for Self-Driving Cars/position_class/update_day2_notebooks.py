"""Script to update and harmonize all Day 2 Notebooks with generic, academic Kalman Filter definitions."""

import json
import numpy as np

def generate_ekf_notebooks():
    # -------------------------------------------------------------
    # EKF INSTRUCTOR NOTEBOOK
    # -------------------------------------------------------------
    cells_inst = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Day 02: The Extended Kalman Filter (EKF) - Nonlinear Kinematics & Radar Tracking\n",
                "**State Estimation and Localization for Self-Driving Cars**\n",
                "\n",
                "### 🎯 Learning Objectives:\n",
                "1. Master the **First-Order Taylor Series Linearization** of nonlinear motion $\\mathbf{f}(\\mathbf{x}, \\mathbf{u})$ and sensor observation $\\mathbf{h}(\\mathbf{x})$ functions.\n",
                "2. Understand the systematic derivation of state transition Jacobians $\\mathbf{F}_{k-1}, \\mathbf{L}_{k-1}$ and measurement Jacobians $\\mathbf{H}_k, \\mathbf{M}_k$.\n",
                "3. Implement the generic multi-dimensional `ExtendedKalmanFilter` class following standard academic formulations.\n",
                "4. Apply EKF to 2D automotive radar tracking with range $r$ and azimuth angle $\\phi$ in polar coordinates.\n",
                "5. Evaluate filter statistical consistency using the Normalized Estimation Error Squared (NEES) and $\\chi^2$ hypothesis testing.\n",
                "6. Build interactive multi-panel Plotly diagnostic dashboards."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 1. Environment Setup"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import plotly.graph_objects as go\n",
                "from plotly.subplots import make_subplots\n",
                "from scipy.stats import chi2\n",
                "\n",
                "# Set random seed for reproducible stochastic simulations\n",
                "np.random.seed(42)\n",
                "print('Environment ready: NumPy, SciPy, and Plotly loaded.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 2. Mathematical Foundation: Discrete Extended Kalman Filter\n",
                "\n",
                "In real-world robotics and autonomous vehicles, dynamics and measurement processes are fundamentally **nonlinear**:\n",
                "\n",
                "$$\\mathbf{x}_k = \\mathbf{f}(\\mathbf{x}_{k-1}, \\mathbf{u}_{k-1}) + \\mathbf{w}_{k-1}, \\quad \\mathbf{w}_{k-1} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{Q}_{k-1})$$\n",
                "$$\\mathbf{y}_k = \\mathbf{h}(\\mathbf{x}_k) + \\mathbf{v}_k, \\quad \\mathbf{v}_k \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{R}_k)$$\n",
                "\n",
                "The Extended Kalman Filter linearizes $\\mathbf{f}$ and $\\mathbf{h}$ locally around the current state estimate via 1st-order Taylor expansions:\n",
                "\n",
                "$$\\mathbf{F}_{k-1} = \\left. \\frac{\\partial \\mathbf{f}}{\\partial \\mathbf{x}} \\right|_{\\hat{\\mathbf{x}}_{k-1}, \\mathbf{u}_{k-1}}, \\quad \\mathbf{L}_{k-1} = \\left. \\frac{\\partial \\mathbf{f}}{\\partial \\mathbf{w}} \\right|_{\\hat{\\mathbf{x}}_{k-1}, \\mathbf{u}_{k-1}}$$\n",
                "$$\\mathbf{H}_k = \\left. \\frac{\\partial \\mathbf{h}}{\\partial \\mathbf{x}} \\right|_{\\check{\\mathbf{x}}_k}, \\quad \\mathbf{M}_k = \\left. \\frac{\\partial \\mathbf{h}}{\\partial \\mathbf{v}} \\right|_{\\check{\\mathbf{x}}_k}$$\n",
                "\n",
                "### 🔁 Discrete EKF Recursive Algorithm:\n",
                "\n",
                "| Step | Mathematical Formula | Academic Definition |\n",
                "| :--- | :--- | :--- |\n",
                "| **1. State Prediction** | $\\check{\\mathbf{x}}_k = \\mathbf{f}(\\hat{\\mathbf{x}}_{k-1}, \\mathbf{u}_{k-1})$ | Propagate state through full nonlinear kinematic dynamics |\n",
                "| **2. Covariance Prediction** | $\\check{\\mathbf{P}}_k = \\mathbf{F}_{k-1}\\hat{\\mathbf{P}}_{k-1}\\mathbf{F}_{k-1}^T + \\mathbf{L}_{k-1}\\mathbf{Q}_{k-1}\\mathbf{L}_{k-1}^T$ | Linearized uncertainty expansion |\n",
                "| **3. Innovation Residual** | $\\mathbf{\\nu}_k = \\mathbf{y}_k - \\mathbf{h}(\\check{\\mathbf{x}}_k)$ | Discrepancy between actual sensor data and predicted measurement |\n",
                "| **4. Innovation Covariance** | $\\mathbf{S}_k = \\mathbf{H}_k\\check{\\mathbf{P}}_k\\mathbf{H}_k^T + \\mathbf{M}_k\\mathbf{R}_k\\mathbf{M}_k^T$ | Total uncertainty projected in measurement space |\n",
                "| **5. Kalman Gain** | $\\mathbf{K}_k = \\check{\\mathbf{P}}_k\\mathbf{H}_k^T \\mathbf{S}_k^{-1}$ | Optimal minimum mean squared error (MMSE) gain |\n",
                "| **6. State Correction** | $\\hat{\\mathbf{x}}_k = \\check{\\mathbf{x}}_k + \\mathbf{K}_k \\mathbf{\\nu}_k$ | A posteriori optimal state estimate |\n",
                "| **7. Covariance Correction** | $\\hat{\\mathbf{P}}_k = (\\mathbf{I} - \\mathbf{K}_k\\mathbf{H}_k)\\check{\\mathbf{P}}_k(\\mathbf{I} - \\mathbf{K}_k\\mathbf{H}_k)^T + \\mathbf{K}_k\\mathbf{R}_{\\text{eff}}\\mathbf{K}_k^T$ | Joseph form for numerical stability & positive semi-definiteness |"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 3. Generic `ExtendedKalmanFilter` Class Implementation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def wrap_angle(angle):\n",
                "    \"\"\"Wraps an angle or array of angles to the interval [-pi, pi].\"\"\"\n",
                "    return (angle + np.pi) % (2 * np.pi) - np.pi\n",
                "\n",
                "class ExtendedKalmanFilter:\n",
                "    \"\"\"Generic Multi-Dimensional Extended Kalman Filter (EKF).\n",
                "    \n",
                "    Keeps all filter methods completely generic, accepting arbitrary nonlinear functions\n",
                "    f(x, u), h(x), Jacobians F, L, H, M, and noise covariance matrices Q, R.\n",
                "    \"\"\"\n",
                "    \n",
                "    def __init__(self, x0: np.ndarray, P0: np.ndarray):\n",
                "        self.x = np.asarray(x0, dtype=np.float64).reshape(-1, 1)\n",
                "        self.P = np.asarray(P0, dtype=np.float64)\n",
                "        self.n = self.x.shape[0]\n",
                "        \n",
                "        self.latest_innovation = None\n",
                "        self.latest_innovation_cov = None\n",
                "        self.latest_gain = None\n",
                "        \n",
                "    def predict(self, f_func, F_jac: np.ndarray, Q: np.ndarray, \n",
                "                u: np.ndarray = None, L_jac: np.ndarray = None):\n",
                "        \"\"\"Executes the EKF State and Covariance Prediction Step:\n",
                "            check_x = f(hat_x, u)\n",
                "            check_P = F * hat_P * F^T + L * Q * L^T\n",
                "        \"\"\"\n",
                "        if u is not None:\n",
                "            self.x = f_func(self.x, u).reshape(-1, 1)\n",
                "        else:\n",
                "            self.x = f_func(self.x).reshape(-1, 1)\n",
                "            \n",
                "        if L_jac is None:\n",
                "            L_jac = np.eye(self.n)\n",
                "            \n",
                "        self.P = F_jac @ self.P @ F_jac.T + L_jac @ Q @ L_jac.T\n",
                "        return self.x.copy(), self.P.copy()\n",
                "        \n",
                "    def update(self, y: np.ndarray, h_func, H_jac: np.ndarray, R: np.ndarray, \n",
                "               M_jac: np.ndarray = None, angle_indices: list = None):\n",
                "        \"\"\"Executes the EKF Measurement Correction Step:\n",
                "            1. Innovation:            nu = y - h(check_x)\n",
                "            2. Innovation Covariance: S  = H * check_P * H^T + M * R * M^T\n",
                "            3. Kalman Gain:           K  = check_P * H^T * inv(S)\n",
                "            4. Corrected State:       hat_x = check_x + K * nu\n",
                "            5. Corrected Covariance:  hat_P = (I - K*H) * check_P * (I - K*H)^T + K * R_eff * K^T (Joseph form)\n",
                "        \"\"\"\n",
                "        y_vec = np.asarray(y, dtype=np.float64).reshape(-1, 1)\n",
                "        m = y_vec.shape[0]\n",
                "        if M_jac is None:\n",
                "            M_jac = np.eye(m)\n",
                "            \n",
                "        # 1. Innovation residual\n",
                "        y_pred = h_func(self.x).reshape(-1, 1)\n",
                "        nu = y_vec - y_pred\n",
                "        if angle_indices is not None:\n",
                "            for idx in angle_indices:\n",
                "                nu[idx, 0] = wrap_angle(nu[idx, 0])\n",
                "                \n",
                "        # 2. Innovation Covariance\n",
                "        R_eff = M_jac @ R @ M_jac.T\n",
                "        S = H_jac @ self.P @ H_jac.T + R_eff\n",
                "        \n",
                "        # 3. Optimal Kalman Gain\n",
                "        K = self.P @ H_jac.T @ np.linalg.inv(S)\n",
                "        \n",
                "        # 4. Posterior State Estimate\n",
                "        self.x = self.x + K @ nu\n",
                "        \n",
                "        # 5. Posterior Covariance (Joseph form for numerical stability)\n",
                "        I_KH = np.eye(self.n) - K @ H_jac\n",
                "        self.P = I_KH @ self.P @ I_KH.T + K @ R_eff @ K.T\n",
                "        self.P = 0.5 * (self.P + self.P.T)\n",
                "        \n",
                "        self.latest_innovation = nu\n",
                "        self.latest_innovation_cov = S\n",
                "        self.latest_gain = K\n",
                "        \n",
                "        return self.x.copy(), self.P.copy()\n",
                "\n",
                "print('Generic ExtendedKalmanFilter class compiled successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 4. Automotive Application: 2D Polar Radar Tracking\n",
                "\n",
                "Now we instantiate our generic filter for an automotive tracking task:\n",
                "- **Vehicle State**: $\\mathbf{x} = [p_x, p_y, v, \\theta]^T$ (Position East, Position North, Speed, Heading Angle).\n",
                "- **Control Input**: $\\mathbf{u} = [a, \\omega]^T$ (Linear Acceleration, Yaw Rate).\n",
                "- **Polar Radar Sensor**: $\\mathbf{y} = [r, \\phi]^T$ measuring Range $r$ and Azimuth Bearing $\\phi$ from origin $(0,0)$.\n",
                "\n",
                "### 📐 Kinematic Motion Model $\\mathbf{f}(\\mathbf{x}, \\mathbf{u})$ & Jacobian $\\mathbf{F}$:\n",
                "$$\\mathbf{f}(\\mathbf{x}, \\mathbf{u}) = \\begin{bmatrix} p_x + v \\cos(\\theta)\\Delta t \\\\ p_y + v \\sin(\\theta)\\Delta t \\\\ v + a \\Delta t \\\\ \\operatorname{wrap}(\\theta + \\omega \\Delta t) \\end{bmatrix}, \\quad\n",
                "\\mathbf{F} = \\begin{bmatrix} 1 & 0 & \\cos(\\theta)\\Delta t & -v \\sin(\\theta)\\Delta t \\\\ 0 & 1 & \\sin(\\theta)\\Delta t & v \\cos(\\theta)\\Delta t \\\\ 0 & 0 & 1 & 0 \\\\ 0 & 0 & 0 & 1 \\end{bmatrix}$$\n",
                "\n",
                "### 📡 Polar Radar Observation Model $\\mathbf{h}(\\mathbf{x})$ & Jacobian $\\mathbf{H}$:\n",
                "$$\\mathbf{h}(\\mathbf{x}) = \\begin{bmatrix} r \\\\ \\phi \\end{bmatrix} = \\begin{bmatrix} \\sqrt{p_x^2 + p_y^2} \\\\ \\operatorname{atan2}(p_y, p_x) \\end{bmatrix}, \\quad\n",
                "\\mathbf{H} = \\begin{bmatrix} \\frac{p_x}{r} & \\frac{p_y}{r} & 0 & 0 \\\\ -\\frac{p_y}{r^2} & \\frac{p_x}{r^2} & 0 & 0 \\end{bmatrix}, \\quad r = \\sqrt{p_x^2 + p_y^2}$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def motion_model(x, u, dt):\n",
                "    \"\"\"Nonlinear kinematic state transition f(x, u).\"\"\"\n",
                "    px, py, v, theta = x.flatten()\n",
                "    a, omega = u.flatten()\n",
                "    px_next = px + v * np.cos(theta) * dt\n",
                "    py_next = py + v * np.sin(theta) * dt\n",
                "    v_next = v + a * dt\n",
                "    theta_next = wrap_angle(theta + omega * dt)\n",
                "    return np.array([px_next, py_next, v_next, theta_next]).reshape(-1, 1)\n",
                "\n",
                "def get_F_jacobian(x, dt):\n",
                "    \"\"\"Computes 4x4 state transition Jacobian F = df/dx.\"\"\"\n",
                "    px, py, v, theta = x.flatten()\n",
                "    return np.array([\n",
                "        [1.0, 0.0, np.cos(theta) * dt, -v * np.sin(theta) * dt],\n",
                "        [0.0, 1.0, np.sin(theta) * dt,  v * np.cos(theta) * dt],\n",
                "        [0.0, 0.0, 1.0,                 0.0],\n",
                "        [0.0, 0.0, 0.0,                 1.0]\n",
                "    ])\n",
                "\n",
                "def measurement_model(x):\n",
                "    \"\"\"Polar radar measurement function h(x) = [r, phi]^T.\"\"\"\n",
                "    px, py = x[0, 0], x[1, 0]\n",
                "    r = np.sqrt(px**2 + py**2)\n",
                "    phi = np.arctan2(py, px)\n",
                "    return np.array([r, phi]).reshape(-1, 1)\n",
                "\n",
                "def get_H_jacobian(x):\n",
                "    \"\"\"Computes 2x4 measurement Jacobian H = dh/dx.\"\"\"\n",
                "    px, py = x[0, 0], x[1, 0]\n",
                "    r2 = max(px**2 + py**2, 1e-8)\n",
                "    r = np.sqrt(r2)\n",
                "    return np.array([\n",
                "        [px / r,       py / r,       0.0, 0.0],\n",
                "        [-py / r2,     px / r2,      0.0, 0.0]\n",
                "    ])\n",
                "\n",
                "print('Automotive motion and radar sensor models defined.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 5. Physical Formulation of $\\mathbf{Q}$ and $\\mathbf{R}$ Matrices\n",
                "\n",
                "### 🔹 Process Noise Covariance $\\mathbf{Q}$ (State $\\mathbf{x} = [p_x, p_y, v, \\theta]^T$):\n",
                "* $\\sigma_{p_x} = 0.05\\text{ m}, \\sigma_{p_y} = 0.05\\text{ m}$: Models tire lateral slip and 1st-order discrete kinematic truncation over $\\Delta t = 0.1\\text{ s}$.\n",
                "* $\\sigma_v = 0.1\\text{ m/s}$: Models engine throttle response lag, road slope variations, and braking jitter.\n",
                "* $\\sigma_\\theta = 0.02\\text{ rad} \\approx 1.15^\\circ$: Models steering backlash, crosswinds, and road bank angle.\n",
                "$$\\mathbf{Q} = \\operatorname{diag}(0.05^2, 0.05^2, 0.1^2, 0.02^2)$$\n",
                "\n",
                "### 🔹 Measurement Noise Covariance $\\mathbf{R}$ (Radar $\\mathbf{y} = [r, \\phi]^T$):\n",
                "* $\\sigma_r = 0.5\\text{ m}$: Automotive mmWave radar range ToF measurement precision.\n",
                "* $\\sigma_\\phi = 1.0^\\circ = 0.0175\\text{ rad}$: Radar antenna array beam azimuth resolution.\n",
                "$$\\mathbf{R} = \\operatorname{diag}(0.5^2, (\\operatorname{deg2rad}(1.0))^2)$$"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 6. Simulation & Filter Execution"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dt = 0.1\n",
                "T_total = 60.0\n",
                "N_steps = int(T_total / dt)\n",
                "time = np.linspace(0, T_total, N_steps)\n",
                "\n",
                "x_true_all = np.zeros((4, N_steps))\n",
                "x_true = np.array([10.0, 5.0, 15.0, 0.0]).reshape(-1, 1)\n",
                "\n",
                "Q = np.diag([0.05**2, 0.05**2, 0.1**2, 0.02**2])\n",
                "R = np.diag([0.5**2, np.deg2rad(1.0)**2])\n",
                "\n",
                "# Generate true kinematics and radar measurements\n",
                "measurements = []\n",
                "for k in range(N_steps):\n",
                "    t = time[k]\n",
                "    a_cmd = 0.5 * np.sin(0.1 * t)\n",
                "    omega_cmd = 0.1 * np.cos(0.08 * t)\n",
                "    u = np.array([a_cmd, omega_cmd]).reshape(-1, 1)\n",
                "    w = np.random.multivariate_normal(np.zeros(4), Q).reshape(-1, 1)\n",
                "    x_true = motion_model(x_true, u, dt) + w\n",
                "    x_true[3, 0] = wrap_angle(x_true[3, 0])\n",
                "    x_true_all[:, k] = x_true.flatten()\n",
                "    v_noise = np.random.multivariate_normal(np.zeros(2), R).reshape(-1, 1)\n",
                "    y = measurement_model(x_true) + v_noise\n",
                "    y[1, 0] = wrap_angle(y[1, 0])\n",
                "    measurements.append((u, y))\n",
                "\n",
                "# Initialize Generic Extended Kalman Filter\n",
                "x0_est = np.array([8.0, 3.0, 10.0, np.deg2rad(10.0)]).reshape(-1, 1)\n",
                "P0_est = np.diag([5.0**2, 5.0**2, 5.0**2, np.deg2rad(20.0)**2])\n",
                "ekf = ExtendedKalmanFilter(x0_est, P0_est)\n",
                "\n",
                "x_est_all = np.zeros((4, N_steps))\n",
                "P_diag_all = np.zeros((4, N_steps))\n",
                "nees_all = np.zeros(N_steps)\n",
                "\n",
                "for k in range(N_steps):\n",
                "    u, y = measurements[k]\n",
                "    # 1. Prediction\n",
                "    F = get_F_jacobian(ekf.x, dt)\n",
                "    ekf.predict(lambda x, u_in: motion_model(x, u_in, dt), F, Q, u=u)\n",
                "    \n",
                "    # 2. Measurement Update\n",
                "    H = get_H_jacobian(ekf.x)\n",
                "    ekf.update(y, measurement_model, H, R, angle_indices=[1])\n",
                "    \n",
                "    # Store trajectory and metrics\n",
                "    x_est_all[:, k] = ekf.x.flatten()\n",
                "    P_diag_all[:, k] = np.diag(ekf.P)\n",
                "    err = x_true_all[:, k:k+1] - ekf.x\n",
                "    err[3, 0] = wrap_angle(err[3, 0])\n",
                "    nees_all[k] = (err.T @ np.linalg.inv(ekf.P) @ err).item()\n",
                "\n",
                "rmse_pos = np.sqrt(np.mean((x_true_all[0, :] - x_est_all[0, :])**2 + (x_true_all[1, :] - x_est_all[1, :])**2))\n",
                "print(f'EKF execution complete. Position RMSE: {rmse_pos:.3f} m | Mean NEES: {np.mean(nees_all):.2f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 7. Plotly Interactive Visualizations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig = make_subplots(\n",
                "    rows=2, cols=2,\n",
                "    subplot_titles=(\n",
                "        '2D Vehicle Trajectory Tracking',\n",
                "        'Position Error vs. 3-Sigma Bound',\n",
                "        'Heading Error vs. 3-Sigma Bounds',\n",
                "        'Filter Consistency: NEES (dim=4)'\n",
                "    )\n",
                ")\n",
                "\n",
                "radar_x = [m[1][0, 0] * np.cos(m[1][1, 0]) for m in measurements]\n",
                "radar_y = [m[1][0, 0] * np.sin(m[1][1, 0]) for m in measurements]\n",
                "\n",
                "fig.add_trace(go.Scatter(x=x_true_all[0, :], y=x_true_all[1, :], mode='lines', name='Ground Truth', line=dict(color='black', width=3)), row=1, col=1)\n",
                "fig.add_trace(go.Scatter(x=radar_x[::4], y=radar_y[::4], mode='markers', name='Radar Points', marker=dict(color='red', size=4, opacity=0.4)), row=1, col=1)\n",
                "fig.add_trace(go.Scatter(x=x_est_all[0, :], y=x_est_all[1, :], mode='lines', name='EKF Estimate', line=dict(color='blue', width=2, dash='dash')), row=1, col=1)\n",
                "\n",
                "pos_err = np.sqrt((x_true_all[0, :] - x_est_all[0, :])**2 + (x_true_all[1, :] - x_est_all[1, :])**2)\n",
                "sigma_pos = 3.0 * np.sqrt(P_diag_all[0, :] + P_diag_all[1, :])\n",
                "fig.add_trace(go.Scatter(x=time, y=pos_err, mode='lines', name='Position Error [m]', line=dict(color='blue', width=1.5)), row=1, col=2)\n",
                "fig.add_trace(go.Scatter(x=time, y=sigma_pos, mode='lines', name='+3-Sigma Bound [m]', line=dict(color='red', dash='dot', width=1.5)), row=1, col=2)\n",
                "\n",
                "heading_err = wrap_angle(x_true_all[3, :] - x_est_all[3, :])\n",
                "sigma_heading = 3.0 * np.sqrt(P_diag_all[3, :])\n",
                "fig.add_trace(go.Scatter(x=time, y=np.rad2deg(heading_err), mode='lines', name='Heading Error [deg]', line=dict(color='green', width=1.5)), row=2, col=1)\n",
                "fig.add_trace(go.Scatter(x=time, y=np.rad2deg(sigma_heading), mode='lines', name='+3-Sigma [deg]', line=dict(color='red', dash='dot', width=1.5)), row=2, col=1)\n",
                "fig.add_trace(go.Scatter(x=time, y=-np.rad2deg(sigma_heading), mode='lines', name='-3-Sigma [deg]', line=dict(color='red', dash='dot', width=1.5), showlegend=False), row=2, col=1)\n",
                "\n",
                "chi2_low, chi2_high = chi2.interval(0.95, df=4)\n",
                "fig.add_trace(go.Scatter(x=time, y=nees_all, mode='lines', name='NEES', line=dict(color='black', width=1.2)), row=2, col=2)\n",
                "fig.add_trace(go.Scatter(x=time, y=[chi2_high]*N_steps, mode='lines', name='95% Upper (9.49)', line=dict(color='red', dash='dash')), row=2, col=2)\n",
                "fig.add_trace(go.Scatter(x=time, y=[chi2_low]*N_steps, mode='lines', name='95% Lower (0.48)', line=dict(color='orange', dash='dash')), row=2, col=2)\n",
                "\n",
                "fig.update_layout(title_text='Day 2 EKF: 2D Radar Tracking & Performance Diagnostics', template='plotly_white', height=750, width=1050)\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 8. Concept Questions\n",
                "1. Why is the state prediction step computed using the full nonlinear motion function $\\mathbf{f}(\\cdot)$ rather than the linear matrix $\\mathbf{F}$?\n",
                "2. What causes NEES to consistently exceed the $95\\%$ upper bound if process noise $\\mathbf{Q}$ is chosen too small?"
            ]
        }
    ]

    # -------------------------------------------------------------
    # EKF STUDENT NOTEBOOK
    # -------------------------------------------------------------
    cells_stud = [
        cells_inst[0], # Title
        cells_inst[1], # Env setup md
        cells_inst[2], # Env setup code
        cells_inst[3], # Math Foundation md
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 3. Implementing the `ExtendedKalmanFilter` Class\n",
                "\n",
                "### 📝 Exercise 1: Implement the Generic EKF Class"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def wrap_angle(angle):\n",
                "    \"\"\"Wraps an angle or array of angles to the interval [-pi, pi].\"\"\"\n",
                "    return (angle + np.pi) % (2 * np.pi) - np.pi\n",
                "\n",
                "class ExtendedKalmanFilter:\n",
                "    \"\"\"Generic Multi-Dimensional Extended Kalman Filter (EKF).\"\"\"\n",
                "    \n",
                "    def __init__(self, x0: np.ndarray, P0: np.ndarray):\n",
                "        self.x = np.asarray(x0, dtype=np.float64).reshape(-1, 1)\n",
                "        self.P = np.asarray(P0, dtype=np.float64)\n",
                "        self.n = self.x.shape[0]\n",
                "        \n",
                "        self.latest_innovation = None\n",
                "        self.latest_innovation_cov = None\n",
                "        self.latest_gain = None\n",
                "        \n",
                "    def predict(self, f_func, F_jac: np.ndarray, Q: np.ndarray, \n",
                "                u: np.ndarray = None, L_jac: np.ndarray = None):\n",
                "        # -------------------------------------------------------------------------\n",
                "        # TODO 1.1: Complete EKF state and covariance prediction equations\n",
                "        # 1. State propagation: check_x = f(hat_x, u) (or f(hat_x) if u is None)\n",
                "        # 2. Covariance propagation: check_P = F * hat_P * F^T + L * Q * L^T (L=I if None)\n",
                "        # -------------------------------------------------------------------------\n",
                "        pass  # <-- YOUR CODE HERE\n",
                "        \n",
                "    def update(self, y: np.ndarray, h_func, H_jac: np.ndarray, R: np.ndarray, \n",
                "               M_jac: np.ndarray = None, angle_indices: list = None):\n",
                "        # -------------------------------------------------------------------------\n",
                "        # TODO 1.2: Complete EKF measurement update equations\n",
                "        # 1. Innovation: nu = y - h(check_x) (wrap angles if angle_indices specified)\n",
                "        # 2. Innovation Covariance: S = H * check_P * H^T + M * R * M^T (M=I if None)\n",
                "        # 3. Kalman Gain: K = check_P * H^T * inv(S)\n",
                "        # 4. State Correction: hat_x = check_x + K * nu\n",
                "        # 5. Covariance Correction: Joseph form (I - K*H) * check_P * (I - K*H)^T + K*R_eff*K^T\n",
                "        # -------------------------------------------------------------------------\n",
                "        pass  # <-- YOUR CODE HERE\n"
            ]
        },
        cells_inst[5], # Automotive radar md
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def motion_model(x, u, dt):\n",
                "    \"\"\"Nonlinear kinematic state transition f(x, u).\"\"\"\n",
                "    # TODO 2.1: Implement motion model\n",
                "    pass\n",
                "\n",
                "def get_F_jacobian(x, dt):\n",
                "    \"\"\"Computes 4x4 state transition Jacobian F.\"\"\"\n",
                "    # TODO 2.2: Implement F Jacobian\n",
                "    pass\n",
                "\n",
                "def measurement_model(x):\n",
                "    \"\"\"Computes range and bearing [r, phi]^T from Cartesian state.\"\"\"\n",
                "    # TODO 2.3: Implement measurement model\n",
                "    pass\n",
                "\n",
                "def get_H_jacobian(x):\n",
                "    \"\"\"Computes 2x4 measurement Jacobian H.\"\"\"\n",
                "    # TODO 2.4: Implement H Jacobian\n",
                "    pass\n"
            ]
        },
        cells_inst[7], # Q and R physical formulation md
        cells_inst[8], # Sim md
        cells_inst[9], # Sim code
        cells_inst[10], # Plotly md
        cells_inst[11], # Plotly code
        cells_inst[12], # Questions md
    ]

    nb_inst = {"cells": cells_inst, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    nb_stud = {"cells": cells_stud, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

    with open("position_class/Day_02_Extended_Kalman_Filter_instructor.ipynb", "w") as f:
        json.dump(nb_inst, f, indent=1)
    with open("position_class/Day_02_Extended_Kalman_Filter_student.ipynb", "w") as f:
        json.dump(nb_stud, f, indent=1)
    print("Extended Kalman Filter notebooks written successfully.")

generate_ekf_notebooks()
