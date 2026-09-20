"""Least Squares and Weighted Least Squares module for Autonomous Vehicle Estimation.

Contains solvers, matrix derivations, and automotive calibration examples:
1. Electrical resistance (Ohm's law)
2. Wheel odometry scale factor / effective tire radius calibration
3. 1D/2D Kinematic initial state estimation (p0, v0)
4. LiDAR ground plane / road slope estimation (a, b, c)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np


@dataclass
class LeastSquaresResult:
    """Container for Least Squares estimation results."""
    x_hat: np.ndarray          # Estimated state / parameter vector (n x 1)
    covariance: np.ndarray     # Parameter error covariance P = (H^T R^-1 H)^-1 (n x n)
    residuals: np.ndarray      # Measurement residuals e = y - H x_hat (m x 1)
    cost: float                # Residual sum of squares e^T R^-1 e


class BatchLeastSquares:
    """Batch Linear Least Squares and Weighted Least Squares (WLS) Solver.
    
    Model:
        y = H * x + v,   v ~ N(0, R)
        
    Normal Equations (BLUE):
        x_hat = (H^T * R^-1 * H)^-1 * H^T * R^-1 * y
        P     = (H^T * R^-1 * H)^-1
    """

    def __init__(self, R: Optional[np.ndarray] = None):
        """Initialize solver with optional measurement noise covariance R.
        
        Args:
            R: Measurement covariance matrix (m x m) or variance vector (m,).
               If None, standard unweighted OLS (R = I) is used.
        """
        self.R = R

    def fit(self, H: np.ndarray, y: np.ndarray, R: Optional[np.ndarray] = None) -> LeastSquaresResult:
        """Solve for parameters x_hat given observation matrix H and measurement vector y.
        
        Args:
            H: Observation / Regressor matrix (m x n), m >= n
            y: Measurement column vector (m x 1) or 1D array (m,)
            R: Optional override for measurement noise covariance (m x m)
            
        Returns:
            LeastSquaresResult containing x_hat, covariance P, residuals, and cost.
        """
        H = np.asarray(H, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        m, n = H.shape
        if m < n:
            raise ValueError(f"System is underdetermined: {m} measurements < {n} parameters.")

        cov_R = R if R is not None else self.R

        if cov_R is None:
            # Ordinary Least Squares (OLS): x_hat = (H^T H)^-1 H^T y
            Ht_H = H.T @ H
            P = np.linalg.inv(Ht_H)
            x_hat = P @ H.T @ y
            residuals = y - H @ x_hat
            cost = float((residuals.T @ residuals).item())
        else:
            # Weighted Least Squares (WLS): x_hat = (H^T R^-1 H)^-1 H^T R^-1 y
            cov_R = np.asarray(cov_R, dtype=float)
            if cov_R.ndim == 1:
                cov_R = np.diag(cov_R)
            
            R_inv = np.linalg.inv(cov_R)
            Ht_Rinv = H.T @ R_inv
            P = np.linalg.inv(Ht_Rinv @ H)
            x_hat = P @ Ht_Rinv @ y
            residuals = y - H @ x_hat
            cost = float((residuals.T @ R_inv @ residuals).item())

        return LeastSquaresResult(
            x_hat=x_hat,
            covariance=P,
            residuals=residuals,
            cost=cost
        )


class RecursiveLeastSquares:
    """Online Recursive Least Squares (RLS) Parameter Estimator.
    
    Streaming Measurement Model:
        y_k = H_k * x + v_k,   v_k ~ N(0, R_k)
        
    Recursive Equations (BLUE with prior):
        Innovation:           nu_k = y_k - H_k * x_hat_{k-1}
        Innovation Cov:       S_k  = H_k * P_{k-1} * H_k^T + R_k
        Estimator Gain:       K_k  = P_{k-1} * H_k^T * S_k^-1
        State Update:         x_hat_k = x_hat_{k-1} + K_k * nu_k
        Covariance Update:    P_k  = (1 / lambda) * (I - K_k * H_k) * P_{k-1}
        
    Initialization Strategies:
        1. Batch Prior (Deterministic): Use first n samples to compute x0 = (H0^T R0^-1 H0)^-1 H0^T R0^-1 y0 and P0 = (H0^T R0^-1 H0)^-1.
        2. Diffuse Prior (Uninformative): Set x0 = nominal_guess (or 0), and P0 = alpha * I with alpha >> R_meas.
    """

    def __init__(
        self,
        n: int,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        forgetting_factor: float = 1.0
    ):
        self.n = n
        self.forgetting_factor = float(forgetting_factor)
        
        if x0 is not None:
            self.x = np.asarray(x0, dtype=np.float64).reshape(n, 1)
        else:
            self.x = np.zeros((n, 1), dtype=np.float64)
            
        if P0 is not None:
            self.P = np.asarray(P0, dtype=np.float64).reshape(n, n)
        else:
            # Default diffuse prior
            self.P = np.eye(n, dtype=np.float64) * 1000.0

    def initialize_from_prior(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Initialize RLS with explicit prior mean and covariance."""
        self.x = np.asarray(x0, dtype=np.float64).reshape(self.n, 1)
        self.P = np.asarray(P0, dtype=np.float64).reshape(self.n, self.n)

    def initialize_from_batch(
        self,
        H0: np.ndarray,
        y0: np.ndarray,
        R0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Initializes x0 and P0 from an initial minimal batch of n measurements.
        
        Exact BLUE formulation:
            P0 = (H0^T * R0^-1 * H0)^-1
            x0 = P0 * H0^T * R0^-1 * y0
        """
        solver = BatchLeastSquares(R=R0)
        res = solver.fit(H=H0, y=y0)
        self.x = res.x_hat.copy()
        self.P = res.covariance.copy()
        return self.x.copy(), self.P.copy()

    def update(
        self,
        H_k: np.ndarray,
        y_k: np.ndarray,
        R_k: Optional[np.ndarray] = None
    ) -> LeastSquaresResult:
        """Process a single streaming measurement observation (H_k, y_k).
        
        Args:
            H_k: (m x n) Regressor matrix for current step k.
            y_k: (m x 1) or scalar measurement at step k.
            R_k: (m x m) measurement noise covariance (defaults to I if None).
        """
        H_k = np.asarray(H_k, dtype=np.float64)
        if H_k.ndim == 1:
            H_k = H_k.reshape(1, self.n)
            
        m = H_k.shape[0]
        y_k = np.asarray(y_k, dtype=np.float64).reshape(m, 1)
        
        if R_k is None:
            R_mat = np.eye(m, dtype=np.float64)
        else:
            R_mat = np.asarray(R_k, dtype=np.float64)
            if R_mat.ndim == 1:
                R_mat = np.diag(R_mat)
            elif R_mat.ndim == 0:
                R_mat = np.array([[float(R_mat)]], dtype=np.float64)

        # 1. Innovation
        nu = y_k - H_k @ self.x
        
        # 2. Innovation Covariance
        S = H_k @ self.P @ H_k.T + R_mat
        
        # 3. Estimator Gain
        K = self.P @ H_k.T @ np.linalg.inv(S)
        
        # 4. State Update
        self.x = self.x + K @ nu
        
        # 5. Covariance Update with optional forgetting factor lambda
        I_KH = np.eye(self.n, dtype=np.float64) - K @ H_k
        self.P = (1.0 / self.forgetting_factor) * (I_KH @ self.P)
        
        cost = float((nu.T @ np.linalg.inv(S) @ nu).item())
        
        return LeastSquaresResult(
            x_hat=self.x.copy(),
            covariance=self.P.copy(),
            residuals=nu,
            cost=cost
        )


# ==============================================================================
# Practical Automotive & Robotics Examples
# ==============================================================================

def ohms_law_example() -> Tuple[float, float]:
    """Example 1: Determine resistance R from current and voltage (Excersice.ipynb).
    
    Model: V_i = I_i * R + v_i
    H = [I_1, I_2, ..., I_N]^T
    y = [V_1, V_2, ..., V_N]^T
    """
    I = np.array([[0.2, 0.3, 0.4, 0.5, 0.6]]).T
    V = np.array([[1.23, 1.38, 2.06, 2.47, 3.17]]).T
    
    solver = BatchLeastSquares()
    res = solver.fit(H=I, y=V)
    R_est = float(res.x_hat[0, 0])
    R_std = float(np.sqrt(res.covariance[0, 0]))
    return R_est, R_std


def wheel_odometry_calibration_example() -> Tuple[float, float]:
    """Example 2: Calibrate effective wheel radius r_eff for autonomous car odometry.
    
    Model: v_gps = r_eff * omega_wheel + v
    """
    # Simulated wheel encoder angular velocities (rad/s)
    omega = np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]).T
    # True wheel radius = 0.33 m with measurement noise
    v_gps = np.array([[3.28, 6.64, 9.87, 13.25, 16.48, 19.82]]).T
    
    # Measurement noise variance for GPS speed (sigma = 0.05 m/s)
    R_cov = np.diag([0.05**2] * len(omega))
    
    solver = BatchLeastSquares(R=R_cov)
    res = solver.fit(H=omega, y=v_gps)
    r_eff = float(res.x_hat[0, 0])
    r_std = float(np.sqrt(res.covariance[0, 0]))
    return r_eff, r_std


def kinematic_position_velocity_example() -> Tuple[np.ndarray, np.ndarray]:
    """Example 3: Estimate 1D initial vehicle position p0 and velocity v0 from distance pings.
    
    Model: d_i = p0 + v0 * t_i + v_i
    State: x = [p0, v0]^T
    H = [[1, t_1], [1, t_2], ..., [1, t_N]]
    """
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    distances = np.array([[10.2, 22.1, 34.5, 46.8, 58.9, 71.3]]).T  # True p0=10.0, v0=12.0
    
    H = np.column_stack([np.ones_like(timestamps), timestamps])
    solver = BatchLeastSquares()
    res = solver.fit(H=H, y=distances)
    return res.x_hat.ravel(), res.covariance


def fit_lidar_ground_plane(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    z_pts: np.ndarray,
    sigma_z: float = 0.03
) -> Dict[str, Union[np.ndarray, float]]:
    """Fits a 3D road surface plane z = a*x + b*y + c to LiDAR point cloud.
    
    Coordinate System (ISO 8855 / ROS REP 103):
        +X : Forward (Longitudinal)
        +Y : Left (Lateral)
        +Z : Up (Elevation)
        
    Model:
        z_i = a * x_i + b * y_i + c + v_i,  v_i ~ N(0, sigma_z^2)
        
    Returns dictionary with:
        - a: longitudinal slope gradient (tan pitch)
        - b: lateral slope gradient (tan roll)
        - c: z-intercept (approx -sensor_height)
        - covariance: 3x3 covariance matrix P
        - pitch_rad, pitch_deg: road longitudinal slope angle
        - roll_rad, roll_deg: road lateral cross-slope angle
        - sensor_height_m: orthogonal distance from sensor to plane
        - normal_vector: unit normal vector of the ground plane [nx, ny, nz]^T
        - rmse_m: root mean square vertical fitting residual
    """
    x_arr = np.asarray(x_pts, dtype=np.float64).flatten()
    y_arr = np.asarray(y_pts, dtype=np.float64).flatten()
    z_arr = np.asarray(z_pts, dtype=np.float64).flatten()
    
    N = len(x_arr)
    if N < 3:
        raise ValueError("At least 3 non-collinear points required to fit a 3D plane.")
        
    H = np.column_stack([x_arr, y_arr, np.ones(N)])
    R_cov = np.eye(N) * (sigma_z ** 2)
    
    solver = BatchLeastSquares(R=R_cov)
    res = solver.fit(H=H, y=z_arr)
    
    a, b, c = float(res.x_hat[0, 0]), float(res.x_hat[1, 0]), float(res.x_hat[2, 0])
    
    # Plane normal n = [-a, -b, 1] / sqrt(a^2 + b^2 + 1)
    norm_factor = np.sqrt(a**2 + b**2 + 1.0)
    normal = np.array([-a, -b, 1.0]) / norm_factor
    
    # Sensor mounting height (perpendicular distance from origin to plane)
    sensor_height = abs(c) / norm_factor
    
    # Pitch (around Y axis) and Roll (around X axis)
    pitch_rad = np.arctan(a)
    roll_rad = np.arctan(-b)
    
    rmse = float(np.sqrt(np.mean(res.residuals ** 2)))
    
    return {
        "params": np.array([a, b, c]),
        "covariance": res.covariance,
        "a": a,
        "b": b,
        "c": c,
        "pitch_rad": pitch_rad,
        "pitch_deg": float(np.degrees(pitch_rad)),
        "roll_rad": roll_rad,
        "roll_deg": float(np.degrees(roll_rad)),
        "sensor_height_m": float(sensor_height),
        "normal_vector": normal,
        "rmse_m": rmse,
        "residuals": res.residuals.flatten()
    }


def lidar_plane_fitting_example() -> Tuple[np.ndarray, np.ndarray]:
    """Example 4: Fit 3D Road surface plane z = a*x + b*y + c to LiDAR ground points.
    
    Simulates a ground LiDAR cluster ahead of the car:
    x in [2.0, 25.0] m (forward), y in [-4.0, 4.0] m (lateral)
    True parameters: a = 0.02 (1.15 deg uphill), b = -0.01 (-0.57 deg roll), c = -1.70 m (sensor height)
    """
    np.random.seed(42)
    N = 100
    x_pts = np.random.uniform(2.0, 25.0, N)
    y_pts = np.random.uniform(-4.0, 4.0, N)
    
    # True road parameters
    a_true, b_true, c_true = 0.02, -0.01, -1.70
    sigma_noise = 0.03
    z_pts = a_true * x_pts + b_true * y_pts + c_true + np.random.normal(0, sigma_noise, N)
    
    res = fit_lidar_ground_plane(x_pts, y_pts, z_pts, sigma_z=sigma_noise)
    return res["params"], res["covariance"]

