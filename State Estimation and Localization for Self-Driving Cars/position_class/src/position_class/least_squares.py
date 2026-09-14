"""Least Squares and Weighted Least Squares module for Autonomous Vehicle Estimation.

Contains solvers, matrix derivations, and automotive calibration examples:
1. Electrical resistance (Ohm's law)
2. Wheel odometry scale factor / effective tire radius calibration
3. 1D/2D Kinematic initial state estimation (p0, v0)
4. LiDAR ground plane / road slope estimation (a, b, c)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
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


def lidar_plane_fitting_example() -> Tuple[np.ndarray, np.ndarray]:
    """Example 4: Fit 3D Road surface plane z = a*x + b*y + c to LiDAR ground points.
    
    State: x = [a, b, c]^T
    H = [[x_1, y_1, 1], ..., [x_N, y_N, 1]]
    y = [z_1, ..., z_N]^T
    """
    # Sample LiDAR (x, y) ground cluster
    np.random.seed(42)
    N = 20
    x_pts = np.random.uniform(2.0, 20.0, N)
    y_pts = np.random.uniform(-5.0, 5.0, N)
    # True road slope: a=0.02 (pitch), b=-0.01 (roll), c=-1.50 (sensor height)
    z_pts = 0.02 * x_pts - 0.01 * y_pts - 1.50 + np.random.normal(0, 0.03, N)
    
    H = np.column_stack([x_pts, y_pts, np.ones(N)])
    solver = BatchLeastSquares()
    res = solver.fit(H=H, y=z_pts)
    return res.x_hat.ravel(), res.covariance
