"""Discrete Linear Kalman Filter (LKF) module for Autonomous Vehicle Estimation.

Provides:
- LinearKalmanFilter: Vectorized LKF tracking class.
- 1D & 2D Vehicle Kinematic Tracking simulation generators.
- Rigorous mathematical documentation of F, G, H, Q, R, P, K.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class KalmanState:
    """Represents the estimated Gaussian state of a system N(x, P)."""
    x: np.ndarray  # Mean state vector (n x 1)
    P: np.ndarray  # Error covariance matrix (n x n)

    @property
    def mean(self) -> np.ndarray:
        return self.x.flatten()

    @property
    def covariance(self) -> np.ndarray:
        return self.P


class LinearKalmanFilter:
    """Discrete-time Linear Kalman Filter (LKF).
    
    System Dynamics:
        x_k = F_{k-1} * x_{k-1} + G_{k-1} * u_{k-1} + w_{k-1},   w ~ N(0, Q)
    
    Measurement Model:
        y_k = H_k * x_k + v_k,                                   v ~ N(0, R)
        
    Mathematical Matrices & Cross-Discipline Equivalents:
        F (State Transition)    <-> A in Control Theory, Phi in Astrodynamics
        G (Control Input Gain)  <-> B in Control Theory, Gamma in Astrodynamics
        H (Measurement Matrix)  <-> C in Control Theory, X (Design matrix) in ML
        Q (Process Noise Cov)   <-> W or Sigma_w (Plant uncertainty)
        R (Sensor Noise Cov)    <-> V or Sigma_v (Sensor variance)
        P (Error Covariance)    <-> Sigma_xx (Estimation confidence)
        K (Kalman Gain)         <-> L (Observer gain in Luenberger filter)
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        G: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None
    ):
        """Initialize the Discrete Linear Kalman Filter."""
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        
        self.n = self.F.shape[0]  # State dimension
        self.m = self.H.shape[0]  # Measurement dimension
        
        if G is not None:
            self.G = np.asarray(G, dtype=float)
        else:
            self.G = np.zeros((self.n, 1))

        self.x = np.zeros((self.n, 1)) if x0 is None else np.asarray(x0, dtype=float).reshape(self.n, 1)
        self.P = np.eye(self.n) * 1.0 if P0 is None else np.asarray(P0, dtype=float).reshape(self.n, self.n)

        # Store latest innovation for diagnostics
        self.latest_innovation: Optional[np.ndarray] = None
        self.latest_innovation_cov: Optional[np.ndarray] = None
        self.latest_gain: Optional[np.ndarray] = None

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Reset/initialize the state mean and covariance."""
        self.x = np.asarray(x0, dtype=float).reshape(self.n, 1)
        self.P = np.asarray(P0, dtype=float).reshape(self.n, self.n)

    def predict(self, u: Optional[np.ndarray] = None, dt_override: Optional[float] = None) -> KalmanState:
        """Execute state prediction step:
        
            check_x_k = F * hat_x_{k-1} + G * u_{k-1}
            check_P_k = F * hat_P_{k-1} * F^T + Q
        """
        if u is not None:
            u_vec = np.asarray(u, dtype=float).reshape(-1, 1)
            self.x = self.F @ self.x + self.G @ u_vec
        else:
            self.x = self.F @ self.x

        self.P = self.F @ self.P @ self.F.T + self.Q
        return KalmanState(x=self.x.copy(), P=self.P.copy())

    def update(self, y: np.ndarray) -> KalmanState:
        """Execute measurement update (correction) step:
        
            nu_k = y_k - H * check_x_k               (Innovation / Residual)
            S_k  = H * check_P_k * H^T + R           (Innovation Covariance)
            K_k  = check_P_k * H^T * S_k^-1          (Optimal Kalman Gain)
            hat_x_k = check_x_k + K_k * nu_k         (Corrected State)
            hat_P_k = (I - K_k * H) * check_P_k      (Corrected Covariance)
        """
        y_vec = np.asarray(y, dtype=float).reshape(self.m, 1)
        
        # Innovation
        nu = y_vec - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State Correction
        self.x = self.x + K @ nu
        
        # Covariance Correction (Joseph form or standard form)
        I_KH = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P

        self.latest_innovation = nu
        self.latest_innovation_cov = S
        self.latest_gain = K

        return KalmanState(x=self.x.copy(), P=self.P.copy())


def create_1d_constant_velocity_tracker(
    dt: float = 0.1,
    sigma_pos_gps: float = 1.5,
    sigma_acc_process: float = 0.5,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> LinearKalmanFilter:
    """Helper creating a 1D Constant Velocity (CV) vehicle tracker.
    
    State: x = [p, v]^T
    Sensors: GPS position y = [p_meas]
    """
    F = np.array([
        [1.0, dt],
        [0.0, 1.0]
    ])
    
    H = np.array([[1.0, 0.0]])
    
    # Continuous white-noise acceleration discretization
    q_pos = (dt**4) / 4.0 * (sigma_acc_process**2)
    q_vel = (dt**2) * (sigma_acc_process**2)
    q_pos_vel = (dt**3) / 2.0 * (sigma_acc_process**2)
    
    Q = np.array([
        [q_pos,     q_pos_vel],
        [q_pos_vel, q_vel    ]
    ])
    
    R = np.array([[sigma_pos_gps**2]])
    if P0 is None:
        P0 = np.diag([10.0**2, 5.0**2])
    
    return LinearKalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)


def create_2d_constant_velocity_tracker(
    dt: float = 0.1,
    sigma_pos_gps: float = 3.0,
    sigma_acc_process: float = 0.5,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> LinearKalmanFilter:
    """Helper creating a 2D Constant Velocity (CV) vehicle tracker.
    
    State: x = [p_x, p_y, v_x, v_y]^T
    Sensors: GPS [y_x, y_y]^T
    """
    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])
    
    # Continuous white-noise acceleration discretization
    dt4 = (dt**4) / 4.0 * (sigma_acc_process**2)
    dt3 = (dt**3) / 2.0 * (sigma_acc_process**2)
    dt2 = (dt**2) * (sigma_acc_process**2)
    
    Q = np.array([
        [dt4, 0.0, dt3, 0.0],
        [0.0, dt4, 0.0, dt3],
        [dt3, 0.0, dt2, 0.0],
        [0.0, dt3, 0.0, dt2]
    ])
    
    R = np.eye(2) * (sigma_pos_gps**2)
    if P0 is None:
        P0 = np.diag([10.0**2, 10.0**2, 5.0**2, 5.0**2])
    
    return LinearKalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)


def compute_nees(
    true_states: np.ndarray,
    est_states: np.ndarray,
    est_covariances: np.ndarray
) -> np.ndarray:
    """Computes Normalized Estimation Error Squared (NEES) for filter consistency analysis.
    
    NEES_k = (x_true_k - x_hat_k)^T * P_hat_k^-1 * (x_true_k - x_hat_k)
    
    For an n-dimensional state, NEES follows a Chi-square distribution with n degrees of freedom:
        E[NEES_k] = n
    """
    N = len(true_states)
    nees_values = np.zeros(N)
    for k in range(N):
        err = (true_states[k] - est_states[k]).reshape(-1, 1)
        P_k = est_covariances[k]
        nees_values[k] = (err.T @ np.linalg.inv(P_k) @ err).item()
    return nees_values

