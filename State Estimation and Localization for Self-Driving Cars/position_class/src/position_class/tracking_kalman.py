"""Kalman Filter Module for 2D Visual Object Tracking.

This module provides a 2D Kinematic Kalman Filter for tracking visual objects,
estimating position and velocity, handling occlusions / missing detections,
and calculating tracking error relative to the image center.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np


@dataclass
class KalmanTrackState:
    """State record for 2D visual tracking at a given time step."""
    step: int
    pred_x: float
    pred_y: float
    pred_vx: float
    pred_vy: float
    est_x: float
    est_y: float
    est_vx: float
    est_vy: float
    meas_x: Optional[float]
    meas_y: Optional[float]
    cov_pos_variance: float
    error_x: float
    error_y: float
    error_distance: float
    is_detected: bool


def build_cv_matrices_2d(
    dt: float = 1.0 / 30.0,
    process_noise_std: float = 1.0,
    measurement_noise_std: float = 2.0,
    initial_covariance: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Didactic builder for 2D Constant Velocity (CV) Kalman filter matrices.

    Follows the Day 01 & Day 02 State Estimation framework:
    -------------------------------------------------------
    State vector (n=4):
        x = [p_x, p_y, v_x, v_y]^T
        where (p_x, p_y) is object Center of Mass in image coordinates (pixels),
        and (v_x, v_y) is image plane velocity (pixels/second).

    Measurement vector (m=2):
        y = [p_x_meas, p_y_meas]^T

    1. State Transition Matrix F (4x4):
        p_x(k) = p_x(k-1) + dt * v_x(k-1)
        p_y(k) = p_y(k-1) + dt * v_y(k-1)
        v_x(k) = v_x(k-1)
        v_y(k) = v_y(k-1)
        F = [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1,  0],
             [0, 0, 0,  1]]

    2. Measurement Matrix H (2x4):
        Directly observes position components:
        H = [[1, 0, 0, 0],
             [0, 1, 0, 0]]

    3. Process Noise Covariance Q (4x4):
        Continuous White Noise Acceleration (CWNA) discretization:
        q_pos = (dt^3 / 3) * sigma_a^2
        q_vel = dt * sigma_a^2
        q_pv  = (dt^2 / 2) * sigma_a^2

    4. Measurement Noise Covariance R (2x2):
        R = diag(sigma_meas^2, sigma_meas^2)

    5. Initial State Covariance P0 (4x4):
        P0 = diag(initial_cov, initial_cov, initial_cov, initial_cov)

    Returns:
        Tuple of (F, H, Q, R, P0)
    """
    dt = float(dt)
    sigma_a = float(process_noise_std)
    sigma_meas = float(measurement_noise_std)

    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float64)

    q_pos = (dt**3) / 3.0 * (sigma_a**2)
    q_vel = dt * (sigma_a**2)
    q_pv = (dt**2) / 2.0 * (sigma_a**2)
    Q = np.array([
        [q_pos, 0.0,   q_pv,  0.0  ],
        [0.0,   q_pos, 0.0,   q_pv ],
        [q_pv,  0.0,   q_vel, 0.0  ],
        [0.0,   q_pv,  0.0,   q_vel]
    ], dtype=np.float64)

    R = np.eye(2, dtype=np.float64) * (sigma_meas**2)
    P0 = np.eye(4, dtype=np.float64) * float(initial_covariance)

    return F, H, Q, R, P0


from .base_kalman_tracker import BaseKalmanTracker2D


class KalmanTracker2D(BaseKalmanTracker2D):
    """2D Constant Velocity Kalman Filter for tracking object Center of Mass.

    State vector: x = [p_x, p_y, v_x, v_y]^T
    Measurement:  z = [p_x, p_y]^T

    Supports both:
    1. Direct injection of fundamental system matrices: F, H, Q, R, P0, G (as taught in Day 01 & Day 02)
    2. Automatic matrix generation from physical parameters (dt, process_noise_std, measurement_noise_std)
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 2.0,
        initial_covariance: float = 50.0,
        F: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        G: Optional[np.ndarray] = None,
    ):
        """Initialize the 2D Kalman filter.

        Args:
            dt: Time step between frames in seconds.
            process_noise_std: Standard deviation of process noise (acceleration disturbance in px/s^2).
            measurement_noise_std: Standard deviation of visual measurement noise (pixels).
            initial_covariance: Initial uncertainty in state estimation.
            F: Optional custom State Transition Matrix (4x4).
            H: Optional custom Measurement Matrix (2x4).
            Q: Optional custom Process Noise Covariance Matrix (4x4).
            R: Optional custom Measurement Noise Covariance Matrix (2x2).
            P0: Optional custom Initial Error Covariance Matrix (4x4).
            G: Optional custom Control Matrix (4 x n_u).
        """
        self.dt = float(dt)
        self.process_noise_std = float(process_noise_std)
        self.measurement_noise_std = float(measurement_noise_std)
        self.initial_cov = float(initial_covariance)

        # Build default matrices if custom matrices are not explicitly passed
        def_F, def_H, def_Q, def_R, def_P0 = build_cv_matrices_2d(
            dt=self.dt,
            process_noise_std=self.process_noise_std,
            measurement_noise_std=self.measurement_noise_std,
            initial_covariance=self.initial_cov
        )

        self.F = np.asarray(F if F is not None else def_F, dtype=np.float64)
        self.H = np.asarray(H if H is not None else def_H, dtype=np.float64)
        self.Q = np.asarray(Q if Q is not None else def_Q, dtype=np.float64)
        self.R = np.asarray(R if R is not None else def_R, dtype=np.float64)
        self.P0 = np.asarray(P0 if P0 is not None else def_P0, dtype=np.float64)
        self.G = np.asarray(G, dtype=np.float64) if G is not None else None

        self.n = self.F.shape[0]  # State dimension (4)
        self.m = self.H.shape[0]  # Measurement dimension (2)

        # State and Covariance
        self.x = np.zeros((self.n, 1), dtype=np.float64)
        self.P = self.P0.copy()
        self.initialized = False
        self.history: List[KalmanTrackState] = []
        self.step_count = 0

    def initialize(self, initial_x: float, initial_y: float, initial_vx: float = 0.0, initial_vy: float = 0.0) -> None:
        """Initialize tracker with starting position and optional velocity."""
        self.x = np.array([[initial_x], [initial_y], [initial_vx], [initial_vy]], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * self.initial_cov
        self.initialized = True
        self.history.clear()
        self.step_count = 0

    def reset(self) -> None:
        """Reset the tracker to uninitialized state."""
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * self.initial_cov
        self.initialized = False
        self.history.clear()
        self.step_count = 0

    def predict(self) -> Tuple[float, float, float, float]:
        """Perform Kalman state and covariance prediction step.

        Returns:
            Tuple of (pred_x, pred_y, pred_vx, pred_vy)
        """
        if not self.initialized:
            raise RuntimeError("KalmanTracker2D must be initialized before prediction.")

        # x_{k|k-1} = F * x_{k-1|k-1}
        self.x = self.F @ self.x
        # P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])

    def update(self, measurement: Optional[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """Perform Kalman measurement correction step.

        If measurement is None (occlusion or detection missed), dead reckoning prediction is retained.

        Args:
            measurement: (meas_x, meas_y) or None.

        Returns:
            Tuple of (est_x, est_y, est_vx, est_vy)
        """
        if not self.initialized:
            raise RuntimeError("KalmanTracker2D must be initialized before update.")

        if measurement is not None:
            z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float64)
            # Innovation (measurement residual): y = z - H * x
            y = z - (self.H @ self.x)
            # Innovation covariance: S = H * P * H^T + R
            S = self.H @ self.P @ self.H.T + self.R
            # Optimal Kalman Gain: K = P * H^T * S^-1
            K = self.P @ self.H.T @ np.linalg.inv(S)
            # State Update: x = x + K * y
            self.x = self.x + K @ y
            # Covariance Update: P = (I - K * H) * P (Joseph form stabilized: (I-KH)P(I-KH)^T + KRK^T)
            I_KH = np.eye(4, dtype=np.float64) - K @ self.H
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])

    def step(
        self,
        measurement: Optional[Tuple[float, float]],
        image_shape: Optional[Tuple[int, int]] = None
    ) -> KalmanTrackState:
        """Run one full cycle: Predict -> Update -> Compute Center Error.

        Args:
            measurement: (x, y) coordinates of detected object center, or None if lost/occluded.
            image_shape: Optional (height, width) of image to compute error against center.

        Returns:
            KalmanTrackState record with all predictions, estimates, and errors.
        """
        if not self.initialized:
            if measurement is not None:
                self.initialize(measurement[0], measurement[1])
            else:
                raise RuntimeError("Cannot step uninitialized tracker without a measurement.")

        # 1. Predict
        px, py, pvx, pvy = self.predict()

        # 2. Update
        is_detected = measurement is not None
        ex, ey, evx, evy = self.update(measurement)

        # 3. Compute error relative to image center
        err_x, err_y, err_dist = 0.0, 0.0, 0.0
        if image_shape is not None:
            h, w = image_shape[:2]
            cx_img = w / 2.0
            cy_img = h / 2.0
            err_x = ex - cx_img
            err_y = ey - cy_img
            err_dist = float(np.hypot(err_x, err_y))

        # Position variance (trace of top-left 2x2 position covariance)
        pos_var = float(self.P[0, 0] + self.P[1, 1])

        state_rec = KalmanTrackState(
            step=self.step_count,
            pred_x=px,
            pred_y=py,
            pred_vx=pvx,
            pred_vy=pvy,
            est_x=ex,
            est_y=ey,
            est_vx=evx,
            est_vy=evy,
            meas_x=float(measurement[0]) if measurement is not None else None,
            meas_y=float(measurement[1]) if measurement is not None else None,
            cov_pos_variance=pos_var,
            error_x=err_x,
            error_y=err_y,
            error_distance=err_dist,
            is_detected=is_detected,
        )

        self.history.append(state_rec)
        self.step_count += 1
        return state_rec

    @property
    def position(self) -> Tuple[float, float]:
        """Current estimated position (x, y)."""
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current estimated velocity (vx, vy)."""
        return float(self.x[2, 0]), float(self.x[3, 0])

    def compute_center_error(self, image_width: int, image_height: int) -> Tuple[float, float, float]:
        """Compute error vector and Euclidean distance of current estimate to image center.

        Returns:
            Tuple of (error_x, error_y, euclidean_distance) in pixels.
        """
        cx_img = image_width / 2.0
        cy_img = image_height / 2.0
        ex = float(self.x[0, 0]) - cx_img
        ey = float(self.x[1, 0]) - cy_img
        dist = float(np.hypot(ex, ey))
        return ex, ey, dist
