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


class KalmanTracker2D:
    """2D Constant Velocity Kalman Filter for tracking object Center of Mass.

    State vector: x = [x, y, vx, vy]^T
    Measurement:  z = [x, y]^T
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 2.0,
        initial_covariance: float = 50.0,
    ):
        """Initialize the 2D Kalman filter.

        Args:
            dt: Time step between frames in seconds.
            process_noise_std: Standard deviation of process noise (acceleration disturbance).
            measurement_noise_std: Standard deviation of visual measurement noise (pixels).
            initial_covariance: Initial uncertainty in state estimation.
        """
        self.dt = float(dt)
        self.process_noise_std = float(process_noise_std)
        self.measurement_noise_std = float(measurement_noise_std)
        self.initial_cov = float(initial_covariance)

        # 4D State Transition Matrix (Constant Velocity Model)
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0    ],
            [0.0, 1.0, 0.0,     self.dt],
            [0.0, 0.0, 1.0,     0.0    ],
            [0.0, 0.0, 0.0,     1.0    ]
        ], dtype=np.float64)

        # 2x4 Measurement Matrix (Observing x, y position directly)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Process Noise Covariance (Continuous White Noise Acceleration discretization)
        q_pos = (self.dt**3) / 3.0 * (self.process_noise_std**2)
        q_vel = self.dt * (self.process_noise_std**2)
        q_pv = (self.dt**2) / 2.0 * (self.process_noise_std**2)
        self.Q = np.array([
            [q_pos, 0.0,   q_pv,  0.0  ],
            [0.0,   q_pos, 0.0,   q_pv ],
            [q_pv,  0.0,   q_vel, 0.0  ],
            [0.0,   q_pv,  0.0,   q_vel]
        ], dtype=np.float64)

        # Measurement Noise Covariance (2x2)
        self.R = np.eye(2, dtype=np.float64) * (self.measurement_noise_std**2)

        # State and Covariance
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * self.initial_cov
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
