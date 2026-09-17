"""Abstract Base Class for 2D Visual Kalman Trackers.

Defines the required mathematical interface and lifecycle contract for
Kalman Filter tracking in autonomous vehicle vision pipelines.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import numpy as np

from .tracking_kalman import KalmanTrackState


class BaseKalmanTracker2D(ABC):
    """Abstract Parent Class defining the Kalman Filter contract for 2D Object Tracking.

    Students and instructors subclass this to implement:
    1. Constructor / System Matrices (__init__): F, H, Q, R, P0
    2. Prediction Step (predict): State propagation and Covariance expansion
    3. Correction Step (update): Innovation residual, Kalman Gain, State update, Covariance update

    State Space Convention:
        State vector x: [p_x, p_y, v_x, v_y]^T   (4x1)
        Observation y:  [p_x, p_y]^T              (2x1)
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        F: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ):
        self.dt = float(dt)
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P0 = P0

        # State vector x and Covariance P
        self.x: np.ndarray = np.zeros((4, 1), dtype=np.float64)
        self.P: np.ndarray = np.eye(4, dtype=np.float64) * 50.0
        self.initialized: bool = False
        self.history: List[KalmanTrackState] = []
        self.step_count: int = 0

    @property
    def is_initialized(self) -> bool:
        """Returns True if the tracker has been initialized with an initial state."""
        return self.initialized

    @property
    def position(self) -> Tuple[float, float]:
        """Returns current estimated position (p_x, p_y) in pixel coordinates."""
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Returns current estimated velocity (v_x, v_y) in pixels/second."""
        return float(self.x[2, 0]), float(self.x[3, 0])

    @abstractmethod
    def initialize(
        self,
        initial_x: float,
        initial_y: float,
        initial_vx: float = 0.0,
        initial_vy: float = 0.0
    ) -> None:
        """Initialize the state vector x and covariance P with initial detection values."""
        pass

    @abstractmethod
    def predict(self) -> Tuple[float, float, float, float]:
        """Executes the Kalman prediction step.

        Propagates state mean and error covariance ahead by dt:
            x_check = F * x_hat
            P_check = F * P_hat * F^T + Q

        Returns:
            Tuple of (pred_x, pred_y, pred_vx, pred_vy)
        """
        pass

    @abstractmethod
    def update(
        self,
        measurement: Optional[Tuple[float, float]]
    ) -> Tuple[float, float, float, float]:
        """Executes the Kalman measurement update (correction) step.

        If a visual measurement is available:
            nu = y - H * x_check                   (Innovation / Residual)
            S  = H * P_check * H^T + R             (Innovation Covariance)
            K  = P_check * H^T * S^-1              (Optimal Kalman Gain)
            x_hat = x_check + K * nu               (State Correction)
            P_hat = (I - K*H) * P_check            (Covariance Correction)

        If measurement is None (occlusion/missed detection), dead reckoning is retained.

        Args:
            measurement: (meas_x, meas_y) or None.

        Returns:
            Tuple of (est_x, est_y, est_vx, est_vy)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the tracker to an uninitialized IDLE state."""
        pass

    def step(
        self,
        measurement: Optional[Tuple[float, float]],
        image_shape: Optional[Tuple[int, int]] = None
    ) -> KalmanTrackState:
        """Runs one full cycle: Predict -> Update -> Compute Center Error.

        Args:
            measurement: (x, y) coordinates of detected object center, or None.
            image_shape: Optional (height, width) of image.

        Returns:
            KalmanTrackState record.
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

    def compute_center_error(self, image_width: int, image_height: int) -> Tuple[float, float, float]:
        """Compute error vector and Euclidean distance of current estimate to image center."""
        cx_img = image_width / 2.0
        cy_img = image_height / 2.0
        ex = float(self.x[0, 0]) - cx_img
        ey = float(self.x[1, 0]) - cy_img
        dist = float(np.hypot(ex, ey))
        return ex, ey, dist
