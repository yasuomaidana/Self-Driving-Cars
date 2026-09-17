"""Instructor Complete Reference Solution: 2D Kinematic Kalman Filter for Visual Tracking.

========================================================================================
COMPLETE INSTRUCTOR GUIDE & REFERENCE SOLUTION
========================================================================================
This module contains the verified instructor implementation of the 2D Constant Velocity
Linear Kalman Filter for interactive visual object tracking.

Features:
- Full continuous-to-discrete white noise acceleration (CWNA) covariance discretization.
- Joseph form covariance correction: P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
  guaranteeing symmetry and strict positive definiteness.
- Seamless execution with live camera feeds, KITTI sequences, or synthetic benchmarks.
"""

from typing import Optional, Tuple, List, Union
import numpy as np

from .base_kalman_tracker import BaseKalmanTracker2D
from .tracking_kalman import KalmanTracker2D, build_cv_matrices_2d
from .interactive_tracking_exercise import InteractiveKalmanAITracker, run_interactive_tracking_exercise
from .tracking_config import parse_tracking_cli_args, TrackingConfig


class InstructorKalmanTracker2D(BaseKalmanTracker2D):
    """Instructor Reference Implementation with Joseph Stabilized Covariance Form."""

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_std: float = 1.5,
        measurement_noise_std: float = 2.0,
        initial_covariance: float = 50.0,
        F: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ):
        super().__init__(dt=dt)
        self.process_noise_std = float(process_noise_std)
        self.measurement_noise_std = float(measurement_noise_std)
        self.initial_cov = float(initial_covariance)

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

        self.n = self.F.shape[0]
        self.m = self.H.shape[0]

        self.x = np.zeros((self.n, 1), dtype=np.float64)
        self.P = self.P0.copy()
        self.initialized = False
        self.history = []
        self.step_count = 0

    def initialize(
        self,
        initial_x: float,
        initial_y: float,
        initial_vx: float = 0.0,
        initial_vy: float = 0.0
    ) -> None:
        """Initialize state vector and reset uncertainty."""
        self.x = np.array([[initial_x], [initial_y], [initial_vx], [initial_vy]], dtype=np.float64)
        self.P = self.P0.copy()
        self.initialized = True
        self.history.clear()
        self.step_count = 0

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        self.x = np.zeros((self.n, 1), dtype=np.float64)
        self.P = self.P0.copy()
        self.initialized = False
        self.history.clear()
        self.step_count = 0

    def predict(self) -> Tuple[float, float, float, float]:
        """Propagate state mean and covariance."""
        if not self.initialized:
            raise RuntimeError("Tracker must be initialized before calling predict().")

        # 1. State mean propagation: x_check = F * x_hat
        self.x = self.F @ self.x

        # 2. Covariance propagation: P_check = F * P_hat * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])

    def update(
        self,
        measurement: Optional[Tuple[float, float]]
    ) -> Tuple[float, float, float, float]:
        """Incorporate sensor measurement using Joseph Stabilized Covariance update."""
        if not self.initialized:
            raise RuntimeError("Tracker must be initialized before calling update().")

        if measurement is not None:
            y = np.array([[measurement[0]], [measurement[1]]], dtype=np.float64)

            # 1. Innovation residual: nu = y - H * x_check
            nu = y - self.H @ self.x

            # 2. Innovation covariance: S = H * P_check * H^T + R
            S = self.H @ self.P @ self.H.T + self.R

            # 3. Optimal Kalman Gain: K = P_check * H^T * S^-1
            K = self.P @ self.H.T @ np.linalg.inv(S)

            # 4. Correct State Mean: x_hat = x_check + K * nu
            self.x = self.x + K @ nu

            # 5. Joseph Form Covariance Correction:
            # P_hat = (I - K*H) * P_check * (I - K*H)^T + K * R * K^T
            I_KH = np.eye(self.n, dtype=np.float64) - K @ self.H
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])


def run_instructor_tracking_exercise(
    source: Optional[Union[str, int]] = None,
    camera_id: Optional[int] = None,
    model_name: str = "yolov8n.pt",
    target_class: Optional[str] = None,
    hybrid_mode: bool = False,
    dt: float = 1.0 / 30.0,
    process_noise: float = 1.5,
    measurement_noise: float = 2.0,
    interactive_gui: bool = True,
    output_dir: Optional[str] = None,
) -> InteractiveKalmanAITracker:
    """Convenience launcher for the Instructor verified tracking exercise."""
    instructor_kf = InstructorKalmanTracker2D(
        dt=dt,
        process_noise_std=process_noise,
        measurement_noise_std=measurement_noise
    )

    return run_interactive_tracking_exercise(
        source=source,
        camera_id=camera_id,
        model_name=model_name,
        target_class=target_class,
        hybrid_mode=hybrid_mode,
        kalman_tracker=instructor_kf,
        interactive_gui=interactive_gui,
        output_dir=output_dir
    )


def main():
    """Main entrypoint for instructor running the full reference solution."""
    config = parse_tracking_cli_args(description="Instructor Visual Kalman Tracking Reference Solution")
    instructor_kf = InstructorKalmanTracker2D(
        dt=config.dt,
        process_noise_std=config.process_noise,
        measurement_noise_std=config.measurement_noise,
        initial_covariance=config.initial_covariance
    )
    print("\n" + "="*75)
    print("[Instructor Reference] Launching Interactive AI Tracking with Verified Solution")
    print(f"  • dt: {config.dt:.4f}s | Q Process Noise: {config.process_noise} | R Measurement Noise: {config.measurement_noise}")
    print("="*75 + "\n")

    run_interactive_tracking_exercise(
        source=config.source,
        camera_id=config.camera_id,
        kitti_dir=config.kitti_dir,
        kitti_label=config.kitti_label,
        use_kitti_demo=config.use_kitti_demo,
        model_name=config.model_name,
        dataset=config.dataset,
        target_class=config.target_class,
        hybrid_mode=config.hybrid_mode,
        feature_only=config.feature_only,
        disable_helper=config.disable_helper,
        feature_algorithm=config.feature_algorithm,
        num_points=config.num_points,
        start_paused=config.start_paused,
        output_dir=config.output_dir,
        interactive_gui=config.interactive_gui,
        kalman_tracker=instructor_kf
    )


if __name__ == "__main__":
    main()
