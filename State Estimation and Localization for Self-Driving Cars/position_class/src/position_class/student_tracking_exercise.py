"""Student Exercise: Implement the 2D Kinematic Kalman Filter for Visual Tracking.

========================================================================================
LAB OBJECTIVE:
========================================================================================
Implement the Discrete Linear Kalman Filter (LKF) from first principles to track
an object's Center of Mass (CoM) in an image/video stream.

State Vector (4D Kinematics):
    x = [p_x, p_y, v_x, v_y]^T
    where (p_x, p_y) is position in pixels, (v_x, v_y) is velocity in pixels/second.

Measurement Vector (2D Visual Detection):
    y = [p_x_meas, p_y_meas]^T

Mathematical Model:
    x_k = F * x_{k-1} + w_k,    w_k ~ N(0, Q)
    y_k = H * x_k + v_k,        v_k ~ N(0, R)
"""

from typing import Optional, Tuple, List, Union
import numpy as np

from .base_kalman_tracker import BaseKalmanTracker2D
from .interactive_tracking_exercise import InteractiveKalmanAITracker, run_interactive_tracking_exercise
from .tracking_config import parse_tracking_cli_args, TrackingConfig


class StudentKalmanTracker2D(BaseKalmanTracker2D):
    """Student Implementation of the 2D Constant Velocity Kalman Filter."""

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
        """Construct the 2D Kalman Filter and define system matrices."""
        super().__init__(dt=dt)
        self.process_noise_std = float(process_noise_std)
        self.measurement_noise_std = float(measurement_noise_std)
        self.initial_cov = float(initial_covariance)

        # -------------------------------------------------------------------------
        # TODO [Student Step 1.1]: Define the 4x4 State Transition Matrix F
        # Equations of Motion (Constant Velocity):
        #   p_x(k) = p_x(k-1) + v_x(k-1) * dt
        #   p_y(k) = p_y(k-1) + v_y(k-1) * dt
        #   v_x(k) = v_x(k-1)
        #   v_y(k) = v_y(k-1)
        #
        # -------------------------------------------------------------------------
        if F is not None:
            self.F = np.asarray(F, dtype=np.float64)
        else:
            self.F = None

        # -------------------------------------------------------------------------
        # TODO [Student Step 1.2]: Define the 2x4 Measurement Matrix H
        # Observation Equation:
        #   y = [p_x_meas, p_y_meas]^T = H * x
        #
        # -------------------------------------------------------------------------
        if H is not None:
            self.H = np.asarray(H, dtype=np.float64)
        else:
            self.H = None

        # -------------------------------------------------------------------------
        # TODO [Student Step 1.3]: Define the 4x4 Process Noise Covariance Matrix Q
        # Continuous White Noise Acceleration (CWNA) Model:
        #   q_pos = (dt^3 / 3) * sigma_a^2
        #   q_vel = dt * sigma_a^2
        #   q_pv  = (dt^2 / 2) * sigma_a^2
        #
        # Matrix Form:
        #       [ q_pos    0    q_pv    0   ]
        #   Q = [   0    q_pos    0    q_pv ]
        #       [  q_pv    0    q_vel   0   ]
        #       [   0     q_pv    0   q_vel ]
        # where sigma_a = self.process_noise_std
        # -------------------------------------------------------------------------
        if Q is not None:
            self.Q = np.asarray(Q, dtype=np.float64)
        else:
            self.Q = None

        # -------------------------------------------------------------------------
        # TODO [Student Step 1.4]: Define the 2x2 Measurement Noise Covariance Matrix R
        # Measurement Variance:
        #   R = [ sigma_meas^2        0      ]
        #       [      0        sigma_meas^2 ]
        # where sigma_meas = self.measurement_noise_std
        # -------------------------------------------------------------------------
        if R is not None:
            self.R = np.asarray(R, dtype=np.float64)
        else:
            self.R = np.eye(2, dtype=np.float64) * (self.measurement_noise_std ** 2)

        # -------------------------------------------------------------------------
        # TODO [Student Step 1.5]: Define Initial State Uncertainty Covariance P0
        # Initial Covariance Matrix:
        #   P0 = sigma_0^2 * I_4
        # where sigma_0^2 = self.initial_cov
        # -------------------------------------------------------------------------
        if P0 is not None:
            self.P0 = np.asarray(P0, dtype=np.float64)
        else:
            self.P0 = None

        self.n = 4 if self.F is None else self.F.shape[0]  # State dimension (4)
        self.m = 2 if self.H is None else self.H.shape[0]  # Measurement dimension (2)

        self.x = np.zeros((self.n, 1), dtype=np.float64)
        self.P = np.eye(self.n, dtype=np.float64) * self.initial_cov if self.P0 is None else self.P0.copy()
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
        """Initialize state vector x and reset covariance P."""
        self.x = np.array([[initial_x], [initial_y], [initial_vx], [initial_vy]], dtype=np.float64)
        self.P = self.P0.copy() if self.P0 is not None else np.eye(4, dtype=np.float64) * self.initial_cov
        self.initialized = True
        self.history.clear()
        self.step_count = 0

    def reset(self) -> None:
        """Reset tracker state to IDLE."""
        self.x = np.zeros((self.n, 1), dtype=np.float64)
        self.P = self.P0.copy() if self.P0 is not None else np.eye(4, dtype=np.float64) * self.initial_cov
        self.initialized = False
        self.history.clear()
        self.step_count = 0

    def predict(self) -> Tuple[float, float, float, float]:
        """Execute Kalman Prediction Step.

        Propagates state mean and covariance ahead by dt:
            x_check = F * x
            P_check = F * P * F^T + Q
        """
        if not self.initialized:
            raise RuntimeError("Tracker must be initialized before calling predict().")

        if self.F is None or self.Q is None:
            raise NotImplementedError("Step 1 matrices F and Q must be defined before predict() can run.")

        # -------------------------------------------------------------------------
        # TODO [Student Step 2.1]: State Mean Propagation
        #   x_check = F * x
        # -------------------------------------------------------------------------

        # -------------------------------------------------------------------------
        # TODO [Student Step 2.2]: Covariance Expansion
        #   P_check = F * P * F^T + Q
        # -------------------------------------------------------------------------

        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])

    def update(
        self,
        measurement: Optional[Tuple[float, float]]
    ) -> Tuple[float, float, float, float]:
        """Execute Kalman Measurement Update (Correction) Step.

        When measurement y is available:
            nu = y - H * x_check (Innovation Residual)
            S = H * P_check * H^T + R (Innovation Covariance)
            K = P_check * H^T * S^-1 (Kalman Gain)
            x_hat = x_check + K * nu (Corrected State Mean)
            P_hat = (I - K * H) * P_check (Corrected Covariance)

        When measurement is None:
            Retain prediction (Dead Reckoning).
        """
        if not self.initialized:
            raise RuntimeError("Tracker must be initialized before calling update().")

        if self.H is None or self.R is None:
            raise NotImplementedError("Step 1 matrices H and R must be defined before update() can run.")

        if measurement is not None:
            # Measurement vector y (2x1 column vector)
            y = np.array([[measurement[0]], [measurement[1]]], dtype=np.float64)

            # ---------------------------------------------------------------------
            # TODO [Student Step 3.1]: Innovation Residual
            #   nu = y - H * x
            # ---------------------------------------------------------------------

            # ---------------------------------------------------------------------
            # TODO [Student Step 3.2]: Innovation Covariance
            #   S = H * P * H^T + R
            # ---------------------------------------------------------------------

            # ---------------------------------------------------------------------
            # TODO [Student Step 3.3]: Optimal Kalman Gain
            #   K = P * H^T * S^-1
            # ---------------------------------------------------------------------

            # ---------------------------------------------------------------------
            # TODO [Student Step 3.4]: Correct State Mean
            #   x = x + K * nu
            # ---------------------------------------------------------------------

            # ---------------------------------------------------------------------
            # TODO [Student Step 3.5]: Correct Covariance
            #   P = (I - K * H) * P
            # ---------------------------------------------------------------------

        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])


def run_student_tracking_exercise(
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
    """Convenience launcher for the Student tracking exercise."""
    # Instantiate student Kalman Filter
    student_kf = StudentKalmanTracker2D(
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
        kalman_tracker=student_kf,
        interactive_gui=interactive_gui,
        output_dir=output_dir
    )


def main():
    """Main entrypoint for students executing their tracking filter."""
    config = parse_tracking_cli_args(description="Student Visual Kalman Tracking Exercise")
    student_kf = StudentKalmanTracker2D(
        dt=config.dt,
        process_noise_std=config.process_noise,
        measurement_noise_std=config.measurement_noise,
        initial_covariance=config.initial_covariance
    )
    print("\n" + "="*75)
    print("[Student Lab] Launching Interactive AI Tracking with StudentKalmanTracker2D")
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
        kalman_tracker=student_kf
    )


if __name__ == "__main__":
    main()
