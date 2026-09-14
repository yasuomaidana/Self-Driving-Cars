"""Error-State Extended Kalman Filter (ES-EKF) for 3D IMU + GNSS Localization."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .inertial_navigation import Quaternion, skew_symmetric


class ErrorStateEKF:
    r"""9D Error-State Extended Kalman Filter (Position, Velocity, Orientation error).
    
    Nominal State x = [p (3x1), v (3x1), q (quaternion)]
    Error State \delta x = [\delta p (3x1), \delta v (3x1), \delta \theta (3x1)] (9x1)
    """

    def __init__(
        self,
        init_pos: np.ndarray,
        init_vel: np.ndarray,
        init_quat: Quaternion,
        init_cov: np.ndarray,
        accel_noise_std: float = 0.1,
        gyro_noise_std: float = 0.01,
        gravity: np.ndarray = np.array([0.0, 0.0, -9.81])
    ):
        self.p = np.asarray(init_pos, dtype=np.float64).copy()
        self.v = np.asarray(init_vel, dtype=np.float64).copy()
        self.q = init_quat
        self.P = np.asarray(init_cov, dtype=np.float64).copy()  # 9x9 covariance
        self.g = np.asarray(gravity, dtype=np.float64).copy()

        # Continuous noise spectral densities
        self.var_acc = accel_noise_std ** 2
        self.var_gyro = gyro_noise_std ** 2

    def predict(self, f_b: np.ndarray, omega_b: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, Quaternion, np.ndarray]:
        """High-rate IMU nominal state propagation and error covariance propagation."""
        # 1. Propagate Nominal State
        C_nb = self.q.to_rotation_matrix()
        a_nav = C_nb @ f_b + self.g
        
        self.p = self.p + self.v * dt + 0.5 * a_nav * (dt ** 2)
        self.v = self.v + a_nav * dt
        
        rotvec = omega_b * dt
        delta_q = Quaternion.from_rotvec(rotvec)
        self.q = self.q.multiply(delta_q)

        # 2. Compute Error-State Transition Matrix F_k (9x9)
        # delta_p_k = delta_p_{k-1} + dt * delta_v_{k-1}
        # delta_v_k = delta_v_{k-1} - dt * C_nb * [f_b]_x * delta_theta_{k-1}
        # delta_theta_k = delta_theta_{k-1}
        F_k = np.eye(9, dtype=np.float64)
        F_k[0:3, 3:6] = np.eye(3) * dt
        F_k[3:6, 6:9] = -skew_symmetric(C_nb @ f_b) * dt

        # Process Noise Covariance Q_k (9x9)
        # L_k noise mapping matrix (9x6)
        L_k = np.zeros((9, 6), dtype=np.float64)
        L_k[3:6, 0:3] = np.eye(3) * dt
        L_k[6:9, 3:6] = np.eye(3) * dt
        
        Q_c = np.diag([
            self.var_acc, self.var_acc, self.var_acc,
            self.var_gyro, self.var_gyro, self.var_gyro
        ])
        Q_k = L_k @ Q_c @ L_k.T

        # 3. Propagate Covariance
        self.P = F_k @ self.P @ F_k.T + Q_k

        return self.p.copy(), self.v.copy(), self.q, self.P.copy()

    def update_gnss_position(self, p_meas: np.ndarray, r_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Quaternion, np.ndarray, float]:
        r"""Low-rate GNSS position measurement correction.
        
        Measurement model: y_k = p_meas - p_nominal = H * \delta x + v_k
        H = [I_3, 0_3x3, 0_3x3] (3x9)
        
        Returns:
            (p, v, q, P, nis): Updated state, covariance, and Normalized Innovation Squared (NIS).
        """
        # Innovation
        y = np.asarray(p_meas, dtype=np.float64) - self.p
        
        # Measurement Jacobian
        H = np.zeros((3, 9), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        
        # Innovation covariance
        S = H @ self.P @ H.T + r_cov
        S_inv = np.linalg.inv(S)
        nis = float(y.T @ S_inv @ y)

        # Kalman gain
        K = self.P @ H.T @ S_inv

        # Error state correction
        delta_x = K @ y

        # 1. State Injection
        self.p = self.p + delta_x[0:3]
        self.v = self.v + delta_x[3:6]
        
        # Quaternion correction: q = q \otimes Exp(0.5 * delta_theta)
        delta_theta = delta_x[6:9]
        dq = Quaternion.from_rotvec(delta_theta)
        self.q = self.q.multiply(dq)

        # 2. Covariance Update (Joseph form for numerical stability)
        I_KH = np.eye(9, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ r_cov @ K.T

        return self.p.copy(), self.v.copy(), self.q, self.P.copy(), nis
