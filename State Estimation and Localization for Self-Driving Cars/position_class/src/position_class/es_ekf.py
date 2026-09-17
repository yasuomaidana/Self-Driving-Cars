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

    def update(
        self,
        y: np.ndarray,
        h_func,
        H_jac: np.ndarray,
        R: np.ndarray,
        M_jac: Optional[np.ndarray] = None,
        inject_func = None
    ) -> Tuple[np.ndarray, np.ndarray, Quaternion, np.ndarray, float]:
        r"""Generic Measurement Correction, State Injection, and Error Reset.
        
        Academic Definition:
            1. Innovation:            \nu = y - h(x_nom)
            2. Innovation Covariance: S  = H * P * H^T + M * R * M^T
            3. Error Kalman Gain:     K  = P * H^T * inv(S)
            4. Error State Estimate:  \delta x = K * \nu
            5. State Injection:       x_nom <- x_nom \oplus \delta x
            6. Error Reset:           \delta x <- 0
            7. Covariance Reset:      P <- (I - K*H) * P * (I - K*H)^T + K * (M*R*M^T) * K^T
            
        Returns:
            (p, v, q, P, nis): Updated state components, error covariance, and NIS.
        """
        y_vec = np.asarray(y, dtype=np.float64)
        m = len(y_vec)
        if M_jac is None:
            M_jac = np.eye(m, dtype=np.float64)
            
        # 1. Innovation
        y_pred = h_func(self.p, self.v, self.q) if callable(h_func) else h_func
        nu = y_vec - np.asarray(y_pred, dtype=np.float64)
        
        # 2. Innovation Covariance
        R_eff = M_jac @ R @ M_jac.T
        S = H_jac @ self.P @ H_jac.T + R_eff
        S_inv = np.linalg.inv(S)
        nis = float(nu.T @ S_inv @ nu)
        
        # 3. Kalman Gain
        K = self.P @ H_jac.T @ S_inv
        
        # 4. Error State Correction
        delta_x = K @ nu
        
        # 5. State Injection
        if inject_func is not None:
            self.p, self.v, self.q = inject_func(self.p, self.v, self.q, delta_x)
        else:
            self.p = self.p + delta_x[0:3]
            self.v = self.v + delta_x[3:6]
            delta_theta = delta_x[6:9]
            dq = Quaternion.from_rotvec(delta_theta)
            self.q = self.q.multiply(dq)
            
        # 6 & 7. Covariance Reset (Joseph form for numerical stability)
        I_KH = np.eye(9, dtype=np.float64) - K @ H_jac
        self.P = I_KH @ self.P @ I_KH.T + K @ R_eff @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        return self.p.copy(), self.v.copy(), self.q, self.P.copy(), nis

    def update_gnss_position(self, p_meas: np.ndarray, r_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Quaternion, np.ndarray, float]:
        r"""Low-rate GNSS position measurement correction (Convenience wrapper around update).
        
        Measurement model: y_k = p_meas - p_nominal = H * \delta x + v_k
        H = [I_3, 0_3x3, 0_3x3] (3x9)
        """
        H = np.zeros((3, 9), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        return self.update(
            y=p_meas,
            h_func=lambda p, v, q: p,
            H_jac=H,
            R=r_cov
        )


class ErrorStateKalmanFilter:
    """Generic Multi-Dimensional Error-State Extended Kalman Filter (ES-EKF).
    
    Maintains nominal kinematic state x_nom and error-state covariance P.
    Provides generic predict (propagate) and update methods completely decoupled
    from specific physical sensors or vehicle dimensions.
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray):
        self.x_nom = np.asarray(x0, dtype=np.float64).reshape(-1, 1)
        self.P = np.asarray(P0, dtype=np.float64)
        self.n = self.x_nom.shape[0]

        self.latest_innovation: Optional[np.ndarray] = None
        self.latest_innovation_cov: Optional[np.ndarray] = None
        self.latest_gain: Optional[np.ndarray] = None

    def predict(
        self,
        f_nom_func,
        F_jac: np.ndarray,
        Q: np.ndarray,
        u: Optional[np.ndarray] = None,
        L_jac: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Executes High-Rate Nominal State Propagation & Error Covariance Propagation:
            check_x_nom = f_nom(hat_x_nom, u)
            check_P = F_delta * hat_P * F_delta^T + L * Q * L^T
        """
        if u is not None:
            self.x_nom = f_nom_func(self.x_nom, u).reshape(-1, 1)
        else:
            try:
                self.x_nom = f_nom_func(self.x_nom).reshape(-1, 1)
            except TypeError:
                self.x_nom = f_nom_func(self.x_nom, None).reshape(-1, 1)

        if L_jac is None:
            L_jac = np.eye(self.n)

        self.P = F_jac @ self.P @ F_jac.T + L_jac @ Q @ L_jac.T
        return self.x_nom.copy(), self.P.copy()

    # Alias matching strapdown dead-reckoning terminology
    propagate = predict

    def update(
        self,
        y: np.ndarray,
        h_func,
        H_jac: np.ndarray,
        R: np.ndarray,
        M_jac: Optional[np.ndarray] = None,
        inject_func=None,
        angle_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Executes Generic Measurement Update, Nominal Injection, and Error Reset:
            1. Innovation:            nu = y - h(check_x_nom)
            2. Innovation Covariance: S  = H * check_P * H^T + M * R * M^T
            3. Error Kalman Gain:     K  = check_P * H^T * inv(S)
            4. Error Correction:      delta_x = K * nu
            5. State Injection:       hat_x_nom = check_x_nom (oplus) delta_x
            6. Error Reset:           delta_x <- 0
            7. Covariance Reset:      hat_P = (I - K*H) * check_P * (I - K*H)^T + K * R_eff * K^T
        """
        y_vec = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        m = y_vec.shape[0]
        if M_jac is None:
            M_jac = np.eye(m, dtype=np.float64)

        # 1. Innovation residual
        y_pred = h_func(self.x_nom).reshape(-1, 1) if callable(h_func) else np.asarray(h_func, dtype=np.float64).reshape(-1, 1)
        nu = y_vec - y_pred
        if angle_indices is not None:
            for idx in angle_indices:
                nu[idx, 0] = float((nu[idx, 0] + np.pi) % (2 * np.pi) - np.pi)

        # 2. Innovation Covariance
        R_eff = M_jac @ R @ M_jac.T
        S = H_jac @ self.P @ H_jac.T + R_eff

        # 3. Optimal Error Kalman Gain
        K = self.P @ H_jac.T @ np.linalg.inv(S)

        # 4. Error State Estimate
        delta_x = K @ nu

        # 5. Nominal State Injection
        if inject_func is not None:
            self.x_nom = inject_func(self.x_nom, delta_x).reshape(-1, 1)
        else:
            self.x_nom = (self.x_nom + delta_x).reshape(-1, 1)

        # 6 & 7. Covariance Reset (Joseph form for numerical stability)
        I_KH = np.eye(self.n, dtype=np.float64) - K @ H_jac
        self.P = I_KH @ self.P @ I_KH.T + K @ R_eff @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.latest_innovation = nu
        self.latest_innovation_cov = S
        self.latest_gain = K

        return self.x_nom.copy(), self.P.copy()


