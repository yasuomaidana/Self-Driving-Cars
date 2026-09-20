"""Extended Kalman Filter (EKF) module with analytical Jacobians.

Contains:
- ExtendedKalmanFilter: Generic non-linear EKF solver.
- LandmarkBearingEKF: Optical landmark bearing filter matching Excersice.ipynb.
- Radar2DTargetTrackerEKF: 2D Polar Radar target tracker (range & bearing).
- UnicycleKinematicsEKF: Non-linear unicycle kinematics with non-additive noise Jacobian L(x).
"""

from typing import Callable, Optional, Tuple
import numpy as np


def wraptopi(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class ExtendedKalmanFilter:
    """Generic Non-linear Extended Kalman Filter (EKF)."""

    def __init__(
        self,
        f_func: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        h_func: Callable[[np.ndarray], np.ndarray],
        F_jacobian: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        L_jacobian: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        H_jacobian: Callable[[np.ndarray], np.ndarray],
        M_jacobian: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        angle_indices: Optional[list] = None
    ):
        self.f = f_func
        self.h = h_func
        self.calc_F = F_jacobian
        self.calc_L = L_jacobian
        self.calc_H = H_jacobian
        self.calc_M = M_jacobian
        
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.x = np.asarray(x0, dtype=float).reshape(-1, 1)
        self.P = np.asarray(P0, dtype=float)
        self.n = len(self.x)
        self.angle_indices = angle_indices or []

    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate state non-linearly and covariance via Jacobians F and L."""
        self.x = self.f(self.x, u).reshape(self.n, 1)
        F = self.calc_F(self.x, u)
        L = self.calc_L(self.x, u)
        self.P = F @ self.P @ F.T + L @ self.Q @ L.T
        return self.x.copy(), self.P.copy()

    def update(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state using non-linear observation h(x) and Jacobians H and M."""
        y_vec = np.asarray(y, dtype=float).reshape(-1, 1)
        m = len(y_vec)
        
        y_hat = self.h(self.x).reshape(m, 1)
        nu = y_vec - y_hat
        for idx in self.angle_indices:
            if idx < len(nu):
                nu[idx, 0] = wraptopi(nu[idx, 0])
                
        H = self.calc_H(self.x)
        M = self.calc_M(self.x)
        
        S = H @ self.P @ H.T + M @ self.R @ M.T
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ nu
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P
        
        return self.x.copy(), self.P.copy()


class LandmarkBearingEKF:
    """Optical Landmark Bearing EKF matching 'The Nonlinear Kalman Filter/Excersice.ipynb'."""

    def __init__(
        self,
        dt: float = 0.5,
        S: float = 20.0,
        D: float = 40.0,
        Q: Optional[np.ndarray] = None,
        R: float = 0.01,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None
    ):
        self.dt = dt
        self.S = S
        self.D = D
        
        Q_mat = np.eye(2) * 0.1 if Q is None else Q
        R_mat = np.array([[R]])
        x_init = np.array([0.0, 5.0]) if x0 is None else x0
        P_init = np.array([[0.01, 0.0], [0.0, 1.0]]) if P0 is None else P0
        
        def f_motion(x, u):
            acc = float(np.asarray(u).flatten()[0]) if u is not None else 0.0
            x_f = np.asarray(x).flatten()
            p, v = float(x_f[0]), float(x_f[1])
            return np.array([[p + v * self.dt], [v + acc * self.dt]])
        
        def h_sensor(x):
            x_f = np.asarray(x).flatten()
            p = float(x_f[0])
            return np.array([[np.arctan(self.S / (self.D - p))]])
        
        def F_jac(x, u):
            return np.array([[1.0, self.dt], [0.0, 1.0]])
        
        def L_jac(x, u):
            return np.eye(2)
        
        def H_jac(x):
            x_f = np.asarray(x).flatten()
            p = float(x_f[0])
            denom = (self.D - p)**2 + self.S**2
            return np.array([[self.S / denom, 0.0]])
        
        def M_jac(x):
            return np.array([[1.0]])

        self.ekf = ExtendedKalmanFilter(
            f_func=f_motion,
            h_func=h_sensor,
            F_jacobian=F_jac,
            L_jacobian=L_jac,
            H_jacobian=H_jac,
            M_jacobian=M_jac,
            Q=Q_mat,
            R=R_mat,
            x0=x_init,
            P0=P_init
        )

    def step(self, u: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one predict + update step."""
        self.ekf.predict(np.array([u]))
        return self.ekf.update(np.array([y]))


class Radar2DTargetTrackerEKF:
    """2D Radar Target Tracking with Polar Measurements (r, phi)."""

    def __init__(
        self,
        dt: float = 0.1,
        sigma_range: float = 0.3,
        sigma_bearing: float = 0.02,
        sigma_acc: float = 0.5,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None
    ):
        self.dt = dt
        
        F_mat = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        Q_mat = np.diag([(dt**3)/3 * sigma_acc**2, (dt**3)/3 * sigma_acc**2, dt * sigma_acc**2, dt * sigma_acc**2])
        R_mat = np.diag([sigma_range**2, sigma_bearing**2])
        
        x_init = np.array([10.0, 10.0, 5.0, 0.0]) if x0 is None else x0
        P_init = np.eye(4) * 5.0 if P0 is None else P0

        def f_motion(x, u):
            return F_mat @ x

        def h_sensor(x):
            x_f = np.asarray(x).flatten()
            px, py = float(x_f[0]), float(x_f[1])
            r = np.sqrt(px**2 + py**2)
            phi = np.arctan2(py, px)
            return np.array([[r], [phi]])

        def F_jac(x, u):
            return F_mat

        def L_jac(x, u):
            return np.eye(4)

        def H_jac(x):
            x_f = np.asarray(x).flatten()
            px, py = float(x_f[0]), float(x_f[1])
            r2 = px**2 + py**2
            r = np.sqrt(r2)
            if r < 1e-6:
                r, r2 = 1e-6, 1e-12
            return np.array([
                [px / r,       py / r,       0.0, 0.0],
                [-py / r2,     px / r2,      0.0, 0.0]
            ])

        def M_jac(x):
            return np.eye(2)

        self.ekf = ExtendedKalmanFilter(
            f_func=f_motion,
            h_func=h_sensor,
            F_jacobian=F_jac,
            L_jacobian=L_jac,
            H_jacobian=H_jac,
            M_jacobian=M_jac,
            Q=Q_mat,
            R=R_mat,
            x0=x_init,
            P0=P_init,
            angle_indices=[1]
        )
