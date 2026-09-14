"""Unscented Kalman Filter (UKF) module.

Implements the Scaled Unscented Transform (UT) with deterministic sigma points:
- Zero analytical Jacobians required (F, L, H, M are never computed).
- Handles arbitrary non-linear dynamic models f(x, u) and sensor models h(x).
- Captures higher-order moments (up to 3rd order for Gaussians).
"""

from typing import Callable, Optional, Tuple
import numpy as np


def wraptopi(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class UnscentedKalmanFilter:
    """Scaled Unscented Kalman Filter (UKF) (Julier-Uhlmann & Wan-Van der Merwe)."""

    def __init__(
        self,
        f_func: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        h_func: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        angle_indices: Optional[list] = None
    ):
        self.f = f_func
        self.h = h_func
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.x = np.asarray(x0, dtype=float).reshape(-1, 1)
        self.P = np.asarray(P0, dtype=float)
        
        self.n = len(self.x)
        self.m = self.R.shape[0] if self.R.ndim > 1 else 1
        self.angle_indices = angle_indices or []
        
        # UT scaling parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.lam = (alpha**2) * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lam)
        
        # Compute weights
        num_sigma = 2 * self.n + 1
        self.Wm = np.full(num_sigma, 1.0 / (2.0 * (self.n + self.lam)))
        self.Wc = np.full(num_sigma, 1.0 / (2.0 * (self.n + self.lam)))
        
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1.0 - alpha**2 + beta)
        
        # Internal cache for sigma points
        self.sigma_pts_pred: Optional[np.ndarray] = None

    def generate_sigma_points(self, x_mean: np.ndarray, P_cov: np.ndarray) -> np.ndarray:
        """Generate 2n + 1 deterministic sigma points from N(x_mean, P_cov)."""
        P_reg = P_cov + np.eye(self.n) * 1e-12
        try:
            L = np.linalg.cholesky(P_reg)
        except np.linalg.LinAlgError:
            eigval, eigvec = np.linalg.eigh(P_cov)
            eigval = np.maximum(eigval, 1e-12)
            L = eigvec @ np.diag(np.sqrt(eigval))

        sigma = np.zeros((2 * self.n + 1, self.n))
        x_flat = x_mean.flatten()
        sigma[0] = x_flat
        
        for i in range(self.n):
            offset = self.gamma * L[:, i]
            sigma[i + 1]          = x_flat + offset
            sigma[i + 1 + self.n] = x_flat - offset
            
        return sigma

    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Execute UKF time propagation."""
        sigma_prior = self.generate_sigma_points(self.x, self.P)
        num_sigma = 2 * self.n + 1
        sigma_pred = np.zeros((num_sigma, self.n))
        for i in range(num_sigma):
            pt = sigma_prior[i].reshape(self.n, 1)
            sigma_pred[i] = self.f(pt, u).flatten()
            
        self.sigma_pts_pred = sigma_pred
        
        self.x = np.sum(self.Wm[:, None] * sigma_pred, axis=0).reshape(self.n, 1)
        
        diff = sigma_pred - self.x.flatten()
        self.P = np.zeros((self.n, self.n))
        for i in range(num_sigma):
            self.P += self.Wc[i] * np.outer(diff[i], diff[i])
        self.P += self.Q
        
        return self.x.copy(), self.P.copy()

    def update(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Execute UKF measurement update."""
        y_meas = np.asarray(y, dtype=float).reshape(-1, 1)
        m = len(y_meas)
        
        sigma_pts = self.generate_sigma_points(self.x, self.P)
        num_sigma = 2 * self.n + 1
        
        Y_sigma = np.zeros((num_sigma, m))
        for i in range(num_sigma):
            pt = sigma_pts[i].reshape(self.n, 1)
            Y_sigma[i] = self.h(pt).flatten()
            
        y_hat = np.sum(self.Wm[:, None] * Y_sigma, axis=0).reshape(m, 1)
        
        S = np.zeros((m, m))
        P_xy = np.zeros((self.n, m))
        
        diff_y = Y_sigma - y_hat.flatten()
        diff_x = sigma_pts - self.x.flatten()
        
        for idx in self.angle_indices:
            if idx < m:
                for i in range(num_sigma):
                    diff_y[i, idx] = wraptopi(diff_y[i, idx])

        for i in range(num_sigma):
            S += self.Wc[i] * np.outer(diff_y[i], diff_y[i])
            P_xy += self.Wc[i] * np.outer(diff_x[i], diff_y[i])
            
        R_mat = self.R if self.R.ndim > 1 else np.array([[self.R]])
        S += R_mat
        
        K = P_xy @ np.linalg.inv(S)
        
        nu = y_meas - y_hat
        for idx in self.angle_indices:
            if idx < m:
                nu[idx, 0] = wraptopi(nu[idx, 0])
                
        self.x = self.x + K @ nu
        self.P = self.P - K @ S @ K.T
        
        return self.x.copy(), self.P.copy()


class LandmarkBearingUKF:
    """Optical Landmark Bearing UKF matching 'The Nonlinear Kalman Filter/Excersice.ipynb'."""

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

        self.ukf = UnscentedKalmanFilter(
            f_func=f_motion,
            h_func=h_sensor,
            Q=Q_mat,
            R=R_mat,
            x0=x_init,
            P0=P_init
        )

    def step(self, u: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one predict + update step with zero Jacobians."""
        self.ukf.predict(np.array([u]))
        return self.ukf.update(np.array([y]))
