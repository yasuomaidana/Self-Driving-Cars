"""Unscented Kalman Filter (UKF) module.

Implements the Scaled Unscented Transform (UT) with deterministic sigma points:
- Zero analytical Jacobians required (F, L, H, M are never computed).
- Handles arbitrary non-linear dynamic models f(x, u) and sensor models h(x).
- Captures higher-order moments (up to 3rd order for Gaussians).
"""

from typing import Callable, Optional, Tuple
import numpy as np


def wraptopi(angle):
    """Normalize angle to [-pi, pi] for scalars or numpy arrays."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


class UnscentedKalmanFilter:
    """Unscented Kalman Filter (UKF) using the standard non-scaled formulation (Julier-Uhlmann).
    
    Parameters:
        f_func: Optional state transition function f(x, u)
        h_func: Optional measurement function h(x)
        Q: Process noise covariance matrix
        R: Measurement noise covariance matrix
        x0: Initial state vector (N,) or (N, 1)
        P0: Initial covariance matrix (N, N)
        kappa: UT parameter kappa (default: 0.0)
        angle_indices: List of state dimension indices representing angles (radians)
    """

    def __init__(
        self,
        f_func: Optional[Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]] = None,
        h_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        kappa: float = 0.0,
        angle_indices: Optional[list] = None
    ):
        self.f = f_func
        self.h = h_func
        self.Q = np.asarray(Q, dtype=float) if Q is not None else None
        self.R = np.asarray(R, dtype=float) if R is not None else None
        
        if x0 is not None:
            self.x = np.asarray(x0, dtype=float).reshape(-1, 1)
            self.n = len(self.x)
            self.L = self.n
        else:
            self.x = None
            self.n = 0
            self.L = 0

        self.P = np.asarray(P0, dtype=float) if P0 is not None else None
        self.angle_indices = angle_indices or []
        self.kappa = float(kappa)
        
        if self.n > 0:
            self._compute_weights()
            
        self.sigma_pts_pred: Optional[np.ndarray] = None
        self.latest_innovation: Optional[np.ndarray] = None
        self.latest_innovation_cov: Optional[np.ndarray] = None
        self.latest_gain: Optional[np.ndarray] = None

    def _compute_weights(self):
        """Compute standard non-scaled UKF weights: alpha_0 = kappa / (N + kappa), alpha_i = 1 / (2*(N + kappa))."""
        self.num_sigmas = 2 * self.n + 1
        denom = self.n + self.kappa
        self.gamma = np.sqrt(denom)
        
        self.alpha_weights = np.full(self.num_sigmas, 1.0 / (2.0 * denom))
        self.alpha_weights[0] = self.kappa / denom
        
        # Mean and Covariance weights are identical in standard non-scaled UKF
        self.Wm = self.alpha_weights
        self.Wc = self.alpha_weights

    def generate_sigma_points(
        self,
        x_mean: np.ndarray,
        P_cov: np.ndarray,
        angle_indices: Optional[list] = None
    ) -> np.ndarray:
        """Generate 2N + 1 deterministic sigma points from N(x_mean, P_cov)."""
        x_mean_arr = np.asarray(x_mean, dtype=float).reshape(-1, 1)
        n = len(x_mean_arr)
        if self.n != n:
            self.n = n
            self.L = n
            self._compute_weights()

        P_sym = 0.5 * (P_cov + P_cov.T)
        try:
            L_mat = np.linalg.cholesky(P_sym + np.eye(n) * 1e-12)
        except np.linalg.LinAlgError:
            eigval, eigvec = np.linalg.eigh(P_sym)
            eigval = np.maximum(eigval, 1e-12)
            L_mat = eigvec @ np.diag(np.sqrt(eigval))

        sigma = np.zeros((2 * n + 1, n))
        x_flat = x_mean_arr.flatten()
        sigma[0] = x_flat
        
        for i in range(n):
            offset = self.gamma * L_mat[:, i]
            sigma[i + 1]     = x_flat + offset
            sigma[i + 1 + n] = x_flat - offset

        if angle_indices:
            for idx in angle_indices:
                if idx < n:
                    sigma[:, idx] = wraptopi(sigma[:, idx])
            
        return sigma

    def predict(
        self,
        f_func: Optional[Callable] = None,
        Q: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        angle_indices: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute UKF time propagation."""
        f = f_func or self.f
        if f is None:
            raise ValueError("State transition function f_func must be provided.")
        Q_mat = Q if Q is not None else self.Q
        if Q_mat is None:
            raise ValueError("Process noise covariance Q must be provided.")
            
        a_indices = angle_indices if angle_indices is not None else self.angle_indices
        
        sigma_prior = self.generate_sigma_points(self.x, self.P, angle_indices=a_indices)
        num_sigma = 2 * self.n + 1
        sigma_pred = np.zeros((num_sigma, self.n))
        
        for i in range(num_sigma):
            pt = sigma_prior[i].reshape(self.n, 1)
            if u is not None:
                try:
                    res = f(pt, u)
                except TypeError:
                    res = f(pt)
            else:
                try:
                    res = f(pt)
                except TypeError:
                    res = f(pt, None)
            sigma_pred[i] = np.asarray(res, dtype=float).flatten()
            
        self.sigma_pts_pred = sigma_pred
        
        # Compute predicted mean with optional circular angle unwrapping
        x_pred = np.zeros((self.n, 1))
        for row in range(self.n):
            if a_indices and row in a_indices:
                ref = sigma_pred[0, row]
                unwrapped = ref + wraptopi(sigma_pred[:, row] - ref)
                x_pred[row, 0] = wraptopi(np.sum(self.Wm * unwrapped))
            else:
                x_pred[row, 0] = np.sum(self.Wm * sigma_pred[:, row])
                
        diff = sigma_pred - x_pred.flatten()
        if a_indices:
            for idx in a_indices:
                if idx < self.n:
                    diff[:, idx] = wraptopi(diff[:, idx])
                    
        P_pred = np.zeros((self.n, self.n))
        for i in range(num_sigma):
            P_pred += self.Wc[i] * np.outer(diff[i], diff[i])
        P_pred += Q_mat
        
        self.x = x_pred
        self.P = 0.5 * (P_pred + P_pred.T)
        return self.x.copy(), self.P.copy()

    def update(
        self,
        y: np.ndarray,
        h_func: Optional[Callable] = None,
        R: Optional[np.ndarray] = None,
        meas_angle_indices: Optional[list] = None,
        state_angle_indices: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute UKF measurement update."""
        h = h_func or self.h
        if h is None:
            raise ValueError("Measurement function h_func must be provided.")
        R_mat = R if R is not None else self.R
        if R_mat is None:
            raise ValueError("Measurement noise covariance R must be provided.")
        R_mat = np.asarray(R_mat, dtype=float)
        if R_mat.ndim == 1:
            R_mat = np.diag(R_mat) if len(R_mat) > 1 else np.array([[R_mat[0]]])
        elif R_mat.ndim == 0:
            R_mat = np.array([[float(R_mat)]])

        y_meas = np.asarray(y, dtype=float).reshape(-1, 1)
        m = len(y_meas)
        
        s_indices = state_angle_indices if state_angle_indices is not None else self.angle_indices
        m_indices = meas_angle_indices or []
        
        sigma_pts = self.generate_sigma_points(self.x, self.P, angle_indices=s_indices)
        num_sigma = 2 * self.n + 1
        
        Y_sigma = np.zeros((num_sigma, m))
        for i in range(num_sigma):
            pt = sigma_pts[i].reshape(self.n, 1)
            Y_sigma[i] = np.asarray(h(pt), dtype=float).flatten()
            
        y_hat = np.zeros((m, 1))
        for row in range(m):
            if m_indices and row in m_indices:
                ref = Y_sigma[0, row]
                unwrapped = ref + wraptopi(Y_sigma[:, row] - ref)
                y_hat[row, 0] = wraptopi(np.sum(self.Wm * unwrapped))
            else:
                y_hat[row, 0] = np.sum(self.Wm * Y_sigma[:, row])
        
        S = np.zeros((m, m))
        P_xy = np.zeros((self.n, m))
        
        diff_y = Y_sigma - y_hat.flatten()
        diff_x = sigma_pts - self.x.flatten()
        
        if m_indices:
            for idx in m_indices:
                if idx < m:
                    diff_y[:, idx] = wraptopi(diff_y[:, idx])
        if s_indices:
            for idx in s_indices:
                if idx < self.n:
                    diff_x[:, idx] = wraptopi(diff_x[:, idx])

        for i in range(num_sigma):
            S += self.Wc[i] * np.outer(diff_y[i], diff_y[i])
            P_xy += self.Wc[i] * np.outer(diff_x[i], diff_y[i])
            
        S += R_mat
        
        K = P_xy @ np.linalg.inv(S)
        
        nu = y_meas - y_hat
        if m_indices:
            for idx in m_indices:
                if idx < m:
                    nu[idx, 0] = wraptopi(nu[idx, 0])
                
        self.x = self.x + K @ nu
        if s_indices:
            for idx in s_indices:
                if idx < self.n:
                    self.x[idx, 0] = wraptopi(self.x[idx, 0])
                    
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        self.latest_innovation = nu
        self.latest_innovation_cov = S
        self.latest_gain = K
        
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
        P0: Optional[np.ndarray] = None,
        kappa: float = 0.0
    ):
        self.dt = dt
        self.S = S
        self.D = D
        
        Q_mat = np.eye(2) * 0.1 if Q is None else Q
        R_mat = np.array([[R]])
        x_init = np.array([0.0, 5.0]) if x0 is None else x0
        P_init = np.array([[0.01, 0.0], [0.0, 1.0]]) if P0 is None else P0

        def f_motion(x, u=None):
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
            P0=P_init,
            kappa=kappa
        )

    def step(self, u: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one predict + update step with zero Jacobians."""
        self.ukf.predict(u=np.array([u]))
        return self.ukf.update(np.array([y]))
