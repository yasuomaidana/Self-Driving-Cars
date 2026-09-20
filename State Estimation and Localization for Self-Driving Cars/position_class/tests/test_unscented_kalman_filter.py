"""Unit Tests for Day 2 Generic Unscented Kalman Filter (UKF) Module."""

import math
import numpy as np
import pytest

from position_class.unscented_kalman_filter import (
    UnscentedKalmanFilter,
    LandmarkBearingUKF,
    wraptopi,
)


class TestGenericUnscentedKalmanFilter:
    """Test suite for the generic non-scaled UnscentedKalmanFilter class."""

    @pytest.fixture
    def linear_cv_ukf(self):
        """Creates a 2D constant-velocity UKF."""
        dt = 0.1
        f_func = lambda x, u=None: np.array([[x[0, 0] + dt * x[1, 0]], [x[1, 0]]])
        h_func = lambda x: np.array([[x[0, 0]]])

        Q = np.diag([0.01, 0.05])
        R = np.array([[0.1]])
        x0 = np.array([0.0, 5.0])
        P0 = np.eye(2) * 1.0

        return UnscentedKalmanFilter(
            f_func=f_func,
            h_func=h_func,
            Q=Q,
            R=R,
            x0=x0,
            P0=P0,
            kappa=1.0,
        )

    def test_initialization_and_weights(self, linear_cv_ukf):
        ukf = linear_cv_ukf
        assert ukf.n == 2
        assert len(ukf.Wm) == 2 * ukf.n + 1
        assert len(ukf.Wc) == 2 * ukf.n + 1
        assert ukf.num_sigmas == 5

        # For N=2, kappa=1.0:
        # alpha[0] = kappa / (N + kappa) = 1/3
        # alpha[i] = 1 / (2 * (N + kappa)) = 1/6
        assert math.isclose(ukf.alpha_weights[0], 1.0 / 3.0, abs_tol=1e-7)
        assert math.isclose(ukf.alpha_weights[1], 1.0 / 6.0, abs_tol=1e-7)
        assert math.isclose(np.sum(ukf.alpha_weights), 1.0, abs_tol=1e-7)

        # Sum of mean & covariance weights should be 1.0
        assert math.isclose(np.sum(ukf.Wm), 1.0, abs_tol=1e-7)
        assert math.isclose(np.sum(ukf.Wc), 1.0, abs_tol=1e-7)

    def test_sigma_points_generation_mean_and_cov(self, linear_cv_ukf):
        ukf = linear_cv_ukf
        x_mean = np.array([[2.0], [-1.0]])
        P_cov = np.array([[0.5, 0.1], [0.1, 0.8]])

        sigma = ukf.generate_sigma_points(x_mean, P_cov)
        assert sigma.shape == (2 * 2 + 1, 2)

        # Reconstructed mean from sigma points must equal x_mean
        recon_mean = np.sum(ukf.Wm[:, None] * sigma, axis=0)
        assert math.isclose(recon_mean[0], 2.0, abs_tol=1e-6)
        assert math.isclose(recon_mean[1], -1.0, abs_tol=1e-6)

        # Reconstructed covariance from sigma points must equal P_cov (up to machine precision)
        diff = sigma - recon_mean
        recon_cov = np.zeros((2, 2))
        for i in range(len(ukf.Wc)):
            recon_cov += ukf.Wc[i] * np.outer(diff[i], diff[i])
        np.testing.assert_allclose(recon_cov, P_cov, atol=1e-4)

    def test_sigma_points_cholesky_fallback(self, linear_cv_ukf):
        """Test that generate_sigma_points handles non-strictly-positive-definite covariance via eigenvalue fallback."""
        ukf = linear_cv_ukf
        # Semi-definite / singular covariance
        P_singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        x_mean = np.array([[0.0], [0.0]])

        sigma = ukf.generate_sigma_points(x_mean, P_singular)
        assert sigma.shape == (5, 2)
        assert not np.isnan(sigma).any()

    def test_predict_step(self, linear_cv_ukf):
        ukf = linear_cv_ukf
        x_pred, P_pred = ukf.predict()

        # x = [0 + 0.1*5, 5] = [0.5, 5.0]
        assert math.isclose(x_pred[0, 0], 0.5, abs_tol=1e-4)
        assert math.isclose(x_pred[1, 0], 5.0, abs_tol=1e-4)
        # Covariance grows after prediction
        assert P_pred[0, 0] > 1.0
        assert P_pred[1, 1] > 1.0

    def test_update_step_reduces_uncertainty(self, linear_cv_ukf):
        ukf = linear_cv_ukf
        ukf.predict()
        cov_prior = ukf.P[0, 0]

        # Update with measurement y = 0.6
        x_post, P_post = ukf.update(np.array([0.6]))
        cov_post = P_post[0, 0]

        assert cov_post < cov_prior
        # Corrected position moves towards measurement
        assert 0.5 < x_post[0, 0] < 0.6

    def test_angle_indices_wrapping(self):
        """Test UKF angle wrapping in measurement update."""
        # 1D heading state
        f_func = lambda x, u: x.copy()
        h_func = lambda x: x.copy()
        Q = np.array([[0.01]])
        R = np.array([[0.1]])
        x0 = np.array([3.10])
        P0 = np.array([[0.5]])

        ukf = UnscentedKalmanFilter(
            f_func=f_func,
            h_func=h_func,
            Q=Q,
            R=R,
            x0=x0,
            P0=P0,
            angle_indices=[0],
        )

        ukf.predict()
        x_post, P_post = ukf.update(np.array([-3.10]))

        assert not np.isnan(x_post).any()
        assert P_post[0, 0] < 0.5


class TestLandmarkBearingUKF:
    """Test suite for optical LandmarkBearingUKF."""

    def test_initialization(self):
        l_ukf = LandmarkBearingUKF(dt=0.5, S=20.0, D=40.0)
        assert l_ukf.dt == 0.5
        assert l_ukf.S == 20.0
        assert l_ukf.D == 40.0
        assert l_ukf.ukf.n == 2

    def test_step_convergence(self):
        """Test that LandmarkBearingUKF tracks ground truth without analytical Jacobians."""
        dt = 0.5
        S = 20.0
        D = 40.0
        l_ukf = LandmarkBearingUKF(dt=dt, S=S, D=D, x0=np.array([0.0, 4.8]))

        true_p = 0.0
        true_v = 5.0

        for step in range(10):
            true_p += true_v * dt
            true_bearing = np.arctan(S / (D - true_p))
            meas_bearing = true_bearing + np.random.normal(0, 0.005)

            x_est, P_est = l_ukf.step(u=0.0, y=meas_bearing)

        est_p = x_est[0, 0]
        est_v = x_est[1, 0]
        assert math.isclose(est_p, true_p, abs_tol=1.5)
        assert math.isclose(est_v, true_v, abs_tol=1.0)
        # Covariance should remain bounded and positive definite
        assert 0.0 < P_est[0, 0] < 5.0
        assert np.linalg.det(P_est) > 0.0
