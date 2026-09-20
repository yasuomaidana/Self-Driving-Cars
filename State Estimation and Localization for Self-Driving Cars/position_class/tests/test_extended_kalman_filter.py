"""Unit Tests for Day 2 Generic Extended Kalman Filter (EKF) Module."""

import math
import numpy as np
import pytest

from position_class.extended_kalman_filter import (
    ExtendedKalmanFilter,
    LandmarkBearingEKF,
    Radar2DTargetTrackerEKF,
    wraptopi,
)


class TestWrapToPi:
    """Test suite for angle normalization wraptopi."""

    def test_within_range(self):
        assert math.isclose(wraptopi(0.0), 0.0, abs_tol=1e-7)
        assert math.isclose(wraptopi(1.5), 1.5, abs_tol=1e-7)
        assert math.isclose(wraptopi(-2.0), -2.0, abs_tol=1e-7)

    def test_boundary_values(self):
        assert math.isclose(abs(wraptopi(np.pi)), np.pi, abs_tol=1e-7)
        assert math.isclose(wraptopi(-np.pi), -np.pi, abs_tol=1e-7)

    def test_multi_turn_positive(self):
        # 3pi -> pi
        val = wraptopi(3 * np.pi)
        assert math.isclose(abs(val), np.pi, abs_tol=1e-6)

        # 2pi + 0.5 -> 0.5
        assert math.isclose(wraptopi(2 * np.pi + 0.5), 0.5, abs_tol=1e-6)

    def test_multi_turn_negative(self):
        # -3pi -> -pi (or pi)
        val = wraptopi(-3 * np.pi)
        assert math.isclose(abs(val), np.pi, abs_tol=1e-6)

        # -2pi - 0.5 -> -0.5
        assert math.isclose(wraptopi(-2 * np.pi - 0.5), -0.5, abs_tol=1e-6)


class TestGenericExtendedKalmanFilter:
    """Test suite for the generic multi-dimensional ExtendedKalmanFilter class."""

    @pytest.fixture
    def linear_cv_ekf(self):
        """Creates a 2D constant-velocity EKF configured with linear dynamics."""
        dt = 0.1
        f_func = lambda x, u: np.array([[x[0, 0] + dt * x[1, 0]], [x[1, 0]]])
        h_func = lambda x: np.array([[x[0, 0]]])
        F_jac = lambda x, u: np.array([[1.0, dt], [0.0, 1.0]])
        L_jac = lambda x, u: np.eye(2)
        H_jac = lambda x: np.array([[1.0, 0.0]])
        M_jac = lambda x: np.array([[1.0]])

        Q = np.diag([0.01, 0.05])
        R = np.array([[0.1]])
        x0 = np.array([0.0, 5.0])
        P0 = np.eye(2) * 1.0

        return ExtendedKalmanFilter(
            f_func=f_func,
            h_func=h_func,
            F_jacobian=F_jac,
            L_jacobian=L_jac,
            H_jacobian=H_jac,
            M_jacobian=M_jac,
            Q=Q,
            R=R,
            x0=x0,
            P0=P0,
        )

    def test_initialization(self, linear_cv_ekf):
        ekf = linear_cv_ekf
        assert ekf.n == 2
        assert ekf.x.shape == (2, 1)
        assert ekf.P.shape == (2, 2)
        assert math.isclose(ekf.x[0, 0], 0.0)
        assert math.isclose(ekf.x[1, 0], 5.0)

    def test_predict_step(self, linear_cv_ekf):
        ekf = linear_cv_ekf
        x_pred, P_pred = ekf.predict()

        # x = [0 + 0.1*5, 5] = [0.5, 5.0]
        assert math.isclose(x_pred[0, 0], 0.5, abs_tol=1e-5)
        assert math.isclose(x_pred[1, 0], 5.0, abs_tol=1e-5)
        # Covariance grows after prediction
        assert P_pred[0, 0] > 1.0
        assert P_pred[1, 1] > 1.0

    def test_update_step_reduces_uncertainty(self, linear_cv_ekf):
        ekf = linear_cv_ekf
        ekf.predict()
        cov_prior = ekf.P[0, 0]

        # Update with measurement y = 0.6
        x_post, P_post = ekf.update(np.array([0.6]))
        cov_post = P_post[0, 0]

        assert cov_post < cov_prior
        # Corrected position moves towards measurement
        assert 0.5 < x_post[0, 0] < 0.6

    def test_angle_wrapping_in_update(self):
        """Tests that angular measurement residuals are wrapped properly across +/- pi."""
        # 1D heading filter measuring angle
        f_func = lambda x, u: x.copy()
        h_func = lambda x: x.copy()
        F_jac = lambda x, u: np.eye(1)
        L_jac = lambda x, u: np.eye(1)
        H_jac = lambda x: np.eye(1)
        M_jac = lambda x: np.eye(1)

        Q = np.array([[0.01]])
        R = np.array([[0.1]])
        # Prior is at +3.10 rad (near +pi)
        x0 = np.array([3.10])
        P0 = np.array([[0.5]])

        ekf = ExtendedKalmanFilter(
            f_func=f_func,
            h_func=h_func,
            F_jacobian=F_jac,
            L_jacobian=L_jac,
            H_jacobian=H_jac,
            M_jacobian=M_jac,
            Q=Q,
            R=R,
            x0=x0,
            P0=P0,
            angle_indices=[0],
        )

        # Measurement at -3.10 rad (near -pi, true angular difference is only +0.083 rad, not -6.2 rad)
        ekf.predict()
        x_post, P_post = ekf.update(np.array([-3.10]))

        # State should smoothly wrap across pi rather than jump around the whole circle
        assert x_post[0, 0] > 3.0 or x_post[0, 0] < -3.0
        assert P_post[0, 0] < 0.5


class TestLandmarkBearingEKF:
    """Test suite for optical LandmarkBearingEKF."""

    def test_initialization(self):
        l_ekf = LandmarkBearingEKF(dt=0.5, S=20.0, D=40.0)
        assert l_ekf.dt == 0.5
        assert l_ekf.S == 20.0
        assert l_ekf.D == 40.0
        assert l_ekf.ekf.x.shape == (2, 1)

    def test_step_convergence(self):
        """Test that LandmarkBearingEKF state tracks truth over multiple steps."""
        dt = 0.5
        S = 20.0
        D = 40.0
        l_ekf = LandmarkBearingEKF(dt=dt, S=S, D=D, x0=np.array([0.0, 4.8]))

        # True trajectory: p(t) = 5.0 * t, v = 5.0
        true_p = 0.0
        true_v = 5.0

        for step in range(10):
            true_p += true_v * dt
            # True bearing measurement
            true_bearing = np.arctan(S / (D - true_p))
            # Sensor noise
            meas_bearing = true_bearing + np.random.normal(0, 0.005)

            x_est, P_est = l_ekf.step(u=0.0, y=meas_bearing)

        # After 10 steps, estimated position should be close to true_p
        est_p = x_est[0, 0]
        est_v = x_est[1, 0]
        assert math.isclose(est_p, true_p, abs_tol=1.5)
        assert math.isclose(est_v, true_v, abs_tol=1.0)
        # Covariance should remain bounded and positive definite
        assert 0.0 < P_est[0, 0] < 5.0
        assert np.linalg.det(P_est) > 0.0


class TestRadar2DTargetTrackerEKF:
    """Test suite for 2D Polar Radar target tracker EKF."""

    def test_initialization(self):
        radar_ekf = Radar2DTargetTrackerEKF(
            dt=0.1,
            sigma_range=0.3,
            sigma_bearing=0.02,
            sigma_acc=0.5,
            x0=np.array([10.0, 10.0, 5.0, 0.0]),
        )
        assert radar_ekf.dt == 0.1
        assert radar_ekf.ekf.n == 4
        assert radar_ekf.ekf.angle_indices == [1]

    def test_predict_and_update(self):
        radar_ekf = Radar2DTargetTrackerEKF(
            dt=0.1,
            x0=np.array([10.0, 10.0, 2.0, 0.0]),
        )
        # Predict 1 step
        x_pred, P_pred = radar_ekf.ekf.predict()
        # px -> 10 + 0.1 * 2 = 10.2, py -> 10.0
        assert math.isclose(x_pred[0, 0], 10.2, abs_tol=1e-5)
        assert math.isclose(x_pred[1, 0], 10.0, abs_tol=1e-5)

        # Measurement at expected position
        r_meas = math.sqrt(10.2**2 + 10.0**2)
        phi_meas = math.atan2(10.0, 10.2)
        x_post, P_post = radar_ekf.ekf.update(np.array([r_meas, phi_meas]))

        assert math.isclose(x_post[0, 0], 10.2, abs_tol=0.2)
        assert math.isclose(x_post[1, 0], 10.0, abs_tol=0.2)
        assert P_post[0, 0] < P_pred[0, 0]
