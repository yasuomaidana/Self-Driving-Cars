"""Unit Tests for Day 2 Error-State Extended Kalman Filter (ES-EKF) Module."""

import math
import numpy as np
import pytest

from position_class.es_ekf import (
    ErrorStateEKF,
    ErrorStateKalmanFilter,
)
from position_class.inertial_navigation import Quaternion, skew_symmetric


class TestGenericErrorStateKalmanFilter:
    """Test suite for the generic N-dimensional ErrorStateKalmanFilter class."""

    def test_initialization(self):
        x0 = np.array([1.0, 2.0, 3.0])
        P0 = np.eye(3) * 0.5
        es_kf = ErrorStateKalmanFilter(x0=x0, P0=P0)

        assert es_kf.n == 3
        assert es_kf.x_nom.shape == (3, 1)
        assert es_kf.P.shape == (3, 3)
        assert math.isclose(es_kf.x_nom[0, 0], 1.0)
        assert es_kf.latest_innovation is None

    def test_predict_and_propagate(self):
        """Test high-rate nominal state propagation and error covariance propagation."""
        dt = 0.1
        f_nom = lambda x, u: np.array([[x[0, 0] + dt * x[1, 0]], [x[1, 0]], [x[2, 0]]])
        F_jac = np.array([
            [1.0, dt, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        Q = np.diag([0.01, 0.02, 0.01])

        x0 = np.array([0.0, 5.0, 10.0])
        P0 = np.eye(3) * 0.1
        es_kf = ErrorStateKalmanFilter(x0=x0, P0=P0)

        # Predict
        x_nom_prop, P_prop = es_kf.predict(f_nom_func=f_nom, F_jac=F_jac, Q=Q)
        assert math.isclose(x_nom_prop[0, 0], 0.5, abs_tol=1e-5)
        assert math.isclose(x_nom_prop[1, 0], 5.0, abs_tol=1e-5)
        assert math.isclose(x_nom_prop[2, 0], 10.0, abs_tol=1e-5)
        assert P_prop[0, 0] > P0[0, 0]

        # propagate alias
        x_nom_prop2, P_prop2 = es_kf.propagate(f_nom_func=f_nom, F_jac=F_jac, Q=Q)
        assert math.isclose(x_nom_prop2[0, 0], 1.0, abs_tol=1e-5)

    def test_update_with_default_additive_injection(self):
        """Test generic measurement update with default additive nominal state injection."""
        x0 = np.array([0.0, 5.0])
        P0 = np.eye(2) * 1.0
        es_kf = ErrorStateKalmanFilter(x0=x0, P0=P0)

        # Measurement model: direct position measurement
        h_func = lambda x: np.array([[x[0, 0]]])
        H_jac = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        x_upd, P_upd = es_kf.update(
            y=np.array([1.0]),
            h_func=h_func,
            H_jac=H_jac,
            R=R,
        )

        # Innovation should be y - h(x) = 1.0 - 0.0 = 1.0
        assert math.isclose(es_kf.latest_innovation[0, 0], 1.0, abs_tol=1e-5)
        # State should be injected with delta_x and updated towards 1.0
        assert 0.0 < x_upd[0, 0] < 1.0
        # Uncertainty reduced
        assert P_upd[0, 0] < P0[0, 0]
        # Joseph form guarantees positive definiteness and symmetry
        np.testing.assert_allclose(P_upd, P_upd.T, atol=1e-8)

    def test_update_with_custom_inject_func_and_angle_wrapping(self):
        """Test generic measurement update with custom manifold injection function and angle wrapping."""
        # 1D angle state
        x0 = np.array([3.10])
        P0 = np.array([[0.5]])
        es_kf = ErrorStateKalmanFilter(x0=x0, P0=P0)

        h_func = lambda x: x.copy()
        H_jac = np.eye(1)
        R = np.array([[0.1]])

        def custom_inject(x_nom, delta_x):
            # Custom manifold injection for circular angles
            x_f = float(np.asarray(x_nom).flatten()[0])
            dx_f = float(np.asarray(delta_x).flatten()[0])
            return np.array([(x_f + dx_f + np.pi) % (2 * np.pi) - np.pi])

        x_upd, P_upd = es_kf.update(
            y=np.array([-3.10]),
            h_func=h_func,
            H_jac=H_jac,
            R=R,
            inject_func=custom_inject,
            angle_indices=[0],
        )

        assert es_kf.latest_gain is not None
        assert P_upd[0, 0] < 0.5


class Test9DErrorStateEKF:
    """Test suite for the 9D ErrorStateEKF (IMU dead-reckoning + GNSS)."""

    @pytest.fixture
    def es_ekf_inst(self):
        init_pos = np.array([0.0, 0.0, 0.0])
        init_vel = np.array([0.0, 0.0, 0.0])
        init_quat = Quaternion([1.0, 0.0, 0.0, 0.0])
        init_cov = np.eye(9) * 0.1

        return ErrorStateEKF(
            init_pos=init_pos,
            init_vel=init_vel,
            init_quat=init_quat,
            init_cov=init_cov,
            accel_noise_std=0.1,
            gyro_noise_std=0.01,
            gravity=np.array([0.0, 0.0, -9.81]),
        )

    def test_initialization(self, es_ekf_inst):
        ekf = es_ekf_inst
        assert ekf.p.shape == (3,)
        assert ekf.v.shape == (3,)
        assert isinstance(ekf.q, Quaternion)
        assert ekf.P.shape == (9, 9)

    def test_static_prediction_with_gravity_compensation(self, es_ekf_inst):
        """Stationary IMU measuring specific force equal to -gravity: vehicle remains at rest."""
        ekf = es_ekf_inst
        # Specific force measuring +9.81 in z-axis to cancel -9.81 gravity
        f_b = np.array([0.0, 0.0, 9.81])
        omega_b = np.array([0.0, 0.0, 0.0])
        dt = 0.01

        p, v, q, P = ekf.predict(f_b, omega_b, dt)

        # Vehicle should remain at origin with zero velocity
        np.testing.assert_allclose(p, [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(v, [0.0, 0.0, 0.0], atol=1e-6)
        assert math.isclose(q.w, 1.0, abs_tol=1e-6)
        # Covariance grows
        assert P[0, 0] > 0.1

    def test_forward_acceleration_propagation(self, es_ekf_inst):
        """IMU accelerating in body x-direction."""
        ekf = es_ekf_inst
        f_b = np.array([2.0, 0.0, 9.81])
        omega_b = np.array([0.0, 0.0, 0.0])
        dt = 0.5

        p, v, q, P = ekf.predict(f_b, omega_b, dt)

        # a = 2 m/s^2 -> v = 2 * 0.5 = 1.0 m/s, p = 0.5 * 2 * (0.5)^2 = 0.25 m
        assert math.isclose(v[0], 1.0, abs_tol=1e-5)
        assert math.isclose(p[0], 0.25, abs_tol=1e-5)

    def test_gnss_position_update(self, es_ekf_inst):
        """Test GNSS position measurement correction reduces position covariance."""
        ekf = es_ekf_inst
        # Propagate 5 steps
        for _ in range(5):
            ekf.predict(np.array([1.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]), 0.1)

        pos_cov_prior = ekf.P[0, 0]

        # Update with GNSS measurement
        p_gnss = np.array([0.05, 0.0, 0.0])
        R_gnss = np.eye(3) * 0.01

        p_upd, v_upd, q_upd, P_upd, nis = ekf.update_gnss_position(p_gnss, R_gnss)

        assert isinstance(nis, float)
        assert nis >= 0.0
        # Position uncertainty must decrease
        assert P_upd[0, 0] < pos_cov_prior
        # Joseph form guarantees positive semi-definiteness & symmetry
        np.testing.assert_allclose(P_upd, P_upd.T, atol=1e-7)
