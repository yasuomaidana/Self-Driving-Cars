"""Unit Tests for 2D Kalman Filter Tracking Module."""

import math
import numpy as np
import pytest

from position_class.tracking_kalman import KalmanTracker2D, KalmanTrackState, build_cv_matrices_2d


class TestKalmanTracker2D:
    """Test suite for KalmanTracker2D."""

    def test_initialization(self):
        """Test tracker initialization state vector and covariance matrix."""
        dt = 0.1
        tracker = KalmanTracker2D(dt=dt, process_noise_std=0.5, measurement_noise_std=1.5)
        
        assert not tracker.initialized
        tracker.initialize(100.0, 200.0, 5.0, -2.0)
        assert tracker.initialized
        
        pos = tracker.position
        vel = tracker.velocity
        assert pos == (100.0, 200.0)
        assert vel == (5.0, -2.0)
        assert tracker.P.shape == (4, 4)
        assert tracker.F.shape == (4, 4)
        assert tracker.H.shape == (2, 4)

    def test_constant_velocity_prediction(self):
        """Test linear propagation under constant velocity without measurements."""
        dt = 1.0
        tracker = KalmanTracker2D(dt=dt, process_noise_std=0.0)
        tracker.initialize(0.0, 10.0, 2.0, 3.0)

        # Predict 3 steps ahead
        p1 = tracker.predict()
        assert math.isclose(p1[0], 2.0, abs_tol=1e-5)
        assert math.isclose(p1[1], 13.0, abs_tol=1e-5)

        p2 = tracker.predict()
        assert math.isclose(p2[0], 4.0, abs_tol=1e-5)
        assert math.isclose(p2[1], 16.0, abs_tol=1e-5)

        p3 = tracker.predict()
        assert math.isclose(p3[0], 6.0, abs_tol=1e-5)
        assert math.isclose(p3[1], 19.0, abs_tol=1e-5)

    def test_measurement_update_reduces_covariance(self):
        """Test that updating with measurements reduces position uncertainty."""
        tracker = KalmanTracker2D(dt=0.1, initial_covariance=100.0, measurement_noise_std=1.0)
        tracker.initialize(50.0, 50.0)

        cov_initial = tracker.P[0, 0]
        tracker.predict()
        tracker.update((52.0, 51.0))
        cov_after_update = tracker.P[0, 0]

        assert cov_after_update < cov_initial

    def test_step_with_missing_measurement_dead_reckoning(self):
        """Test that when measurement is None (occlusion), dead reckoning prediction is used."""
        tracker = KalmanTracker2D(dt=0.5, process_noise_std=0.1, measurement_noise_std=1.0)
        tracker.initialize(10.0, 20.0, 4.0, 0.0)

        # Step with measurement
        s1 = tracker.step((12.0, 20.0), image_shape=(480, 640))
        assert s1.is_detected
        assert s1.meas_x == 12.0

        # Step without measurement (occluded)
        s2 = tracker.step(None, image_shape=(480, 640))
        assert not s2.is_detected
        assert s2.meas_x is None
        # Should have predicted along velocity
        assert s2.est_x > s1.est_x

    def test_center_error_calculation(self):
        """Test error calculation relative to image center (W/2, H/2)."""
        tracker = KalmanTracker2D()
        tracker.initialize(320.0, 240.0)

        # Exact center of a 640x480 frame
        ex, ey, dist = tracker.compute_center_error(image_width=640, image_height=480)
        assert math.isclose(ex, 0.0, abs_tol=1e-5)
        assert math.isclose(ey, 0.0, abs_tol=1e-5)
        assert math.isclose(dist, 0.0, abs_tol=1e-5)

        # Off-center: (350, 280) -> dx = 30, dy = 40 -> dist = 50
        tracker.initialize(350.0, 280.0)
        ex, ey, dist = tracker.compute_center_error(image_width=640, image_height=480)
        assert math.isclose(ex, 30.0, abs_tol=1e-5)
        assert math.isclose(ey, 40.0, abs_tol=1e-5)
        assert math.isclose(dist, 50.0, abs_tol=1e-5)

    def test_uninitialized_error_handling(self):
        """Test that calling predict without initialization raises RuntimeError."""
        tracker = KalmanTracker2D()
        with pytest.raises(RuntimeError):
            tracker.predict()
        with pytest.raises(RuntimeError):
            tracker.update((10.0, 10.0))

    def test_explicit_matrices_injection(self):
        """Test initializing KalmanTracker2D with custom explicit F, H, Q, R, P0 matrices."""
        dt = 0.05
        F, H, Q, R, P0 = build_cv_matrices_2d(dt=dt, process_noise_std=0.8, measurement_noise_std=1.2, initial_covariance=30.0)
        tracker = KalmanTracker2D(F=F, H=H, Q=Q, R=R, P0=P0)
        
        assert tracker.n == 4
        assert tracker.m == 2
        np.testing.assert_allclose(tracker.F, F)
        np.testing.assert_allclose(tracker.H, H)
        np.testing.assert_allclose(tracker.Q, Q)
        np.testing.assert_allclose(tracker.R, R)
        np.testing.assert_allclose(tracker.P, P0)

        tracker.initialize(50.0, 60.0, 2.0, -1.0)
        px, py, pvx, pvy = tracker.predict()
        assert math.isclose(px, 50.0 + 2.0 * dt, abs_tol=1e-5)
        assert math.isclose(py, 60.0 - 1.0 * dt, abs_tol=1e-5)

    def test_build_cv_matrices_structure(self):
        """Verify the mathematical structure of the 2D Constant Velocity matrices."""
        dt = 0.1
        sigma_a = 0.5
        sigma_meas = 2.0
        F, H, Q, R, P0 = build_cv_matrices_2d(dt=dt, process_noise_std=sigma_a, measurement_noise_std=sigma_meas)

        # F matrix: constant velocity kinematics
        assert F.shape == (4, 4)
        assert F[0, 2] == dt
        assert F[1, 3] == dt
        assert F[0, 0] == 1.0 and F[1, 1] == 1.0

        # H matrix: direct position observation
        assert H.shape == (2, 4)
        assert H[0, 0] == 1.0 and H[1, 1] == 1.0
        assert H[0, 2] == 0.0 and H[1, 3] == 0.0

        # R matrix: sensor noise variance
        assert R.shape == (2, 2)
        assert math.isclose(R[0, 0], sigma_meas**2)
        assert math.isclose(R[1, 1], sigma_meas**2)

        # Q matrix: symmetric positive semi-definite
        assert Q.shape == (4, 4)
        assert np.allclose(Q, Q.T)
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues >= -1e-10)

    def test_student_kalman_tracker(self):
        """Test that StudentKalmanTracker2D properly implements the BaseKalmanTracker2D interface."""
        from position_class import StudentKalmanTracker2D
        tracker = StudentKalmanTracker2D(dt=0.1, process_noise_std=1.0, measurement_noise_std=2.0)
        tracker.initialize(10.0, 20.0, 1.0, 2.0)
        assert tracker.is_initialized
        
        px, py, pvx, pvy = tracker.predict()
        assert math.isclose(px, 10.1, abs_tol=1e-5)
        assert math.isclose(py, 20.2, abs_tol=1e-5)

        ex, ey, evx, evy = tracker.update((10.5, 20.3))
        assert tracker.position == (ex, ey)

    def test_instructor_kalman_tracker(self):
        """Test that InstructorKalmanTracker2D executes Joseph stabilized update."""
        from position_class import InstructorKalmanTracker2D
        tracker = InstructorKalmanTracker2D(dt=0.1, process_noise_std=1.0, measurement_noise_std=2.0)
        tracker.initialize(10.0, 20.0, 1.0, 2.0)
        
        tracker.predict()
        ex, ey, evx, evy = tracker.update((10.5, 20.3))
        assert tracker.P.shape == (4, 4)
        # Covariance must remain symmetric positive definite
        assert np.allclose(tracker.P, tracker.P.T)
        assert np.all(np.linalg.eigvalsh(tracker.P) > 0)

