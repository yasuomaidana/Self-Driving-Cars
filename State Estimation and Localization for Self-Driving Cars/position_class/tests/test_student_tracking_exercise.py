"""Granular Unit Tests for Student and Instructor 2D Tracking Exercises.

Execute tests individually per step using pytest flags:
    # 1. Run all matrix tests:
    pytest tests/test_student_tracking_exercise.py -k test_step1

    # 2. Run specific matrix tests:
    pytest tests/test_student_tracking_exercise.py -k test_step1_1_matrix_f
    pytest tests/test_student_tracking_exercise.py -k test_step1_2_matrix_h
    pytest tests/test_student_tracking_exercise.py -k test_step1_3_matrix_q
    pytest tests/test_student_tracking_exercise.py -k test_step1_4_matrix_r
    pytest tests/test_student_tracking_exercise.py -k test_step1_5_matrix_p0

    # 3. Run prediction step tests:
    pytest tests/test_student_tracking_exercise.py -k test_step2_predict

    # 4. Run update step & dead reckoning tests:
    pytest tests/test_student_tracking_exercise.py -k test_step3_update
    pytest tests/test_student_tracking_exercise.py -k test_dead_reckoning

    # 5. Run verified instructor reference tests:
    pytest tests/test_student_tracking_exercise.py -k instructor
"""

import math
import numpy as np
import pytest

from position_class.student_tracking_exercise import StudentKalmanTracker2D
from position_class.instructor_tracking_exercise import InstructorKalmanTracker2D


# =============================================================================
# STUDENT UNIT TESTS (Tested against student implementation)
# =============================================================================

class TestStudentExerciseStep1Matrices:
    """Test Suite for Step 1: System Matrix Definitions (F, H, Q, R, P0)."""

    def test_step1_1_matrix_f(self):
        """Test Step 1.1: 4x4 State Transition Matrix F kinematics."""
        dt = 0.05
        tracker = StudentKalmanTracker2D(dt=dt)
        if tracker.F is None:
            pytest.fail("Step 1.1 Incomplete: self.F is None. Define the 4x4 F matrix in __init__.")

        assert tracker.F.shape == (4, 4), f"F must be shape (4, 4), got {tracker.F.shape}"
        expected_F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        np.testing.assert_allclose(tracker.F, expected_F, err_msg="Matrix F does not match constant-velocity kinematics.")

    def test_step1_2_matrix_h(self):
        """Test Step 1.2: 2x4 Observation Matrix H projection."""
        tracker = StudentKalmanTracker2D()
        if tracker.H is None:
            pytest.fail("Step 1.2 Incomplete: self.H is None. Define the 2x4 H matrix in __init__.")

        assert tracker.H.shape == (2, 4), f"H must be shape (2, 4), got {tracker.H.shape}"
        expected_H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)
        np.testing.assert_allclose(tracker.H, expected_H, err_msg="Matrix H does not extract 2D position correctly.")

    def test_step1_3_matrix_q(self):
        """Test Step 1.3: 4x4 Process Noise Covariance Matrix Q (CWNA model)."""
        dt = 0.1
        sigma_a = 2.0
        tracker = StudentKalmanTracker2D(dt=dt, process_noise_std=sigma_a)
        if tracker.Q is None:
            pytest.fail("Step 1.3 Incomplete: self.Q is None. Define the 4x4 Q matrix in __init__.")

        assert tracker.Q.shape == (4, 4), f"Q must be shape (4, 4), got {tracker.Q.shape}"
        
        q_pos = (dt**3) / 3.0 * (sigma_a**2)
        q_vel = dt * (sigma_a**2)
        q_pv = (dt**2) / 2.0 * (sigma_a**2)
        expected_Q = np.array([
            [q_pos, 0.0,   q_pv,  0.0  ],
            [0.0,   q_pos, 0.0,   q_pv ],
            [q_pv,  0.0,   q_vel, 0.0  ],
            [0.0,   q_pv,  0.0,   q_vel]
        ], dtype=np.float64)
        np.testing.assert_allclose(tracker.Q, expected_Q, err_msg="Matrix Q does not match CWNA formulation.")
        
        # Verify symmetry and positive semi-definiteness
        np.testing.assert_allclose(tracker.Q, tracker.Q.T, err_msg="Matrix Q must be symmetric.")
        eigenvalues = np.linalg.eigvalsh(tracker.Q)
        assert np.all(eigenvalues >= -1e-10), "Matrix Q must be positive semi-definite."

    def test_step1_4_matrix_r(self):
        """Test Step 1.4: 2x2 Measurement Noise Covariance Matrix R."""
        sigma_meas = 3.0
        tracker = StudentKalmanTracker2D(measurement_noise_std=sigma_meas)
        if tracker.R is None:
            pytest.fail("Step 1.4 Incomplete: self.R is None. Define the 2x2 R matrix in __init__.")

        assert tracker.R.shape == (2, 2), f"R must be shape (2, 2), got {tracker.R.shape}"
        expected_R = np.array([
            [sigma_meas**2, 0.0],
            [0.0, sigma_meas**2]
        ], dtype=np.float64)
        np.testing.assert_allclose(tracker.R, expected_R, err_msg="Matrix R diagonal does not match measurement variance.")

    def test_step1_5_matrix_p0(self):
        """Test Step 1.5: 4x4 Initial Uncertainty Matrix P0."""
        init_cov = 75.0
        tracker = StudentKalmanTracker2D(initial_covariance=init_cov)
        if tracker.P0 is None:
            pytest.fail("Step 1.5 Incomplete: self.P0 is None. Define the 4x4 P0 matrix in __init__.")

        assert tracker.P0.shape == (4, 4), f"P0 must be shape (4, 4), got {tracker.P0.shape}"
        expected_P0 = np.eye(4, dtype=np.float64) * init_cov
        np.testing.assert_allclose(tracker.P0, expected_P0, err_msg="Matrix P0 does not match initial uncertainty.")


class TestStudentExerciseStep2Predict:
    """Test Suite for Step 2: Kalman Prediction Step (Time Update)."""

    def test_step2_predict_state_propagation(self):
        """Test Step 2.1: State mean propagation x_check = F @ x."""
        dt = 0.1
        tracker = StudentKalmanTracker2D(dt=dt, process_noise_std=0.0)
        tracker.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        tracker.Q = np.zeros((4, 4), dtype=np.float64)
        
        tracker.initialize(initial_x=100.0, initial_y=200.0, initial_vx=10.0, initial_vy=-5.0)
        
        px, py, pvx, pvy = tracker.predict()

        assert math.isclose(px, 101.0, abs_tol=1e-5), f"Step 2.1 Failure: Expected px=101.0, got {px}. Did you implement self.x = self.F @ self.x?"
        assert math.isclose(py, 199.5, abs_tol=1e-5), f"Step 2.1 Failure: Expected py=199.5, got {py}"
        assert math.isclose(pvx, 10.0, abs_tol=1e-5), f"Step 2.1 Failure: Expected pvx=10.0, got {pvx}"
        assert math.isclose(pvy, -5.0, abs_tol=1e-5), f"Step 2.1 Failure: Expected pvy=-5.0, got {pvy}"

    def test_step2_predict_covariance_expansion(self):
        """Test Step 2.2: Covariance expansion P_check = F @ P @ F^T + Q."""
        dt = 0.1
        tracker = StudentKalmanTracker2D(dt=dt, process_noise_std=1.0)
        tracker.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        tracker.Q = np.eye(4, dtype=np.float64) * 2.0
        tracker.P0 = np.eye(4, dtype=np.float64) * 10.0
        
        tracker.initialize(10.0, 10.0, 0.0, 0.0)
        cov_prior = tracker.P[0, 0]
        
        tracker.predict()

        cov_predicted = tracker.P[0, 0]
        assert cov_predicted > cov_prior, "Step 2.2 Failure: Prediction step must increase uncertainty. Did you implement self.P = self.F @ self.P @ self.F.T + self.Q?"
        expected_P = tracker.F @ (np.eye(4) * 10.0) @ tracker.F.T + tracker.Q
        np.testing.assert_allclose(tracker.P, expected_P, err_msg="Step 2.2 Failure: P_check does not match F @ P @ F.T + Q")


class TestStudentExerciseStep3Update:
    """Test Suite for Step 3: Kalman Measurement Update (Correction Step)."""

    def test_step3_update_with_measurement(self):
        """Test Steps 3.1-3.5: Innovation, Kalman Gain, and Posterior state/covariance update."""
        dt = 0.1
        tracker = StudentKalmanTracker2D(dt=dt, process_noise_std=1.0, measurement_noise_std=2.0)
        tracker.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        tracker.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        tracker.Q = np.eye(4, dtype=np.float64) * 0.5
        tracker.R = np.eye(2, dtype=np.float64) * 4.0
        tracker.P0 = np.eye(4, dtype=np.float64) * 50.0

        tracker.initialize(100.0, 100.0, 0.0, 0.0)
        tracker.predict()

        prior_cov = tracker.P[0, 0]
        
        ex, ey, evx, evy = tracker.update((105.0, 102.0))

        assert 100.0 < ex <= 105.0, f"Step 3.4 Failure: Corrected x ({ex}) must be pulled towards measurement (105). Check self.x = self.x + K @ nu."
        assert 100.0 < ey <= 102.0, f"Step 3.4 Failure: Corrected y ({ey}) must be pulled towards measurement (102)."
        assert tracker.P[0, 0] < prior_cov, "Step 3.5 Failure: Measurement update must reduce position uncertainty covariance."

    def test_step3_dead_reckoning_on_none_measurement(self):
        """Test that when measurement is None (occlusion), the predicted state is preserved."""
        dt = 0.1
        tracker = StudentKalmanTracker2D(dt=dt)
        tracker.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        tracker.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        tracker.Q = np.zeros((4, 4), dtype=np.float64)
        tracker.R = np.eye(2, dtype=np.float64)

        tracker.initialize(50.0, 50.0, 10.0, 20.0)
        pred_x, pred_y, pred_vx, pred_vy = tracker.predict()

        upd_x, upd_y, upd_vx, upd_vy = tracker.update(None)

        assert math.isclose(upd_x, pred_x, abs_tol=1e-5), "Dead Reckoning Failure: When measurement is None, state x must equal predicted x."
        assert math.isclose(upd_y, pred_y, abs_tol=1e-5), "Dead Reckoning Failure: When measurement is None, state y must equal predicted y."


# =============================================================================
# INSTRUCTOR REFERENCE VERIFICATION (Guaranteed to pass)
# =============================================================================

class TestInstructorReferenceSuite:
    """Full ground-truth verification of the instructor reference solution."""

    def test_instructor_matrices(self):
        """Verify instructor matrix construction."""
        dt = 0.1
        sigma_a = 1.5
        sigma_meas = 2.0
        init_cov = 50.0
        tracker = InstructorKalmanTracker2D(dt=dt, process_noise_std=sigma_a, measurement_noise_std=sigma_meas, initial_covariance=init_cov)
        
        assert tracker.F.shape == (4, 4)
        assert tracker.H.shape == (2, 4)
        assert tracker.Q.shape == (4, 4)
        assert tracker.R.shape == (2, 2)
        assert tracker.P0.shape == (4, 4)

    def test_instructor_predict_and_update_cycle(self):
        """Verify instructor complete cycle with Joseph covariance form."""
        tracker = InstructorKalmanTracker2D(dt=0.1, process_noise_std=1.0, measurement_noise_std=2.0)
        tracker.initialize(0.0, 0.0, 10.0, 10.0)

        # 1. Predict
        px, py, pvx, pvy = tracker.predict()
        assert math.isclose(px, 1.0, abs_tol=1e-5)
        assert math.isclose(py, 1.0, abs_tol=1e-5)

        # 2. Update with measurement
        ux, uy, uvx, uvy = tracker.update((1.2, 0.9))
        assert 1.0 <= ux <= 1.2
        assert 0.9 <= uy <= 1.0
        
        # 3. Joseph form covariance symmetry and positive definiteness
        np.testing.assert_allclose(tracker.P, tracker.P.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(tracker.P) > 0.0)
