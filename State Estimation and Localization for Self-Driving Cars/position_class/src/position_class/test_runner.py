"""Convenience CLI Test Runner for Kalman Filter Student Exercises.

Provides entrypoints for running specific unit tests via `uv run test_f`, `uv run test_predict`, etc.
"""

import sys
import pytest


def run_test_f():
    """Run Step 1.1 Matrix F unit test."""
    sys.exit(pytest.main(["tests/test_student_tracking_exercise.py", "-k", "test_step1_1_matrix_f", "-v"]))


def run_test_h():
    """Run Step 1.2 Matrix H unit test."""
    sys.exit(pytest.main(["tests/test_student_tracking_exercise.py", "-k", "test_step1_2_matrix_h", "-v"]))


def run_test_q():
    """Run Step 1.3 Matrix Q unit test."""
    sys.exit(pytest.main(["tests/test_student_tracking_exercise.py", "-k", "test_step1_3_matrix_q", "-v"]))


def run_test_r():
    """Run Step 1.4 Matrix R unit test."""
    sys.exit(pytest.main(["tests/test_student_tracking_exercise.py", "-k", "test_step1_4_matrix_r", "-v"]))


def run_test_p0():
    """Run Step 1.5 Matrix P0 unit test."""
    sys.exit(pytest.main(["tests/test_student_tracking_exercise.py", "-k", "test_step1_5_matrix_p0", "-v"]))


def run_test_matrices():
    """Run all Step 1 Matrix definition unit tests."""
    sys.exit(pytest.main(["tests/test_student_tracking_exercise.py", "-k", "test_step1", "-v"]))


def run_test_predict():
    """Run Step 2 Kalman Prediction unit tests."""
    sys.exit(pytest.main(["tests/test_student_tracking_exercise.py", "-k", "test_step2_predict", "-v"]))


def run_test_update():
    """Run Step 3 Kalman Measurement Update and Dead Reckoning unit tests."""
    sys.exit(pytest.main(["tests/test_student_tracking_exercise.py", "-k", "test_step3_update", "-v"]))


def run_test_dead_reckoning():
    """Run Step 3 Dead Reckoning (occlusion) unit test."""
    sys.exit(pytest.main(["tests/test_student_tracking_exercise.py", "-k", "test_dead_reckoning", "-v"]))


def run_test_student():
    """Run all Student exercise unit tests."""
    sys.exit(pytest.main(["tests/test_student_tracking_exercise.py", "-k", "TestStudent", "-v"]))


def run_test_instructor():
    """Run Instructor reference solution unit tests."""
    sys.exit(pytest.main(["tests/test_student_tracking_exercise.py", "-k", "TestInstructor", "-v"]))


def run_test_all():
    """Run all exercise unit tests."""
    sys.exit(pytest.main(["tests/test_student_tracking_exercise.py", "-v"]))
