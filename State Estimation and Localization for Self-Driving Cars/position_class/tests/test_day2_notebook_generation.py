"""Unit Tests for Day 2 Notebook Generation and Academic Integrity."""

import json
from pathlib import Path
import pytest

from generate_day2_notebooks import generate_all_day2_notebooks, BASE_DIR


@pytest.fixture(scope="module", autouse=True)
def generated_notebooks():
    """Generate all Day 2 notebooks before running assertions."""
    generate_all_day2_notebooks()


class TestDay2Notebooks:
    """Test suite for Day 2 notebook generation and generic definitions."""

    @pytest.mark.parametrize("filename", [
        "Day_02_Extended_Kalman_Filter_instructor.ipynb",
        "Day_02_Extended_Kalman_Filter_student.ipynb",
        "Day_02_Unscented_Kalman_Filter_instructor.ipynb",
        "Day_02_Unscented_Kalman_Filter_student.ipynb",
        "Day_02_Error_State_Kalman_Filter_instructor.ipynb",
        "Day_02_Error_State_Kalman_Filter_student.ipynb",
    ])
    def test_notebook_exists_and_valid_json(self, filename):
        nb_path = Path(BASE_DIR) / filename
        assert nb_path.exists(), f"Notebook file {filename} was not generated."
        
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "cells" in data
        assert len(data["cells"]) > 5
        assert data.get("nbformat") == 4

    def test_ekf_generic_class_definition_present(self):
        nb_path = Path(BASE_DIR) / "Day_02_Extended_Kalman_Filter_instructor.ipynb"
        with open(nb_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "class ExtendedKalmanFilter:" in content
        assert "def predict(" in content
        assert "def update(" in content
        assert "wrap_angle" in content

    def test_ukf_generic_class_definition_present(self):
        nb_path = Path(BASE_DIR) / "Day_02_Unscented_Kalman_Filter_instructor.ipynb"
        with open(nb_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "class UnscentedKalmanFilter:" in content
        assert "generate_sigma_points" in content
        assert "def predict(" in content
        assert "def update(" in content

    def test_es_ekf_generic_class_definition_present(self):
        nb_path = Path(BASE_DIR) / "Day_02_Error_State_Kalman_Filter_instructor.ipynb"
        with open(nb_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "class ErrorStateKalmanFilter:" in content
        assert "def predict(" in content
        assert "def update(" in content
        assert "propagate = predict" in content

    @pytest.mark.parametrize("student_nb,expected_tests", [
        ("Day_02_Extended_Kalman_Filter_student.ipynb", [
            "Generic ExtendedKalmanFilter Sanity Check",
            "2D Polar Radar Kinematics & Jacobians Sanity Check",
            "assert np.isclose(x_pred[0, 0], 10.2)",
            "assert np.all(np.linalg.eigvals(P_upd) > 0)"
        ]),
        ("Day_02_Unscented_Kalman_Filter_student.ipynb", [
            "Generic UnscentedKalmanFilter Sanity Check",
            "assert np.isclose(np.sum(ukf_t.alpha_weights), 1.0)",
            "assert ukf_t.num_sigmas == 5",
            "assert np.isclose(ukf_t.alpha_weights[0], 1.0 / 3.0)",
            "assert np.allclose(recon_mean, x0_t"
        ]),
        ("Day_02_Error_State_Kalman_Filter_student.ipynb", [
            "Generic ErrorStateKalmanFilter Sanity Check",
            "Vehicle Dead-Reckoning, Error Jacobians & Manifold Injection",
            "assert np.isclose(x_p[0, 0], 0.5)",
            "assert np.allclose(H_pos, np.array([[1.0, 0.0, 0.0, 0.0]"
        ]),
    ])
    def test_student_notebooks_contain_unit_test_assertions(self, student_nb, expected_tests):
        """Verify student notebooks have immediate assertion cells to self-verify implementation."""
        nb_path = Path(BASE_DIR) / student_nb
        with open(nb_path, "r", encoding="utf-8") as f:
            content = f.read()

        for exp in expected_tests:
            assert exp in content, f"Expected '{exp}' in {student_nb}"

