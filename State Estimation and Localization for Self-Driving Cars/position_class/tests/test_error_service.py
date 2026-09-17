"""Unit Tests for Error Service and Publisher Module."""

import math
from pathlib import Path
import pytest

from position_class.error_service import ErrorPublisherService, TrackingErrorSignal


class TestErrorService:
    """Test suite for error service and subscribers."""

    def test_error_signal_computation(self):
        """Test error calculation relative to image center."""
        # Image shape: 480x640 -> Center is (320, 240)
        # Target at (350, 280) -> ex = +30, ey = +40 -> dist = 50.0
        sig = TrackingErrorSignal.compute(
            frame_index=1,
            target_pos=(350.0, 280.0),
            image_shape=(480, 640),
            is_tracked=True,
            predicted_pos=(352.0, 281.0)
        )

        assert sig.frame_index == 1
        assert sig.image_center_x == 320.0
        assert sig.image_center_y == 240.0
        assert math.isclose(sig.error_x, 30.0, abs_tol=1e-5)
        assert math.isclose(sig.error_y, 40.0, abs_tol=1e-5)
        assert math.isclose(sig.error_distance, 50.0, abs_tol=1e-5)
        assert sig.predicted_x == 352.0
        assert sig.predicted_y == 281.0

    def test_subscriber_callback_and_queue(self, tmp_path):
        """Test subscriber notification and CSV logging."""
        service = ErrorPublisherService()
        csv_file = tmp_path / "error_log.csv"
        service.enable_csv_logging(csv_file)

        received_signals = []
        service.subscribe(lambda s: received_signals.append(s))
        queue = service.create_queue_subscriber()

        # Publish 3 signals
        for i in range(3):
            sig = TrackingErrorSignal.compute(
                frame_index=i,
                target_pos=(100.0 + i * 10, 100.0),
                image_shape=(200, 200)
            )
            service.publish(sig)

        assert len(received_signals) == 3
        assert queue.qsize() == 3
        assert len(service.get_history()) == 3

        service.close()
        assert csv_file.exists()
        lines = csv_file.read_text().strip().split("\n")
        assert len(lines) == 4  # Header + 3 records

    def test_metrics_computation(self):
        """Test MAE, RMSE, and Max Error calculations."""
        service = ErrorPublisherService()
        sig1 = TrackingErrorSignal.compute(0, (110.0, 100.0), (200, 200)) # dist = 10
        sig2 = TrackingErrorSignal.compute(1, (120.0, 100.0), (200, 200)) # dist = 20
        service.publish(sig1)
        service.publish(sig2)

        metrics = service.compute_metrics()
        assert math.isclose(metrics["mean_error"], 15.0, abs_tol=1e-5)
        assert math.isclose(metrics["max_error"], 20.0, abs_tol=1e-5)
        assert metrics["total_samples"] == 2
