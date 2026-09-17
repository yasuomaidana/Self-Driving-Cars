"""Integration tests for Program 1 (JSON tracking) and Program 2 (Interactive AI tracking)."""

from pathlib import Path
import pytest

from position_class.tracking_exercise_json import run_json_tracking_exercise
from position_class.interactive_tracking_exercise import (
    InteractiveKalmanAITracker,
    run_interactive_tracking_exercise,
)
from position_class.tracking_io import generate_sample_tracking_dataset


class TestTrackingPipelines:
    """End-to-end integration tests for the exercise pipelines."""

    def test_program_1_json_tracking_pipeline(self, tmp_path):
        """Test full execution of Program 1 with generated sample dataset."""
        dataset_dir = tmp_path / "dataset"
        json_path, _ = generate_sample_tracking_dataset(dataset_dir, num_frames=15)

        out_dir = tmp_path / "output_json"
        error_service = run_json_tracking_exercise(
            json_path=json_path,
            output_dir=out_dir,
            save_video=False,
            display=False
        )

        assert len(error_service.get_history()) == 15
        assert (out_dir / "tracking_error_log.csv").exists()
        assert (out_dir / "annotated_frames" / "vis_0000.png").exists()

        metrics = error_service.compute_metrics()
        assert metrics["total_samples"] == 15
        assert metrics["mean_error"] > 0.0

    def test_program_2_interactive_ai_tracking_pipeline(self, tmp_path):
        """Test execution of Program 2 with simulated user click in headless mode."""
        out_dir = tmp_path / "output_interactive"
        tracker = run_interactive_tracking_exercise(
            source=None,
            output_dir=out_dir,
            auto_click_frame=2,
            interactive_gui=False,
            save_output=True
        )

        assert tracker.state in ("TRACKING", "OCCLUDED / LOST (PREDICTING)")
        history = tracker.error_service.get_history()
        assert len(history) > 0
        assert (out_dir / "interactive_error_log.csv").exists()
        assert (out_dir / "annotated_frames" / "frame_0000.png").exists()

    def test_program_2_live_camera_mode(self, tmp_path):
        """Test Program 2 handling with camera_id parameter."""
        out_dir = tmp_path / "output_live_cam"
        # In automated test environments without a physical camera, falls back gracefully to synthetic
        tracker = run_interactive_tracking_exercise(
            camera_id=999,  # Non-existent camera triggers graceful fallback
            output_dir=out_dir,
            auto_click_frame=2,
            interactive_gui=False,
            save_output=False
        )
        assert tracker is not None
        assert tracker.state in ("IDLE", "TRACKING", "OCCLUDED / LOST (PREDICTING)")

    def test_hover_highlighting_and_label_retention(self, tmp_path):
        """Verify that hovering highlights candidate boxes and labels are preserved upon click."""
        import numpy as np
        tracker = InteractiveKalmanAITracker()
        
        # Synthetic test frame with two distinct bright shapes
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        import cv2
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(frame, (400, 200), (520, 320), (255, 255, 255), -1)

        # 1. Hover outside all objects
        vis_idle, _ = tracker.process_frame(frame, frame_idx=0, hover_pos=(10, 10))
        assert tracker.state == "IDLE"
        assert vis_idle.shape == frame.shape

        # 2. Hover inside first object (150, 150)
        vis_hovered, _ = tracker.process_frame(frame, frame_idx=0, hover_pos=(150, 150))
        assert vis_hovered.shape == frame.shape

        # 3. Click first object
        tracker.handle_mouse_click(150, 150, frame)
        assert tracker.state == "TRACKING"
        assert tracker.selected_label != ""

        # 4. Next frame in tracking mode
        vis_track, error_sig = tracker.process_frame(frame, frame_idx=1)
        assert tracker.state == "TRACKING"
        assert error_sig is not None
        assert tracker.selected_label in tracker.error_service.service_name or len(tracker.selected_label) > 0

    def test_target_class_filtering_and_slot_selection(self, tmp_path):
        """Test target class filtering and slot selection by index."""
        import numpy as np
        import cv2

        # Tracker with target_class filter
        tracker = InteractiveKalmanAITracker(target_class="Car")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (250, 180), (255, 255, 255), -1)  # Aspect ratio > 1.2 -> detected as Car

        vis, _ = tracker.process_frame(frame, frame_idx=0)
        assert tracker.state == "IDLE"
        assert len(tracker.current_candidates) >= 1

        # Select candidate via slot #1
        selected = tracker.select_candidate_by_index(1, frame)
        assert selected is True
        assert tracker.state == "TRACKING"

    def test_auto_lock_mode(self, tmp_path):
        """Test auto-lock freeze selection and confirmation."""
        import numpy as np
        import cv2

        tracker = InteractiveKalmanAITracker(auto_lock=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)

        # Frame 0: in freeze selection
        vis, error_sig = tracker.process_frame(frame, frame_idx=0, is_paused=True)
        assert tracker.state == "IDLE"
        assert len(tracker.current_candidates) >= 1

        # Confirm selection (e.g. Enter / key 1 / click)
        tracker.select_candidate_by_index(1, frame)
        assert tracker.state == "TRACKING"

        # Frame 1: tracking
        vis_track, error_sig_track = tracker.process_frame(frame, frame_idx=1)
        assert tracker.state == "TRACKING"
        assert error_sig_track is not None

    def test_custom_roi_fast_feature_drawing(self, tmp_path):
        """Test user drag-to-draw custom ROI with FAST feature extraction."""
        import numpy as np
        import cv2

        tracker = InteractiveKalmanAITracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw some rich textured content inside ROI
        cv2.circle(frame, (250, 200), 30, (255, 255, 255), -1)
        cv2.rectangle(frame, (230, 180), (270, 220), (0, 0, 0), -1)

        # 1. Test live rubber-band drawing overlay
        vis_drawing, _ = tracker.process_frame(
            frame,
            frame_idx=0,
            is_paused=True,
            drawing_roi=(200, 150, 100, 100)
        )
        assert vis_drawing.shape == frame.shape

        # 2. Test handle_custom_roi
        custom_bbox = (200, 150, 100, 100)
        tracker.handle_custom_roi(custom_bbox, frame, label="FAST Custom Box")
        assert tracker.state == "TRACKING"
        assert tracker.selected_label == "FAST Custom Box"
        assert tracker.kalman.initialized is True
        assert tracker.kalman.x[0, 0] == 250.0  # cx
        assert tracker.kalman.x[1, 0] == 200.0  # cy

        # 3. Next frame tracking
        vis_track, error_sig = tracker.process_frame(frame, frame_idx=1)
        assert tracker.state in ("TRACKING", "OCCLUDED / LOST (PREDICTING)")
        assert error_sig is not None

    def test_fast_feature_tracker_3_points_and_dynamic_refresh(self):
        """Test FastFeatureTracker extracts 3 FAST points and refreshes them dynamically."""
        import cv2
        import numpy as np
        from position_class.ai_detectors import FastFeatureTracker

        tracker = FastFeatureTracker(fast_threshold=10)
        # Create textured synthetic image
        frame1 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (100, 100), (160, 160), (255, 255, 255), -1)
        cv2.circle(frame1, (120, 120), 5, (0, 0, 0), -1)
        cv2.circle(frame1, (140, 120), 5, (0, 0, 0), -1)
        cv2.circle(frame1, (130, 140), 5, (0, 0, 0), -1)

        init_res = tracker.init_target(frame1, (130, 130), bbox=(100, 100, 60, 60), label="Target")
        assert init_res is not None
        assert tracker.fast_points is not None
        assert len(tracker.fast_points) == 3
        for pt in tracker.fast_points:
            assert 100 <= pt[0] <= 160
            assert 100 <= pt[1] <= 160

        # Frame 2 with small translation
        frame2 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (105, 105), (165, 165), (255, 255, 255), -1)
        cv2.circle(frame2, (125, 125), 5, (0, 0, 0), -1)
        cv2.circle(frame2, (145, 125), 5, (0, 0, 0), -1)
        cv2.circle(frame2, (135, 145), 5, (0, 0, 0), -1)

        track_res = tracker.track(frame2, expected_pos=(135, 135), auto_refresh=True)
        assert track_res is not None
        assert track_res.confidence >= 0.70
        assert len(track_res.fast_points) == 3

    def test_hybrid_tracker_tracking_and_reacquisition(self):
        """Test HybridTracker performs FAST 3-point tracking and YOLO recovery on target jump."""
        import cv2
        import numpy as np
        from position_class.ai_detectors import HybridTracker, YOLODetector

        yolo = YOLODetector()
        hybrid = HybridTracker(yolo_detector=yolo, fast_threshold=10)

        # Initial frame
        frame1 = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (100, 100), (180, 180), (255, 255, 255), -1)
        cv2.circle(frame1, (130, 130), 8, (0, 0, 0), -1)
        cv2.circle(frame1, (150, 130), 8, (0, 0, 0), -1)
        cv2.circle(frame1, (140, 160), 8, (0, 0, 0), -1)

        init_res = hybrid.init_target(frame1, (140, 140), bbox=(100, 100, 80, 80), label="Target")
        assert hybrid.is_initialized is True
        assert len(init_res.fast_points) == 3

        # Tracking frame (small shift) -> HYBRID_FAST
        frame2 = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (105, 105), (185, 185), (255, 255, 255), -1)
        cv2.circle(frame2, (135, 135), 8, (0, 0, 0), -1)
        cv2.circle(frame2, (155, 135), 8, (0, 0, 0), -1)
        cv2.circle(frame2, (145, 165), 8, (0, 0, 0), -1)

        det, mode = hybrid.track(frame2, expected_pos=(145, 145), kalman_uncertainty_std=15.0)
        assert det is not None
        assert mode == "HYBRID_FAST"
        assert len(det.fast_points) == 3

        # Occluded / empty frame -> DEAD_RECKONING
        empty_frame = np.zeros((400, 400, 3), dtype=np.uint8)
        det_empty, mode_empty = hybrid.track(empty_frame, expected_pos=(145, 145), kalman_uncertainty_std=15.0)
        assert det_empty is None
        assert mode_empty == "DEAD_RECKONING"

    def test_interactive_hybrid_mode_pipeline(self, tmp_path):
        """Test full InteractiveKalmanAITracker pipeline running in --hybrid mode."""
        dataset_dir = tmp_path / "dataset"
        json_path, _ = generate_sample_tracking_dataset(dataset_dir, num_frames=12)

        out_dir = tmp_path / "output_hybrid"
        tracker = run_interactive_tracking_exercise(
            source=json_path,
            output_dir=out_dir,
            auto_click_frame=2,
            auto_click_coords=(120, 120),
            hybrid_mode=True,
            interactive_gui=False,
            save_output=False
        )

        assert tracker.hybrid_mode is True
        history = tracker.error_service.get_history()
        assert len(history) > 0
        metrics = tracker.error_service.compute_metrics()
        assert "mean_error" in metrics
        assert metrics["mean_error"] >= 0.0

    def test_rubber_band_drawing_in_all_states(self):
        """Test rubber band drawing ROI works in IDLE, TRACKING, and PAUSED states."""
        import numpy as np
        tracker = InteractiveKalmanAITracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. In IDLE state
        vis_idle, _ = tracker.process_frame(frame, frame_idx=0, drawing_roi=(50, 50, 100, 100))
        assert vis_idle is not None

        # 2. Lock a target -> TRACKING state
        tracker.handle_mouse_click(100, 100, frame)
        assert tracker.state == "TRACKING"

        # 3. In TRACKING state while dragging
        vis_track, _ = tracker.process_frame(frame, frame_idx=1, is_paused=True, drawing_roi=(150, 150, 80, 80))
        assert vis_track is not None

    def test_eye_detection_filter(self):
        """Test YOLODetector handles eye/face target filter."""
        import cv2
        import numpy as np
        from position_class.ai_detectors import YOLODetector

        detector = YOLODetector()
        frame = np.ones((400, 400, 3), dtype=np.uint8) * 180
        # Draw high contrast circular eye-like pupil
        cv2.circle(frame, (200, 200), 18, (20, 20, 20), -1)

        dets = detector.detect_all(frame, target_class="eye")
        assert len(dets) >= 1
        assert any("eye" in d.label.lower() or "pupil" in d.label.lower() for d in dets)

    def test_full_frame_reacquisition_after_large_jump(self):
        """Test tracker recovers object anywhere in image even when it jumps far outside local gate."""
        import cv2
        import numpy as np
        from position_class.ai_detectors import FastFeatureTracker

        tracker = FastFeatureTracker(fast_threshold=10)
        frame1 = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (50, 50), (110, 110), (255, 255, 255), -1)
        cv2.circle(frame1, (70, 70), 5, (0, 0, 0), -1)
        cv2.circle(frame1, (90, 70), 5, (0, 0, 0), -1)
        cv2.circle(frame1, (80, 90), 5, (0, 0, 0), -1)

        tracker.init_target(frame1, (80, 80), bbox=(50, 50, 60, 60), label="Target")

        # Frame 2: Target jumps 350 pixels away to (400, 400)
        frame2 = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (400, 400), (460, 460), (255, 255, 255), -1)
        cv2.circle(frame2, (420, 420), 5, (0, 0, 0), -1)
        cv2.circle(frame2, (440, 420), 5, (0, 0, 0), -1)
        cv2.circle(frame2, (430, 440), 5, (0, 0, 0), -1)

        # Expected pos was still (80, 80)
        res = tracker.track(frame2, expected_pos=(80, 80), enable_full_frame_search=True)
        assert res is not None
        assert abs(res.center_of_mass[0] - 430.0) < 5.0
        assert abs(res.center_of_mass[1] - 430.0) < 5.0
        assert res.bbox[0] == 400
        assert res.bbox[1] == 400

    def test_dataset_selection_and_idle_feature_region_drawing(self):
        """Test YOLO dataset selection and rubber-band drawing in IDLE mode."""
        import numpy as np
        from position_class.ai_detectors import YOLODetector

        # 1. Test dataset selection in detector
        kitti_yolo = YOLODetector(model_name="yolo26n", dataset="kitti")
        assert "yolo26n.pt" in kitti_yolo.model_name
        assert "KITTI" in kitti_yolo.get_model_info()

        # 2. Test rubber-band feature region appears in IDLE frame
        tracker = InteractiveKalmanAITracker(yolo_model_name="yolo26n", dataset="kitti")
        frame = np.zeros((375, 1242, 3), dtype=np.uint8)
        vis, _ = tracker.process_frame(frame, frame_idx=0, drawing_roi=(100, 100, 200, 150))
        assert vis is not None
        assert vis.shape == (375, 1242, 3)








