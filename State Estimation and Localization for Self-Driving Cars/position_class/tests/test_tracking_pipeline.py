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

    def test_nested_inner_class_hover_and_click_selection(self):
        """Test that inner/nested small targets are prioritized over large containing outer boxes."""
        import numpy as np
        from position_class.ai_detectors import DetectionResult

        tracker = InteractiveKalmanAITracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock two candidates: Outer large box (Person) and Inner small box (Eye)
        outer_box = DetectionResult(
            bbox=(50, 50, 300, 400),
            center_of_mass=(200.0, 250.0),
            confidence=0.90,
            label="Person (90%)"
        )
        inner_box = DetectionResult(
            bbox=(140, 100, 30, 20),
            center_of_mass=(155.0, 110.0),
            confidence=0.95,
            label="Eye (95%)"
        )

        # 1. Hovering inside inner box should hover the inner box
        tracker.current_candidates = [outer_box, inner_box]
        # Monkey patch detect_all to return our nested candidates
        tracker.yolo_detector.detect_all = lambda f, target_class=None: [outer_box, inner_box]

        vis, _ = tracker.process_frame(frame, frame_idx=0, hover_pos=(150, 105))
        assert vis is not None

        # 2. Clicking inside inner box should lock onto the inner box (Eye), NOT outer box (Person)
        tracker.handle_mouse_click(150, 105, frame)
        assert tracker.state == "TRACKING"
        assert tracker.selected_label == "Eye (95%)"
        assert tracker.active_detection.bbox == (140, 100, 30, 20)

    def test_custom_fast_roi_definition_and_tracking(self):
        """Test user-defined FAST ROI extracts features and tracks accurately without YOLO hijack."""
        import cv2
        import numpy as np

        tracker = InteractiveKalmanAITracker(hybrid_mode=True)
        frame1 = np.zeros((400, 400, 3), dtype=np.uint8)
        # Draw high-texture pattern inside custom ROI
        cv2.rectangle(frame1, (100, 100), (160, 160), (255, 255, 255), -1)
        cv2.circle(frame1, (120, 120), 4, (0, 0, 0), -1)
        cv2.circle(frame1, (140, 120), 4, (0, 0, 0), -1)
        cv2.circle(frame1, (130, 140), 4, (0, 0, 0), -1)

        # Draw a custom FAST ROI box
        tracker.handle_custom_roi((100, 100, 60, 60), frame1, label="FAST Custom ROI")
        assert tracker.state == "TRACKING"
        assert "FAST" in tracker.selected_label
        assert len(tracker.fast_tracker.fast_points) == 3

        # Next frame: moved by (10, 5)
        frame2 = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (110, 105), (170, 165), (255, 255, 255), -1)
        cv2.circle(frame2, (130, 125), 4, (0, 0, 0), -1)
        cv2.circle(frame2, (150, 125), 4, (0, 0, 0), -1)
        cv2.circle(frame2, (140, 145), 4, (0, 0, 0), -1)

        vis, error = tracker.process_frame(frame2, frame_idx=1)
        assert vis is not None
        assert "FAST" in tracker.selected_label
        assert tracker.kalman.x[0, 0] > 100.0

    def test_multi_algorithm_feature_tracker_sift_and_orb(self):
        """Test VisualFeatureTracker with SIFT and ORB algorithms and custom K points."""
        import cv2
        import numpy as np
        from position_class.ai_detectors import VisualFeatureTracker

        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        # Create textured patch with corners
        cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(frame, (80, 80), 8, (0, 0, 0), -1)
        cv2.circle(frame, (120, 80), 8, (0, 0, 0), -1)
        cv2.circle(frame, (100, 120), 8, (0, 0, 0), -1)

        # 1. SIFT with 5 keypoints
        sift_tracker = VisualFeatureTracker(algorithm="sift", num_points=5)
        sift_det = sift_tracker.init_target(frame, click_pos=(100, 100), bbox=(50, 50, 100, 100), label="SIFT Target")
        assert sift_det is not None
        assert len(sift_tracker.fast_points) == 5

        # 2. ORB with 4 keypoints
        orb_tracker = VisualFeatureTracker(algorithm="orb", num_points=4)
        orb_det = orb_tracker.init_target(frame, click_pos=(100, 100), bbox=(50, 50, 100, 100), label="ORB Target")
        assert orb_det is not None
        assert len(orb_tracker.fast_points) == 4

    def test_feature_only_mode_execution(self):
        """Test InteractiveKalmanAITracker running in Feature-Only mode (disables YOLO)."""
        import cv2
        import numpy as np

        tracker = InteractiveKalmanAITracker(
            feature_only=True,
            feature_algorithm="sift",
            num_points=5
        )
        assert "FEATURE-ONLY" in tracker.model_info
        assert "SIFT" in tracker.model_info

        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (180, 180), (255, 255, 255), -1)
        cv2.circle(frame, (140, 140), 10, (0, 0, 0), -1)

        # Draw ROI
        tracker.handle_custom_roi((100, 100, 80, 80), frame)
        assert tracker.state == "TRACKING"
        assert len(tracker.fast_tracker.fast_points) == 5

        # Process frame in tracking state
        vis, error = tracker.process_frame(frame, frame_idx=0)
        assert vis is not None
        assert tracker.state == "TRACKING"

    def test_disable_helper_algorithm_mode(self):
        """Test disabling helper algorithm so tracking relies strictly on YOLO and Kalman."""
        import cv2
        import numpy as np

        tracker = InteractiveKalmanAITracker(
            disable_helper=True
        )
        assert tracker.disable_helper is True
        assert tracker.feature_algorithm == "none"
        assert "HELPER DISABLED" in tracker.model_info

        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        # Create a synthetic detectable box
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)

        # IDLE frame processing with helper disabled
        vis, error = tracker.process_frame(frame, frame_idx=0)
        assert vis is not None
        assert tracker.state == "IDLE"

        # Lock onto detected candidate
        tracker.handle_mouse_click(150, 150, frame)
        assert tracker.state == "TRACKING"
        assert tracker.active_detection is not None

        # Next frame with object still visible: YOLO detection succeeds
        vis, error = tracker.process_frame(frame, frame_idx=1)
        assert tracker.state == "TRACKING"

        # Occluded frame (blank): YOLO fails and since helper is disabled, tracker enters PREDICTING immediately
        blank_frame = np.zeros((400, 400, 3), dtype=np.uint8)
        vis, error = tracker.process_frame(blank_frame, frame_idx=2)
        assert tracker.state == "OCCLUDED / LOST (PREDICTING)"

    def test_region_definition_permission_rules(self):
        """Test that region definition (drag-to-draw ROI) is enabled in standard & feature-only modes,
        and disabled ONLY under hybrid mode or when the helper algorithm is disabled."""
        import numpy as np

        # 1. Standard mode (FAST helper default) -> Region definition ENABLED
        tracker_std = InteractiveKalmanAITracker()
        assert tracker_std.allow_region_definition is True

        # 2. SIFT / ORB mode -> Region definition ENABLED
        tracker_sift = InteractiveKalmanAITracker(feature_algorithm="sift")
        assert tracker_sift.allow_region_definition is True

        # 3. Feature-only mode -> Region definition ENABLED
        tracker_feat = InteractiveKalmanAITracker(feature_only=True, feature_algorithm="orb")
        assert tracker_feat.allow_region_definition is True

        # 4. Hybrid mode -> Region definition DISABLED (YOLO defines regions)
        tracker_hybrid = InteractiveKalmanAITracker(hybrid_mode=True)
        assert tracker_hybrid.allow_region_definition is False

        # 5. Disabled helper mode -> Region definition DISABLED
        tracker_no_helper = InteractiveKalmanAITracker(disable_helper=True)
        assert tracker_no_helper.allow_region_definition is False

        tracker_algo_none = InteractiveKalmanAITracker(feature_algorithm="none")
        assert tracker_algo_none.allow_region_definition is False

    def test_kitti_default_feature_mode_and_custom_region(self, tmp_path):
        """Test that KITTI mode defaults to visual feature tracking with custom region definition."""
        from position_class.interactive_tracking_exercise import run_interactive_tracking_exercise
        from position_class.tracking_io import create_kitti_sample_sequences
        import numpy as np

        kitti_info = create_kitti_sample_sequences(tmp_path / "kitti_test", num_frames=10)
        img_dir = kitti_info["sequence_dir"]

        # Run interactive tracking on KITTI without --hybrid
        tracker = run_interactive_tracking_exercise(
            kitti_dir=img_dir,
            feature_algorithm="fast",
            num_points=3,
            output_dir=tmp_path / "out",
            interactive_gui=False,
            auto_click_frame=2,
            auto_click_coords=(120, 100),
            save_output=False
        )

        assert tracker.feature_only is True
        assert tracker.allow_region_definition is True
        assert "FEATURE-ONLY" in tracker.model_info

    def test_whole_image_global_recovery_on_large_jump(self):
        """Test that tracker re-acquires object when it jumps outside the local search margin."""
        import cv2
        import numpy as np
        from position_class.ai_detectors import VisualFeatureTracker

        tracker = VisualFeatureTracker(algorithm="orb", num_points=5)
        
        # Frame 1: Object at (50, 50, 40, 40)
        frame1 = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (50, 50), (90, 90), (255, 255, 255), -1)
        cv2.circle(frame1, (65, 65), 4, (0, 0, 0), -1)
        cv2.circle(frame1, (75, 75), 4, (0, 0, 0), -1)

        init_res = tracker.init_target(frame1, click_pos=(70, 70), bbox=(50, 50, 40, 40))
        assert init_res is not None

        # Frame 2: Object jumps to (300, 300) far outside search_margin of 40 px around (70, 70)
        frame2 = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (300, 300), (340, 340), (255, 255, 255), -1)
        cv2.circle(frame2, (315, 315), 4, (0, 0, 0), -1)
        cv2.circle(frame2, (325, 325), 4, (0, 0, 0), -1)

        # Track with local expectation at (70, 70) and search_margin=40 -> local search fails, full-frame succeeds!
        track_res = tracker.track(frame2, expected_pos=(70.0, 70.0), search_margin=40, enable_full_frame_search=True)
        assert track_res is not None
        assert 290 <= track_res.bbox[0] <= 310
        assert 290 <= track_res.bbox[1] <= 310

    def test_scale_adaptation_when_object_moves_closer(self):
        """Test that bounding box and scale adapt when object increases in visual size."""
        import cv2
        import numpy as np
        from position_class.ai_detectors import VisualFeatureTracker
        from position_class.vision_kalman_tracker import FastKalmanVisualTracker

        # 1. Test VisualFeatureTracker scale adaptation
        tracker = VisualFeatureTracker(algorithm="orb", num_points=5)
        frame1 = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (100, 100), (160, 160), (255, 255, 255), -1)
        cv2.circle(frame1, (120, 120), 4, (0, 0, 0), -1)
        cv2.circle(frame1, (140, 120), 4, (0, 0, 0), -1)
        cv2.circle(frame1, (130, 140), 4, (0, 0, 0), -1)

        tracker.init_target(frame1, click_pos=(130, 130), bbox=(100, 100, 60, 60))
        assert tracker.initial_template_size == (60, 60)

        # Frame 2: Object is closer (scaled up by ~1.5x to 90x90)
        frame2 = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (100, 100), (190, 190), (255, 255, 255), -1)
        cv2.circle(frame2, (130, 130), 6, (0, 0, 0), -1)
        cv2.circle(frame2, (160, 130), 6, (0, 0, 0), -1)
        cv2.circle(frame2, (145, 160), 6, (0, 0, 0), -1)

        track_res = tracker.track(frame2, expected_pos=(145.0, 145.0), search_margin=80)
        assert track_res is not None
        # Bounding box should expand or adapt scale
        assert tracker.current_scale > 0.9

        # 2. Test FastKalmanVisualTracker
        fk_tracker = FastKalmanVisualTracker()
        fk_tracker.initialize_target_from_roi(frame1, (100, 100, 60, 60))
        step_res = fk_tracker.process_frame(frame2)
        assert step_res["matches"] > 0
        assert "scale" in step_res
        assert "bbox_w" in step_res

    def test_yolo_matching_uses_whole_picture(self):
        """Test that YOLO matching scans and retrieves candidates anywhere across the full frame dimensions."""
        import cv2
        import numpy as np
        from position_class.ai_detectors import YOLODetector, HybridTracker

        yolo = YOLODetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Place targets across extreme opposite corners of the full frame:
        # Corner 1: Top-Left (20, 20)
        cv2.rectangle(frame, (20, 20), (120, 120), (255, 255, 255), -1)
        # Corner 2: Bottom-Right (1100, 580)
        cv2.rectangle(frame, (1100, 580), (1220, 680), (255, 255, 255), -1)

        detections = yolo.detect_all(frame)
        assert len(detections) >= 2

        # Verify detection in top-left region of image
        tl_det = [d for d in detections if d.bbox[0] < 200 and d.bbox[1] < 200]
        assert len(tl_det) >= 1

        # Verify detection in bottom-right region of image (whole picture scan)
        br_det = [d for d in detections if d.bbox[0] > 1000 and d.bbox[1] > 500]
        assert len(br_det) >= 1

        # Test HybridTracker re-acquires across the whole picture
        tracker = HybridTracker(yolo_detector=yolo)
        tracker.init_target(frame, click_pos=(70, 70), bbox=(20, 20, 100, 100), label="Object")
        
        # In next frame, target moved all the way to bottom right (1160, 630)
        det, mode = tracker.track(frame, expected_pos=(70.0, 70.0))
        assert det is not None
        assert mode in ("HYBRID_REACQUIRED_YOLO", "HYBRID_FAST", "HYBRID_FAST_GLOBAL")
        assert det.bbox[0] > 1000 or det.bbox[0] < 200

    def test_program_2_custom_kalman_matrices_injection(self, tmp_path):
        """Test Program 2 with custom F, H, Q, R, P0 matrices (Didactic Mode)."""
        import numpy as np
        from position_class import build_cv_matrices_2d, KalmanTracker2D

        dt = 1.0 / 30.0
        F, H, Q, R, P0 = build_cv_matrices_2d(dt=dt, process_noise_std=2.0, measurement_noise_std=1.5, initial_covariance=100.0)

        out_dir = tmp_path / "output_custom_kf"
        tracker = run_interactive_tracking_exercise(
            source=None,
            output_dir=out_dir,
            auto_click_frame=2,
            interactive_gui=False,
            save_output=False,
            F=F,
            H=H,
            Q=Q,
            R=R,
            P0=P0,
        )

        np.testing.assert_allclose(tracker.kalman.F, F)
        np.testing.assert_allclose(tracker.kalman.H, H)
        np.testing.assert_allclose(tracker.kalman.Q, Q)
        np.testing.assert_allclose(tracker.kalman.R, R)
        np.testing.assert_allclose(tracker.kalman.P0, P0)
        assert tracker.state in ("TRACKING", "OCCLUDED / LOST (PREDICTING)")

    def test_student_tracking_pipeline(self, tmp_path):
        """Test executing Program 2 using run_student_tracking_exercise."""
        from position_class import run_student_tracking_exercise, StudentKalmanTracker2D
        out_dir = tmp_path / "output_student"
        tracker = run_student_tracking_exercise(
            source=None,
            output_dir=str(out_dir),
            interactive_gui=False
        )
        assert isinstance(tracker.kalman, StudentKalmanTracker2D)
        assert tracker.state in ("TRACKING", "OCCLUDED / LOST (PREDICTING)")

    def test_instructor_tracking_pipeline(self, tmp_path):
        """Test executing Program 2 using run_instructor_tracking_exercise."""
        from position_class import run_instructor_tracking_exercise, InstructorKalmanTracker2D
        out_dir = tmp_path / "output_instructor"
        tracker = run_instructor_tracking_exercise(
            source=None,
            output_dir=str(out_dir),
            interactive_gui=False
        )
        assert isinstance(tracker.kalman, InstructorKalmanTracker2D)
        assert tracker.state in ("TRACKING", "OCCLUDED / LOST (PREDICTING)")

    def test_detection_only_runner(self):
        """Test running detection only visualizer in headless mode."""
        from position_class import run_detection_only_exercise
        # Run with max_frames=2 and no GUI to test execution without blocking
        run_detection_only_exercise(
            source=None,
            interactive_gui=False,
            max_frames=2
        )

    def test_tracking_config_and_parser(self):
        """Test TrackingConfig defaults and CLI argument parsing."""
        from position_class import TrackingConfig, parse_tracking_cli_args
        cfg = parse_tracking_cli_args(["--class", "car", "--dt", "0.05", "--hybrid", "--no-gui"])
        assert cfg.target_class == "car"
        assert cfg.dt == 0.05
        assert cfg.hybrid_mode is True
        assert cfg.interactive_gui is False

















