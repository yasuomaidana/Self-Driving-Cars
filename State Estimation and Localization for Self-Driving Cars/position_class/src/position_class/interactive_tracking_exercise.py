"""Program 2: Interactive AI (YOLO / FAST) + Kalman Visual Object Tracking Exercise.

Stage 1: IDLE - Not tracking anything. System waits for user to click an object.
Stage 2: TRACKING - User clicks target. AI/Vision detector segments target and computes
         its Center of Mass (CoM). Kalman Filter uses the Center of Mass to predict motion
         and track position, while publishing error relative to the image center.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from position_class import (
    YOLODetector,
    VisualFeatureTracker,
    FastFeatureTracker,
    HybridTracker,
    DetectionResult,
)
from position_class import KalmanTracker2D, build_cv_matrices_2d
from .base_kalman_tracker import BaseKalmanTracker2D
from .camera_stream import CameraStream, InteractiveMouseHandler
from .tracking_config import TrackingConfig, parse_tracking_cli_args
from position_class import ErrorPublisherService, TrackingErrorSignal
from position_class import TrackingVisualizer
from position_class import (
    generate_sample_tracking_dataset,
    TrackingDatasetLoader,
    KITTITrackingLoader,
    create_kitti_sample_sequences,
)


class InteractiveKalmanAITracker:
    """Orchestrates the 5 modular components for the interactive AI tracking exercise:
    1. Input Module (Video / Synthetic frames / Live Camera / KITTI)
    2. AI Detection & Segmentation Module (YOLO / FAST Feature matching)
    3. Kalman Filter Module (Constant Velocity 2D Motion Model with F, H, Q, R, P0 matrices)
    4. Image Output Module (Visualizer with crosshair, CoM, prediction, error line)
    5. Error Output Module (ErrorPublisherService publishing deviation to center)

    Didactic Kalman Filter Framework (Day 01 & Day 02):
    --------------------------------------------------
    State vector (4D):
        x = [p_x, p_y, v_x, v_y]^T
    Observation vector (2D):
        y = [p_x_meas, p_y_meas]^T

    Matrices:
        F: State Transition Matrix (4x4)  -> Kinematics propagation [p + v*dt]
        H: Measurement Matrix (2x4)        -> Visual CoM observation of [p_x, p_y]
        Q: Process Noise Covariance (4x4)  -> Acceleration disturbance uncertainty
        R: Measurement Noise Cov (2x2)     -> Pixel detection noise uncertainty
        P0: Initial Error Covariance (4x4) -> Initial state uncertainty
    """

    def __init__(
        self,
        yolo_model_name: str = "yolov8n.pt",
        dataset: Optional[str] = None,
        target_class: Optional[str] = None,
        auto_lock: bool = False,
        hybrid_mode: bool = False,
        feature_only: bool = False,
        disable_helper: bool = False,
        feature_algorithm: str = "fast",
        num_points: int = 3,
        dt: float = 1.0 / 30.0,
        process_noise: float = 1.5,
        measurement_noise: float = 2.0,
        service_name: str = "TrackingVisualErrorService",
        # Didactic Kalman Matrix Overrides (matching Day 01 & Day 02):
        F: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        kalman_tracker: Optional[Union[KalmanTracker2D, BaseKalmanTracker2D]] = None,
    ):
        self.dt = dt
        self.dataset = dataset
        self.target_class = target_class
        self.auto_lock = auto_lock
        self.hybrid_mode = hybrid_mode
        self.feature_only = feature_only
        raw_algo = feature_algorithm.lower().strip()
        self.disable_helper = disable_helper or (raw_algo in ("none", "off", "disabled", "no", "false", ""))
        self.feature_algorithm = "none" if self.disable_helper else raw_algo
        self.num_points = max(1, num_points)
        self.state = "IDLE"  # "IDLE" -> "TRACKING" -> "LOST"
        
        # 1. Kalman Filter Module (supports direct instance or explicit F, H, Q, R, P0 matrices)
        if kalman_tracker is not None:
            self.kalman = kalman_tracker
        else:
            self.kalman = KalmanTracker2D(
                dt=dt,
                process_noise_std=process_noise,
                measurement_noise_std=measurement_noise,
                F=F,
                H=H,
                Q=Q,
                R=R,
                P0=P0,
            )
        self.yolo_detector = YOLODetector(model_name=yolo_model_name, dataset=dataset)
        self.fast_tracker = VisualFeatureTracker(
            algorithm=self.feature_algorithm,
            num_points=self.num_points
        )
        self.hybrid_tracker = HybridTracker(
            yolo_detector=self.yolo_detector,
            feature_algorithm=self.feature_algorithm,
            num_points=self.num_points,
            fast_threshold=15,
            yolo_model_name=yolo_model_name,
            dataset=dataset
        )
        self.visualizer = TrackingVisualizer()
        self.error_service = ErrorPublisherService(service_name=service_name)
        
        base_ai_info = self.yolo_detector.get_model_info()
        algo_tag = self.feature_algorithm.upper()
        if self.disable_helper:
            self.model_info = f"{base_ai_info} [HELPER DISABLED]"
        elif self.feature_only:
            self.model_info = f"FEATURE-ONLY: {algo_tag} ({self.num_points} Points + Kalman)"
        elif self.hybrid_mode:
            self.model_info = f"HYBRID: {self.num_points}-pt {algo_tag} (Auto-Refresh) + {base_ai_info}"
        else:
            self.model_info = base_ai_info

        self.clicked_point: Optional[Tuple[int, int]] = None
        self.active_detection: Optional[DetectionResult] = None
        self.selected_label: str = "Selected Object"
        self.current_candidates: List[DetectionResult] = []
        self.is_custom_fast_mode: bool = False

    @property
    def allow_region_definition(self) -> bool:
        """Region definition (drag-to-draw ROI) is allowed ONLY when helper algorithm
        is enabled AND not running in hybrid mode (where YOLO defines regions)."""
        return (not self.hybrid_mode) and (not self.disable_helper)

    def handle_mouse_click(self, x: int, y: int, frame: np.ndarray) -> None:
        """User click callback to select and initialize target object."""
        self.clicked_point = (x, y)
        self.is_custom_fast_mode = False
        print(f"[Interactive Tracker] Target clicked at ({x}, {y}). Initializing AI detection...")

        # 1. Check if clicked within any YOLO / Salient candidate bounding boxes
        candidates = self.yolo_detector.detect_all(frame, target_class=self.target_class)
        matched = None
        matching_cands = [
            det for det in candidates
            if det.bbox[0] <= x <= det.bbox[0] + det.bbox[2]
            and det.bbox[1] <= y <= det.bbox[1] + det.bbox[3]
        ]
        if matching_cands:
            # Pick the candidate with the smallest bounding box area (innermost / most specific target)
            matched = min(matching_cands, key=lambda d: d.bbox[2] * d.bbox[3])

        # 2. Initialize FAST/ORB template with EXACT bounding box to maintain coordinate alignment
        if matched is None:
            if self.disable_helper:
                print(f"[Interactive Tracker] Helper algorithm is disabled. Please click directly on a detected YOLO candidate or press 1-9.")
                return
            matched = self.fast_tracker.init_target(frame, (x, y), window_size=70, label="Target Object")
            if self.hybrid_mode:
                self.hybrid_tracker.init_target(frame, (x, y), bbox=matched.bbox, label="Target Object", target_class=self.target_class)
        else:
            if not self.disable_helper:
                self.fast_tracker.init_target(
                    frame,
                    (int(matched.center_of_mass[0]), int(matched.center_of_mass[1])),
                    bbox=matched.bbox,
                    label=matched.label
                )
                if self.hybrid_mode:
                    self.hybrid_tracker.init_target(
                        frame,
                        (int(matched.center_of_mass[0]), int(matched.center_of_mass[1])),
                        bbox=matched.bbox,
                        label=matched.label,
                        target_class=self.target_class
                    )

        self.active_detection = matched
        self.selected_label = matched.label

        # 3. Initialize Kalman Filter with the Center of Mass
        cx, cy = matched.center_of_mass
        self.kalman.initialize(cx, cy)
        self.visualizer.reset_trails()
        self.state = "TRACKING"
        mode_tag = " (HYBRID FAST 3-pt + YOLO)" if self.hybrid_mode else (" (YOLO-Only, Helper Disabled)" if self.disable_helper else "")
        print(f"[Interactive Tracker] Locked Target{mode_tag}: '{self.selected_label}' at CoM: ({cx:.1f}, {cy:.1f})")

    def handle_custom_roi(self, bbox: Tuple[int, int, int, int], frame: np.ndarray, label: Optional[str] = None) -> None:
        """Extract FAST/SIFT/ORB corner features and descriptors from user-drawn rectangle.
        Runs in pure feature tracking mode (no YOLO)."""
        bx, by, bw, bh = bbox
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        if self.fast_tracker.algorithm == "none" or self.fast_tracker.fast is None:
            self.fast_tracker.algorithm = "fast"
            self.fast_tracker._init_detectors()
        algo_tag = self.fast_tracker.algorithm.upper()
        roi_label = label if label is not None else f"{algo_tag} Custom ROI"
        print(f"[Interactive Tracker] User drew custom {algo_tag} ROI: ({bx}, {by}, {bw}, {bh}). Tracking with {algo_tag} only ({self.num_points} pts)...")

        det = self.fast_tracker.init_target(
            frame=frame,
            click_pos=(int(cx), int(cy)),
            bbox=bbox,
            label=roi_label
        )
        self.is_custom_fast_mode = True
        self.active_detection = det
        self.selected_label = roi_label

        # Initialize Kalman Filter
        self.kalman.initialize(cx, cy)
        self.visualizer.reset_trails()
        self.state = "TRACKING"
        print(f"[Interactive Tracker] {algo_tag} Feature Tracking locked on custom ROI at CoM: ({cx:.1f}, {cy:.1f})")

    def select_candidate_by_index(self, index: int, frame: np.ndarray) -> bool:
        """Select a candidate directly by its 1-indexed number (e.g. from keyboard 1-9)."""
        if not self.current_candidates or index < 1 or index > len(self.current_candidates):
            return False
        candidate = self.current_candidates[index - 1]
        cx, cy = candidate.center_of_mass
        self.handle_mouse_click(int(cx), int(cy), frame)
        return True

    def reset(self) -> None:
        """Reset tracking to IDLE stage."""
        self.state = "IDLE"
        self.clicked_point = None
        self.active_detection = None
        self.selected_label = "Selected Object"
        self.current_candidates = []
        self.is_custom_fast_mode = False
        self.kalman.reset()
        self.visualizer.reset_trails()
        self.hybrid_tracker.is_initialized = False
        print("[Interactive Tracker] Reset to IDLE. Waiting for object selection click.")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        fps: Optional[float] = None,
        is_live: bool = False,
        is_paused: bool = False,
        hover_pos: Optional[Tuple[int, int]] = None,
        drawing_roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, Optional[TrackingErrorSignal]]:
        """Process a single frame through the complete pipeline."""
        h, w = frame.shape[:2]
        error_signal = None
        algo_tag = self.feature_algorithm.upper()

        if self.state == "IDLE":
            # Stage 1: IDLE - Not tracking anything
            vis = frame.copy()
            candidates = [] if self.feature_only else self.yolo_detector.detect_all(frame, target_class=self.target_class)
            self.current_candidates = candidates

            # Find if user is hovering over any candidate (pick smallest / innermost box if overlapping)
            hovered_idx = None
            if hover_pos is not None and hover_pos[0] is not None and hover_pos[1] is not None and len(candidates) > 0:
                hx, hy = hover_pos
                matching_indices = [
                    idx for idx, c in enumerate(candidates)
                    if c.bbox[0] <= hx <= c.bbox[0] + c.bbox[2]
                    and c.bbox[1] <= hy <= c.bbox[1] + c.bbox[3]
                ]
                if matching_indices:
                    hovered_idx = min(matching_indices, key=lambda i: candidates[i].bbox[2] * candidates[i].bbox[3])

            # Draw alpha color fill ONLY for the hovered candidate box
            if hovered_idx is not None and drawing_roi is None:
                c_hov = candidates[hovered_idx]
                hbx, hby, hbw, hbh = c_hov.bbox
                overlay = vis.copy()
                cv2.rectangle(overlay, (hbx, hby), (hbx + hbw, hby + hbh), (0, 220, 255), -1)
                cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

            # Draw candidate boxes (draw non-hovered first sorted by area descending, then hovered on top)
            draw_order = [i for i in range(len(candidates)) if i != hovered_idx]
            draw_order.sort(key=lambda i: candidates[i].bbox[2] * candidates[i].bbox[3], reverse=True)
            if hovered_idx is not None:
                draw_order.append(hovered_idx)

            # Draw candidate box outlines and numbered object name badges
            for idx in draw_order:
                c = candidates[idx]
                bx, by, bw, bh = c.bbox
                is_hov = (idx == hovered_idx and drawing_roi is None)
                slot_num = f"[{idx+1}] " if idx < 9 else ""
                
                if is_hov:
                    # Highlighted border for hovered box
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 255), 3, cv2.LINE_AA)
                    tag = f" CLICK / PRESS {idx+1}: {c.label} "
                    tag_color = (0, 255, 255)
                    bg_color = (10, 10, 80)
                else:
                    # Clean outline for non-hovered box
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 190, 240), 2, cv2.LINE_AA)
                    tag = f" {slot_num}{c.label} "
                    tag_color = (255, 255, 255)
                    bg_color = (25, 25, 25)

                # Name badge above the box
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                badge_y1 = max(2, by - th - 8)
                badge_y2 = badge_y1 + th + 6
                badge_x2 = min(w - 2, bx + tw + 6)
                
                pill = vis.copy()
                cv2.rectangle(pill, (bx, badge_y1), (badge_x2, badge_y2), bg_color, -1)
                cv2.addWeighted(pill, 0.85, vis, 0.15, 0, vis)
                cv2.rectangle(vis, (bx, badge_y1), (badge_x2, badge_y2), tag_color if is_hov else (0, 190, 240), 1)
                cv2.putText(vis, tag, (bx + 3, badge_y2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, tag_color, 1, cv2.LINE_AA)

            # Draw image center crosshair
            cx_img, cy_img = w // 2, h // 2
            cv2.drawMarker(vis, (cx_img, cy_img), (200, 200, 200), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
            cv2.circle(vis, (cx_img, cy_img), 4, (255, 255, 255), -1)

            # Prominent Instructions Banner
            banner_overlay = vis.copy()
            cv2.rectangle(banner_overlay, (20, 15), (w - 20, 75), (0, 0, 0), -1)
            cv2.addWeighted(banner_overlay, 0.75, vis, 0.25, 0, vis)
            cv2.rectangle(vis, (20, 15), (w - 20, 75), (0, 180, 255), 2)
            
            filter_info = f" [FILTER: {self.target_class.upper()}]" if self.target_class else " [ALL CLASSES]"
            if self.feature_only:
                mode_info = f" [FEATURE-ONLY: {algo_tag} {self.num_points}-PT]"
                hint_text = f"--> DRAG WITH MOUSE TO DRAW {algo_tag} BOUNDING BOX <--"
            elif self.disable_helper:
                mode_info = " [YOLO-ONLY (HELPER DISABLED)]"
                hint_text = "--> CLICK CANDIDATE OR PRESS 1-9 (HELPER DISABLED) <--"
            elif self.hybrid_mode:
                mode_info = f" [HYBRID: {algo_tag} + YOLO]"
                hint_text = "--> CLICK CANDIDATE OR PRESS 1-9 (HYBRID: YOLO DEFINES REGION) <--"
            else:
                mode_info = ""
                hint_text = f"--> DRAG-TO-DRAW {algo_tag} ROI, CLICK CANDIDATE, OR PRESS 1-9 <--"

            paused_hint = " [FREEZE SELECTION]" if is_paused else " [SPACE=FREEZE]"
            cv2.putText(vis, f"IDLE: NOT TRACKING{filter_info}{mode_info}{paused_hint}", (35, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, hint_text,
                        (35, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # Telemetry HUD
            status_text = "FREEZE SELECT" if is_paused else "IDLE"
            self.visualizer._draw_hud(vis, frame_idx, status_text, None, None, None, (cx_img, cy_img), fps, is_live, self.model_info, None)

            # Draw active user-drawn rectangle in real-time (rubber-band) in IDLE / selection state
            if drawing_roi is not None and self.allow_region_definition:
                dbx, dby, dbw, dbh = drawing_roi
                overlay = vis.copy()
                cv2.rectangle(overlay, (dbx, dby), (dbx + dbw, dby + dbh), (0, 255, 120), -1)
                cv2.addWeighted(overlay, 0.30, vis, 0.70, 0, vis)
                cv2.rectangle(vis, (dbx, dby), (dbx + dbw, dby + dbh), (0, 255, 120), 2, cv2.LINE_AA)
                roi_tag = f" {algo_tag} Feature ROI ({dbw}x{dbh}) "
                cv2.putText(vis, roi_tag, (dbx + 4, max(20, dby - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 2, cv2.LINE_AA)

            return vis, None

        # Stage 2: TRACKING with AI + Kalman Filter
        # 1. Kalman Predict
        pred_x, pred_y, _, _ = self.kalman.predict()

        # 2. AI / Vision Detect:
        meas_com = None
        bbox = None
        polygon = None
        fast_pts = None

        if self.feature_only or self.is_custom_fast_mode:
            # Feature-Only / Custom Region: Pure FAST / SIFT / ORB Tracking ONLY (No YOLO involvement)
            det = self.fast_tracker.track(
                frame=frame,
                expected_pos=(pred_x, pred_y),
                auto_refresh=True,
                enable_full_frame_search=True
            )
            if det is not None:
                meas_com = det.center_of_mass
                bbox = det.bbox
                polygon = det.polygon
                fast_pts = det.fast_points
                self.state = "TRACKING"
            else:
                self.state = "OCCLUDED / LOST (PREDICTING)"

        elif self.hybrid_mode:
            # Hybrid Mode: K-point feature tracking derived & refreshed from previous tracking regions + YOLO re-acquisition
            uncertainty_std = float(np.sqrt(self.kalman.P[0, 0] + self.kalman.P[1, 1]))
            det, track_mode = self.hybrid_tracker.track(
                frame=frame,
                expected_pos=(pred_x, pred_y),
                kalman_uncertainty_std=uncertainty_std,
                target_class=self.target_class
            )
            if det is not None:
                meas_com = det.center_of_mass
                bbox = det.bbox
                polygon = det.polygon
                fast_pts = det.fast_points
                if track_mode == "HYBRID_REACQUIRED_YOLO":
                    self.state = "RE-ACQUIRED (FULL-FRAME YOLO)"
                    self.selected_label = det.label
                elif track_mode == "HYBRID_FAST_GLOBAL":
                    self.state = f"RE-ACQUIRED (FULL-FRAME {algo_tag})"
                else:
                    self.state = f"TRACKING (HYBRID {algo_tag} {self.num_points}-PT)"
            else:
                self.state = "OCCLUDED / LOST (PREDICTING)"

        else:
            # Standard AI / Template tracking with full-frame recovery fallback
            is_custom_roi = ("Custom ROI" in self.selected_label or "Target Object" in self.selected_label)

            # Try YOLO candidate match only if tracking a semantic AI class (not a custom texture ROI)
            if not is_custom_roi:
                yolo_candidates = self.yolo_detector.detect_all(frame, target_class=self.target_class)
                best_cand = None
                min_dist = float("inf")
                for c in yolo_candidates:
                    ccx, ccy = c.center_of_mass
                    dist = float(np.hypot(ccx - pred_x, ccy - pred_y))
                    if dist < min_dist:
                        min_dist = dist
                        best_cand = c

                if best_cand is not None and (min_dist < 80.0 or best_cand.confidence >= 0.60):
                    meas_com = best_cand.center_of_mass
                    bbox = best_cand.bbox
                    polygon = best_cand.polygon
                    self.state = "TRACKING" if min_dist < 80.0 else "RE-ACQUIRED (YOLO)"

            # If no YOLO match, run feature template matching ONLY if helper is enabled
            if meas_com is None and not self.disable_helper:
                det = self.fast_tracker.track(frame, expected_pos=(pred_x, pred_y), enable_full_frame_search=True)
                if det is not None:
                    meas_com = det.center_of_mass
                    bbox = det.bbox
                    polygon = det.polygon
                    fast_pts = det.fast_points
                    self.state = "TRACKING"
                else:
                    # No visual confirmation: rely 100% on Kalman Filter Dead Reckoning
                    self.state = "OCCLUDED / LOST (PREDICTING)"
            elif meas_com is None:
                # Helper disabled: strictly dead-reckoning prediction
                self.state = "OCCLUDED / LOST (PREDICTING)"

        # 3. Kalman Update (retains kinematic constant-velocity dead reckoning if meas_com is None)
        est_x, est_y, est_vx, est_vy = self.kalman.update(meas_com)

        # 4. Synthesize Dead-Reckoning Bounding Box if visual measurement was lost
        if bbox is None and self.fast_tracker.is_initialized:
            tw, th = self.fast_tracker.template_size
            bbox = (int(est_x - tw / 2.0), int(est_y - th / 2.0), int(tw), int(th))

        # 5. Error Output Service: Publish deviation from image center
        error_signal = TrackingErrorSignal.compute(
            frame_index=frame_idx,
            target_pos=(est_x, est_y),
            image_shape=(h, w),
            is_tracked=(meas_com is not None),
            predicted_pos=(pred_x, pred_y)
        )
        self.error_service.publish(error_signal)

        # 6. Visualizer Module: Render segmented rectangle with alpha fill, CoM, prediction, error line, and FAST 3-pt markers
        status_disp = f"PAUSED ({self.state})" if is_paused else self.state
        vis = self.visualizer.draw_frame(
            frame=frame,
            frame_idx=frame_idx,
            bbox=bbox,
            polygon=polygon,
            meas_point=meas_com,
            est_point=(est_x, est_y),
            pred_point=(pred_x, pred_y),
            velocity=(est_vx, est_vy),
            error_distance=error_signal.error_distance,
            status_text=status_disp,
            label=self.selected_label,
            fps=fps,
            is_live=is_live,
            model_name=self.model_info,
            selection_alpha=0.30,
            fill_color=(0, 220, 255),
            fast_points=fast_pts
        )

        # Draw active user-drawn rectangle in real-time (rubber-band) across all states
        if drawing_roi is not None and self.allow_region_definition:
            dbx, dby, dbw, dbh = drawing_roi
            overlay = vis.copy()
            cv2.rectangle(overlay, (dbx, dby), (dbx + dbw, dby + dbh), (0, 255, 120), -1)
            cv2.addWeighted(overlay, 0.30, vis, 0.70, 0, vis)
            cv2.rectangle(vis, (dbx, dby), (dbx + dbw, dby + dbh), (0, 255, 120), 2, cv2.LINE_AA)
            roi_tag = f" {algo_tag} Feature ROI ({dbw}x{dbh}) "
            cv2.putText(vis, roi_tag, (dbx + 4, max(20, dby - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 2, cv2.LINE_AA)

        return vis, error_signal


def run_interactive_tracking_exercise(
    source: Optional[Union[str, int]] = None,
    kitti_dir: Optional[Union[str, Path]] = None,
    kitti_label: Optional[Union[str, Path]] = None,
    use_kitti_demo: bool = False,
    camera_id: Optional[int] = None,
    model_name: str = "yolov8n.pt",
    dataset: Optional[str] = None,
    target_class: Optional[str] = None,
    auto_lock: bool = False,
    hybrid_mode: bool = False,
    feature_only: bool = False,
    disable_helper: bool = False,
    feature_algorithm: str = "fast",
    num_points: int = 3,
    start_paused: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    auto_click_frame: int = 5,
    auto_click_coords: Optional[Tuple[int, int]] = None,
    interactive_gui: bool = True,
    save_output: bool = True,
    max_live_frames: Optional[int] = None,
    dt: float = 1.0 / 30.0,
    process_noise: float = 1.5,
    measurement_noise: float = 2.0,
    F: Optional[np.ndarray] = None,
    H: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    kalman_tracker: Optional[Union[KalmanTracker2D, BaseKalmanTracker2D]] = None,
) -> InteractiveKalmanAITracker:
    """Run Program 2: Interactive AI + Kalman Object Tracking Exercise.

    Supports:
    - Target class filtering (e.g. --target-class person, --class car, --class cup)
    - Dataset selection (e.g. --dataset kitti, --dataset coco, --dataset visdrone)
    - Hybrid mode (--hybrid) tracking 3 dynamically refreshed FAST features with YOLO re-acquisition
    - Helper algorithm selection (FAST, SIFT, ORB, or --disable-helper)
    - Auto-lock mode (--auto-lock) to lock the first detected candidate without clicking
    - Keyboard quick selection (press 1-9 to lock onto candidate [1]-[9])
    - Pause & Click target selection (press Space anytime to freeze and pick an object)
    - Semi-transparent alpha colored rectangle selection with object name badge
    - YOLO models (yolov8n.pt, yolov8s.pt, yolo26n.pt, etc.) or OpenCV AI vision fallback
    - Live webcam/camera capture (camera_id=0 or --camera)
    - KITTI driving sequences (--kitti or --kitti-dir)
    - Synthetic test sequences

    Args:
        source: Video file path, image directory, camera index, or None.
        kitti_dir: Path to KITTI image_02 sequence folder.
        kitti_label: Path to KITTI label_02 text file.
        use_kitti_demo: If True, uses realistic KITTI driving street sequence.
        camera_id: Camera device index for live webcam mode (e.g. 0).
        model_name: YOLO model file or name (e.g. "yolov8n.pt", "yolo26n.pt", "kitti_best.pt").
        dataset: Dataset specification for YOLO classes (e.g. "kitti", "coco", "visdrone", "bdd100k").
        target_class: Target object class to filter (e.g. "person", "car", "bottle", "cup").
        auto_lock: If True, automatically locks onto the first detected matching candidate.
        hybrid_mode: If True, uses FAST 3-point dynamic feature tracking + YOLO re-acquisition.
        feature_only: If True, tracks purely with visual features without YOLO.
        disable_helper: If True, disables visual feature helper algorithm and tracks YOLO only.
        feature_algorithm: Feature algorithm ('fast', 'sift', 'orb', 'none').
        num_points: Number of feature keypoints (default 3).
        start_paused: If True, starts video in paused state for easy initial selection.
        output_dir: Directory for saved outputs and error logs.
        auto_click_frame: In automated/test mode, frame index at which to trigger target selection click.
        auto_click_coords: Optional click coordinates (x, y) for target selection.
        interactive_gui: If True, opens OpenCV interactive window with mouse click support.
        save_output: If True, saves annotated frames and CSV error log.
        max_live_frames: Maximum frames to process in live camera mode before exit (None for infinite).

    Returns:
        InteractiveKalmanAITracker instance.
    """
    if output_dir is None:
        output_dir = Path("tracking_output_interactive")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    effective_dataset = dataset if dataset is not None else ("kitti" if (use_kitti_demo or kitti_dir is not None) else None)
    is_kitti = bool(use_kitti_demo or kitti_dir is not None)
    effective_feature_only = feature_only or (is_kitti and not hybrid_mode and not disable_helper)

    tracker = InteractiveKalmanAITracker(
        yolo_model_name=model_name,
        dataset=effective_dataset,
        target_class=target_class,
        auto_lock=auto_lock,
        hybrid_mode=hybrid_mode,
        feature_only=effective_feature_only,
        disable_helper=disable_helper,
        feature_algorithm=feature_algorithm,
        num_points=num_points,
        dt=dt,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        F=F,
        H=H,
        Q=Q,
        R=R,
        P0=P0,
        kalman_tracker=kalman_tracker,
    )
    csv_log_path = out_path / "interactive_error_log.csv"
    tracker.error_service.enable_csv_logging(csv_log_path)

    annotated_dir = out_path / "annotated_frames"
    if save_output:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    # Mouse callback state
    mouse_click_received = [False]
    click_xy = [0, 0]
    hover_xy = [None, None]
    
    # Drag-to-draw ROI state
    is_drawing_roi = [False]
    roi_start = [0, 0]
    roi_current = [0, 0]
    drawn_bbox = [None]

    def on_mouse(event, x, y, flags, param):
        can_draw = tracker.allow_region_definition
        if event == cv2.EVENT_LBUTTONDOWN:
            if can_draw:
                is_drawing_roi[0] = True
                roi_start[0] = x
                roi_start[1] = y
                roi_current[0] = x
                roi_current[1] = y
            else:
                is_drawing_roi[0] = False
        elif event == cv2.EVENT_MOUSEMOVE:
            hover_xy[0] = x
            hover_xy[1] = y
            if is_drawing_roi[0] and can_draw:
                roi_current[0] = x
                roi_current[1] = y
            elif (flags & cv2.EVENT_FLAG_LBUTTON) and can_draw:
                if not is_drawing_roi[0]:
                    is_drawing_roi[0] = True
                    roi_start[0] = x
                    roi_start[1] = y
                roi_current[0] = x
                roi_current[1] = y
        elif event == cv2.EVENT_LBUTTONUP:
            if is_drawing_roi[0] and can_draw:
                is_drawing_roi[0] = False
                roi_current[0] = x
                roi_current[1] = y
                dx = abs(roi_current[0] - roi_start[0])
                dy = abs(roi_current[1] - roi_start[1])
                if dx >= 4 and dy >= 4:
                    # User dragged a valid ROI box
                    bx = min(roi_start[0], roi_current[0])
                    by = min(roi_start[1], roi_current[1])
                    drawn_bbox[0] = (bx, by, dx, dy)
                else:
                    # Quick click without drag
                    mouse_click_received[0] = True
                    click_xy[0] = x
                    click_xy[1] = y
            else:
                is_drawing_roi[0] = False
                mouse_click_received[0] = True
                click_xy[0] = x
                click_xy[1] = y

    window_name = "Program 2: Interactive AI & Kalman Tracking Exercise"
    if interactive_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse)

    # Check if live camera mode
    is_live_camera = camera_id is not None or (isinstance(source, int) or (isinstance(source, str) and source.isdigit()))
    active_camera_idx = camera_id if camera_id is not None else (int(source) if (isinstance(source, int) or (isinstance(source, str) and source.isdigit())) else None)

    if is_live_camera:
        cam_idx = active_camera_idx if active_camera_idx is not None else 0
        print(f"\n[Program 2] Initializing Live Camera (Device ID: {cam_idx})...")
        print(f"  • AI Model: {tracker.model_info}")
        if target_class:
            print(f"  • Target Class Filter: '{target_class}'")
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"[Program 2] Warning: Could not open camera {cam_idx}. Falling back to synthetic demo.")
            is_live_camera = False
        else:
            # Warm up camera sensor to avoid initial dark/black auto-exposure frame
            print("[Program 2] Calibrating camera auto-exposure...")
            last_frame = None
            for _ in range(15):
                ret, warm_frame = cap.read()
                if ret and warm_frame is not None:
                    last_frame = warm_frame
                    if float(np.mean(warm_frame)) > 15.0:
                        break

            print(f"[Program 2] Live camera active! Point camera at an object and LEFT-CLICK, PRESS 1-9, or DRAG TO DRAW ROI.")
            print(f"  • Controls: Space = Pause/Freeze | Drag = Draw FAST Box | Click/1-9 = Select | 'r' = Reset | 'q' = Quit")
            print(f"{'='*75}")

            frame_idx = 0
            paused = start_paused or auto_lock
            prev_time = cv2.getTickCount()
            fps = 30.0

            while True:
                if not paused or last_frame is None:
                    ret, frame = cap.read()
                    if not ret:
                        print("[Program 2] Failed to grab frame from live camera.")
                        break
                    last_frame = frame

                    # Compute real-time FPS
                    curr_time = cv2.getTickCount()
                    time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
                    prev_time = curr_time
                    if time_diff > 0:
                        fps = 0.9 * fps + 0.1 * (1.0 / time_diff)

                    # Automated click in test mode
                    if not interactive_gui and frame_idx == auto_click_frame and tracker.state == "IDLE":
                        click_pt = auto_click_coords or (frame.shape[1] // 2, frame.shape[0] // 2)
                        tracker.handle_mouse_click(click_pt[0], click_pt[1], frame)

                frame = last_frame

                # Handle user-drawn custom bounding box for feature extraction
                if drawn_bbox[0] is not None and tracker.allow_region_definition:
                    custom_box = drawn_bbox[0]
                    drawn_bbox[0] = None
                    tracker.handle_custom_roi(custom_box, frame)
                    if paused:
                        paused = False
                        algo_name = tracker.feature_algorithm.upper()
                        print(f"[Program 2] {algo_name} ROI locked! Resuming tracking.")

                # Check for user mouse click (works whether running or paused!)
                elif mouse_click_received[0]:
                    mouse_click_received[0] = False
                    tracker.handle_mouse_click(click_xy[0], click_xy[1], frame)
                    if paused and tracker.state == "TRACKING":
                        paused = False
                        print(f"[Program 2] Target locked! Resuming tracking.")

                # Calculate live rubber-band drawing rectangle
                active_drag_rect = None
                if is_drawing_roi[0] and tracker.allow_region_definition:
                    rx = min(roi_start[0], roi_current[0])
                    ry = min(roi_start[1], roi_current[1])
                    rw = abs(roi_current[0] - roi_start[0])
                    rh = abs(roi_current[1] - roi_start[1])
                    if rw > 2 and rh > 2:
                        active_drag_rect = (rx, ry, rw, rh)

                # Process frame
                current_hover = (hover_xy[0], hover_xy[1]) if hover_xy[0] is not None else None
                vis_frame, error_sig = tracker.process_frame(
                    frame,
                    frame_idx,
                    fps=fps,
                    is_live=True,
                    is_paused=paused,
                    hover_pos=current_hover,
                    drawing_roi=active_drag_rect
                )

                if error_sig is not None and frame_idx % 15 == 0 and not paused:
                    print(f"[Live Frame #{frame_idx:04d} | {fps:4.1f} FPS] Target: '{tracker.selected_label}' | "
                          f"Center Error: {error_sig.error_distance:5.1f} px (dx={error_sig.error_x:+5.1f}, dy={error_sig.error_y:+5.1f})")

                if save_output and frame_idx < 300 and not paused:
                    cv2.imwrite(str(annotated_dir / f"live_{frame_idx:04d}.png"), vis_frame)

                if not paused:
                    frame_idx += 1
                    if max_live_frames is not None and frame_idx >= max_live_frames:
                        break

                if interactive_gui:
                    cv2.imshow(window_name, vis_frame)
                    key = cv2.waitKey(20 if paused else 1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                    elif key == ord('r'):
                        tracker.reset()
                    elif key == ord(' '):
                        paused = not paused
                        print(f"[Program 2] {'Paused' if paused else 'Resumed'}. Drag to draw box, click, or press 1-9.")
                    elif (key in (10, 13) or key == ord('\r') or key == ord('\n')) and tracker.state == "IDLE":
                        # Enter key auto-locks candidate #1
                        if tracker.select_candidate_by_index(1, frame):
                            paused = False
                            print(f"[Program 2] Selected candidate #1: '{tracker.selected_label}'. Resuming tracking.")
                    elif ord('1') <= key <= ord('9') and tracker.state == "IDLE":
                        idx = key - ord('0')
                        if tracker.select_candidate_by_index(idx, frame):
                            paused = False
                            print(f"[Program 2] Selected candidate #{idx}: '{tracker.selected_label}'. Resuming tracking.")

            cap.release()
            if interactive_gui:
                cv2.destroyAllWindows()

            tracker.error_service.close()
            metrics = tracker.error_service.compute_metrics()
            print(f"{'='*75}")
            print("[Program 2] Live Camera Session Complete!")
            print(f"  • Total Tracked Frames:        {metrics.get('total_samples', 0)}")
            print(f"  • Mean Distance Error to Center: {metrics['mean_error']:.2f} px")
            print(f"  • CSV Error Log Saved:            {csv_log_path}")
            print(f"{'='*75}\n")
            return tracker

    # 2. Sequence / File Mode
    frames: List[np.ndarray] = []
    if kitti_dir is not None and Path(kitti_dir).exists():
        req_class = target_class if target_class is not None else "Car"
        print(f"[Program 2] Loading KITTI tracking sequence from: {kitti_dir} (Class: {req_class})...")
        dataset = KITTITrackingLoader.load_kitti_sequence(
            sequence_dir=kitti_dir,
            label_file=kitti_label,
            class_type=req_class
        )
        frames = [dataset.load_frame_image(i) for i in range(len(dataset))]
        if auto_click_coords is None and len(dataset) > 0:
            first_ann = dataset[auto_click_frame if auto_click_frame < len(dataset) else 0]
            if first_ann.center_of_mass:
                auto_click_coords = (int(first_ann.center_of_mass[0]), int(first_ann.center_of_mass[1]))
    elif use_kitti_demo:
        print("[Program 2] Generating realistic KITTI-format driving sequence (1242x375)...")
        kitti_info = create_kitti_sample_sequences(out_path / "kitti_demo", num_frames=80)
        dataset = kitti_info["dataset"]
        frames = [dataset.load_frame_image(i) for i in range(len(dataset))]
        if auto_click_coords is None and len(dataset) > 0:
            first_ann = dataset[auto_click_frame if auto_click_frame < len(dataset) else 0]
            if first_ann.center_of_mass:
                auto_click_coords = (int(first_ann.center_of_mass[0]), int(first_ann.center_of_mass[1]))
    elif source is not None and isinstance(source, (str, Path)) and Path(source).suffix.lower() == ".json":
        dataset = TrackingDatasetLoader.load_from_json(source)
        frames = [dataset.load_frame_image(i) for i in range(len(dataset))]
    elif source is not None and isinstance(source, (str, Path)) and Path(source).is_file():
        cap = cv2.VideoCapture(str(source))
        while cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()
    else:
        print("[Program 2] No input source specified. Generating high-resolution test sequence...")
        sample_dir = out_path / "test_frames"
        json_path, dataset = generate_sample_tracking_dataset(sample_dir, num_frames=80)
        frames = [dataset.load_frame_image(i) for i in range(len(dataset))]
        if auto_click_coords is None and len(dataset) > 0:
            first_ann = dataset[auto_click_frame if auto_click_frame < len(dataset) else 0]
            if first_ann.center_of_mass:
                auto_click_coords = (int(first_ann.center_of_mass[0]), int(first_ann.center_of_mass[1]))

    print(f"\n[Program 2] Starting interactive session...")
    print(f"  • AI Model: {tracker.model_info}")
    if target_class:
        print(f"  • Target Class Filter: '{target_class}'")
    print(f"  • Controls: Space = Pause/Freeze | Drag = Draw FAST Box | Click/1-9 = Pick Candidate | 'r' = Reset | 'q' = Quit")
    print(f"{'='*75}")

    frame_idx = 0
    paused = start_paused or auto_lock

    while frame_idx < len(frames):
        frame = frames[frame_idx]

        # Handle user-drawn custom bounding box for feature extraction
        if drawn_bbox[0] is not None and tracker.allow_region_definition:
            custom_box = drawn_bbox[0]
            drawn_bbox[0] = None
            tracker.handle_custom_roi(custom_box, frame)
            if paused:
                paused = False
                algo_name = tracker.feature_algorithm.upper()
                print(f"[Program 2] {algo_name} ROI locked! Resuming tracking.")

        # Check for user mouse click (can happen while running or paused!)
        elif mouse_click_received[0]:
            mouse_click_received[0] = False
            tracker.handle_mouse_click(click_xy[0], click_xy[1], frame)
            if paused and tracker.state == "TRACKING":
                paused = False
                print(f"[Program 2] Target locked! Resuming tracking.")

        # In non-interactive / headless test mode: simulate user click at auto_click_frame
        if not interactive_gui and frame_idx == auto_click_frame and tracker.state == "IDLE":
            click_pt = auto_click_coords or (frame.shape[1] // 2, frame.shape[0] // 2)
            tracker.handle_mouse_click(click_pt[0], click_pt[1], frame)

        # Calculate live rubber-band drawing rectangle
        active_drag_rect = None
        if is_drawing_roi[0] and tracker.allow_region_definition:
            rx = min(roi_start[0], roi_current[0])
            ry = min(roi_start[1], roi_current[1])
            rw = abs(roi_current[0] - roi_start[0])
            rh = abs(roi_current[1] - roi_start[1])
            if rw > 2 and rh > 2:
                active_drag_rect = (rx, ry, rw, rh)

        # Process frame
        current_hover = (hover_xy[0], hover_xy[1]) if hover_xy[0] is not None else None
        vis_frame, error_sig = tracker.process_frame(
            frame,
            frame_idx,
            is_paused=paused,
            hover_pos=current_hover,
            drawing_roi=active_drag_rect
        )

        if error_sig is not None and frame_idx % 10 == 0 and not paused:
            print(f"[Frame {frame_idx:03d}] Target: '{tracker.selected_label}' | "
                  f"Center Error: {error_sig.error_distance:.2f} px (dx={error_sig.error_x:+.1f}, dy={error_sig.error_y:+.1f})")

        if save_output and not paused:
            cv2.imwrite(str(annotated_dir / f"frame_{frame_idx:04d}.png"), vis_frame)

        if interactive_gui:
            cv2.imshow(window_name, vis_frame)
            key = cv2.waitKey(30 if not paused else 20) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                tracker.reset()
            elif key == ord(' '):
                paused = not paused
                print(f"[Program 2] {'Paused' if paused else 'Resumed'}. Drag to draw box, click, or press 1-9 to select.")
            elif (key in (10, 13) or key == ord('\r') or key == ord('\n')) and tracker.state == "IDLE":
                if tracker.select_candidate_by_index(1, frame):
                    paused = False
                    print(f"[Program 2] Selected candidate #1: '{tracker.selected_label}'. Resuming tracking.")
            elif ord('1') <= key <= ord('9') and tracker.state == "IDLE":
                idx = key - ord('0')
                if tracker.select_candidate_by_index(idx, frame):
                    paused = False
                    print(f"[Program 2] Selected candidate #{idx}: '{tracker.selected_label}'. Resuming tracking.")

        if not paused:
            frame_idx += 1

    if interactive_gui:
        cv2.destroyAllWindows()

    tracker.error_service.close()

    metrics = tracker.error_service.compute_metrics()
    print(f"{'='*75}")
    print("[Program 2] Interactive Session Complete!")
    print(f"  • Total Tracked Points:         {metrics.get('total_samples', 0)}")
    print(f"  • Mean Distance Error to Center: {metrics['mean_error']:.2f} px")
    print(f"  • Maximum Distance Error:        {metrics['max_error']:.2f} px")
    print(f"  • CSV Error Log Saved:            {csv_log_path}")
    print(f"{'='*75}\n")

    return tracker


def main():
    config = parse_tracking_cli_args(description="Program 2: Interactive AI (YOLO/FAST) + Kalman Visual Tracking")
    run_interactive_tracking_exercise(
        source=config.source,
        kitti_dir=config.kitti_dir,
        kitti_label=config.kitti_label,
        use_kitti_demo=config.use_kitti_demo,
        camera_id=config.camera_id,
        model_name=config.model_name,
        dataset=config.dataset,
        target_class=config.target_class,
        auto_lock=config.auto_lock,
        hybrid_mode=config.hybrid_mode,
        feature_only=config.feature_only,
        disable_helper=config.disable_helper,
        feature_algorithm=config.feature_algorithm,
        num_points=config.num_points,
        start_paused=config.start_paused,
        output_dir=config.output_dir,
        interactive_gui=config.interactive_gui,
        dt=config.dt,
        process_noise=config.process_noise,
        measurement_noise=config.measurement_noise,
    )


if __name__ == "__main__":
    main()


