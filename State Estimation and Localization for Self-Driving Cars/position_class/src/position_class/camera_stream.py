"""Camera and Video Feed abstraction module with Interactive UI handlers.

Provides:
- CameraStream: Robust OpenCV video/camera capture manager with sensor calibration and FPS tracking.
- InteractiveMouseHandler: Thread-safe mouse click, hover, and drag-to-draw ROI bounding box management.
- show_detection_only_stream: Program that displays AI / feature detection only without Kalman tracking.
"""

from pathlib import Path
from typing import Optional, Tuple, Union, List
import cv2
import numpy as np

from .ai_detectors import YOLODetector, VisualFeatureTracker, DetectionResult
from .tracking_config import TrackingConfig


class CameraStream:
    """Manages video capture devices and video files with sensor warm-up and FPS tracking."""

    def __init__(
        self,
        source: Optional[Union[str, int]] = 0,
        camera_id: Optional[int] = None,
        warmup_frames: int = 15
    ):
        self.device_id = camera_id if camera_id is not None else source
        self.warmup_frames = warmup_frames
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self.fps = 30.0
        self.prev_tick = cv2.getTickCount()

    def open(self) -> bool:
        """Opens the camera or video stream and performs sensor exposure warm-up."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        # Handle numeric string or int camera device ID
        src = self.device_id
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.is_opened = False
            return False

        self.is_opened = True

        # Sensor auto-exposure calibration for live camera hardware
        if isinstance(src, int):
            for _ in range(self.warmup_frames):
                ret, warm_frame = self.cap.read()
                if ret and warm_frame is not None and float(np.mean(warm_frame)) > 15.0:
                    break

        self.prev_tick = cv2.getTickCount()
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads next frame and updates dynamic real-time FPS."""
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        curr_tick = cv2.getTickCount()
        dt = (curr_tick - self.prev_tick) / cv2.getTickFrequency()
        self.prev_tick = curr_tick
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        return True, frame

    def release(self) -> None:
        """Closes stream and frees video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class InteractiveMouseHandler:
    """Manages OpenCV mouse interactions including clicks, hover tracking, and drag ROI."""

    def __init__(self, allow_drag_roi: bool = True):
        self.allow_drag_roi = allow_drag_roi
        self.mouse_clicked = False
        self.click_xy: Tuple[int, int] = (0, 0)
        self.hover_xy: Optional[Tuple[int, int]] = None

        # Drag ROI state
        self.is_drawing_roi = False
        self.roi_start = (0, 0)
        self.roi_current = (0, 0)
        self.drawn_bbox: Optional[Tuple[int, int, int, int]] = None

    def on_mouse(self, event, x, y, flags, param):
        """OpenCV mouse callback handler."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.allow_drag_roi:
                self.is_drawing_roi = True
                self.roi_start = (x, y)
                self.roi_current = (x, y)
            else:
                self.is_drawing_roi = False
        elif event == cv2.EVENT_MOUSEMOVE:
            self.hover_xy = (x, y)
            if self.is_drawing_roi and self.allow_drag_roi:
                self.roi_current = (x, y)
            elif (flags & cv2.EVENT_FLAG_LBUTTON) and self.allow_drag_roi:
                if not self.is_drawing_roi:
                    self.is_drawing_roi = True
                    self.roi_start = (x, y)
                self.roi_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing_roi and self.allow_drag_roi:
                self.is_drawing_roi = False
                self.roi_current = (x, y)
                dx = abs(self.roi_current[0] - self.roi_start[0])
                dy = abs(self.roi_current[1] - self.roi_start[1])
                if dx >= 4 and dy >= 4:
                    bx = min(self.roi_start[0], self.roi_current[0])
                    by = min(self.roi_start[1], self.roi_current[1])
                    self.drawn_bbox = (bx, by, dx, dy)
                else:
                    self.mouse_clicked = True
                    self.click_xy = (x, y)
            else:
                self.is_drawing_roi = False
                self.mouse_clicked = True
                self.click_xy = (x, y)

    def attach_to_window(self, window_name: str) -> None:
        """Binds this handler to the named OpenCV window."""
        cv2.setMouseCallback(window_name, self.on_mouse)

    def pop_click(self) -> Optional[Tuple[int, int]]:
        """Consumes and returns latest mouse click coordinates if present."""
        if self.mouse_clicked:
            self.mouse_clicked = False
            return self.click_xy
        return None

    def pop_drawn_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Consumes and returns latest user-drawn bounding box ROI."""
        if self.drawn_bbox is not None:
            box = self.drawn_bbox
            self.drawn_bbox = None
            return box
        return None

    def get_active_drag_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Returns the rubber-band rectangle currently being dragged by user."""
        if self.is_drawing_roi and self.allow_drag_roi:
            rx = min(self.roi_start[0], self.roi_current[0])
            ry = min(self.roi_start[1], self.roi_current[1])
            rw = abs(self.roi_current[0] - self.roi_start[0])
            rh = abs(self.roi_current[1] - self.roi_start[1])
            if rw > 2 and rh > 2:
                return (rx, ry, rw, rh)
        return None


def show_detection_only_stream(
    config: Optional[TrackingConfig] = None,
    source: Optional[Union[str, int]] = 0,
    camera_id: Optional[int] = None,
    model_name: str = "yolov8n.pt",
    target_class: Optional[str] = None,
    interactive_gui: bool = True,
    max_frames: Optional[int] = None
) -> None:
    """Standalone Program: Displays Vision / AI Detections only (No Kalman Filter).

    Useful for verifying camera feeds, testing detector performance, and inspecting
    raw sensor observations before adding state estimation.
    """
    if config is None:
        config = TrackingConfig(
            source=source,
            camera_id=camera_id,
            model_name=model_name,
            target_class=target_class,
            interactive_gui=interactive_gui,
            max_live_frames=max_frames
        )

    detector = YOLODetector(model_name=config.model_name, dataset=config.dataset)
    window_name = "Detection Only Visualizer (YOLO & Vision)"
    if config.interactive_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cam_idx = config.camera_id if config.camera_id is not None else config.source
    stream = CameraStream(source=cam_idx)
    if not stream.open():
        print(f"[Detection Only] Warning: Could not open source {cam_idx}. Exiting.")
        return

    print(f"\n[Detection Only] Camera feed opened ({config.model_name})")
    print("  • Controls: Space = Freeze | 'q' / ESC = Quit")
    print(f"{'='*60}")

    frame_count = 0
    paused = False
    last_frame = None

    try:
        while True:
            if not paused or last_frame is None:
                ret, frame = stream.read()
                if not ret:
                    break
                last_frame = frame
            else:
                frame = last_frame

            vis = frame.copy()
            h, w = vis.shape[:2]

            # Detect objects
            detections = detector.detect_all(frame, target_class=config.target_class)

            # Draw detections
            for idx, det in enumerate(detections):
                bx, by, bw, bh = det.bbox
                cx, cy = det.center_of_mass
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 220, 255), 2)
                cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                label_text = f"[{idx+1}] {det.label} ({det.confidence:.2f})"
                cv2.putText(vis, label_text, (bx, max(15, by - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

            # Draw HUD
            hud_text = f"DETECTION ONLY | FPS: {stream.fps:.1f} | Objects: {len(detections)}"
            cv2.putText(vis, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if config.interactive_gui:
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(20 if paused else 1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord(' '):
                    paused = not paused

            if not paused:
                frame_count += 1
                if config.max_live_frames is not None and frame_count >= config.max_live_frames:
                    break
    finally:
        stream.release()
        if config.interactive_gui:
            cv2.destroyAllWindows()
