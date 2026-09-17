"""AI and Computer Vision Detection Module for Interactive Visual Tracking.

Provides object detectors (YOLO / OpenCV FAST & Contour Segmentation) and an
interactive click-to-track manager that handles the initial idle state,
user click target selection, segmentation rectangle, and Center of Mass extraction.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Object detection result with bounding box, segmentation polygon, Center of Mass, and FAST features."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center_of_mass: Tuple[float, float]  # (cx, cy)
    confidence: float = 1.0
    label: str = "target"
    polygon: Optional[List[Tuple[float, float]]] = None
    fast_points: Optional[List[Tuple[float, float]]] = None  # Coordinates of the 3 tracked FAST features



class BaseDetector:
    """Base interface for vision / AI object detectors."""

    def detect_all(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect all objects in frame."""
        raise NotImplementedError

    def track_target(
        self,
        frame: np.ndarray,
        expected_pos: Optional[Tuple[float, float]] = None,
        search_radius: int = 150
    ) -> Optional[DetectionResult]:
        """Detect or segment target near an expected position (from Kalman prediction)."""
        raise NotImplementedError


class YOLODetector(BaseDetector):
    """YOLO AI Detector with optional ultralytics backend and OpenCV DNN / color segmentation fallback."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        dataset: Optional[str] = None,
        conf_threshold: float = 0.3
    ):
        # Normalize model name: append .pt if missing extension
        if not any(model_name.endswith(ext) for ext in (".pt", ".onnx", ".engine", ".tflite", ".yaml")):
            self.model_name = f"{model_name}.pt"
        else:
            self.model_name = model_name
        self.dataset = dataset
        self.conf_threshold = conf_threshold
        self.yolo_model = None
        self.backend_name = "OpenCV Deep Vision Detector (Salient Contour + FAST)"
        self._init_backend()

    def _init_backend(self) -> None:
        """Attempt to load ultralytics YOLO; if unavailable, log fallback mode."""
        try:
            from ultralytics import YOLO  # type: ignore
            self.yolo_model = YOLO(self.model_name)
            ds_str = f" - {self.dataset.upper()} Dataset" if self.dataset else " - COCO 80 classes"
            self.backend_name = f"YOLO ({self.model_name}{ds_str})"
        except Exception:
            self.yolo_model = None
            ds_str = f" [{self.dataset.upper()} Dataset]" if self.dataset else ""
            self.backend_name = f"OpenCV AI Vision Detector{ds_str} (FAST + Morphological Contour)"

    def get_model_info(self) -> str:
        """Return human-readable description of the active AI model."""
        return self.backend_name

    def detect_all(self, frame: np.ndarray, target_class: Optional[str] = None) -> List[DetectionResult]:
        """Detect objects using YOLO or OpenCV edge/contour feature detector.
        
        Args:
            frame: Input BGR image.
            target_class: Optional class filter (e.g. 'person', 'car', 'cup', 'cell phone').
                          If provided, only objects matching this class name will be returned.
        """
        # Auto-apply KITTI benchmark classes if dataset is KITTI and no filter specified
        if target_class is None and self.dataset and self.dataset.lower() == "kitti":
            target_filter = ["car", "van", "truck", "pedestrian", "cyclist", "tram"]
        else:
            target_filter = [c.strip().lower() for c in target_class.split(",")] if (target_class and target_class.lower() != "all") else None

        if self.yolo_model is not None:
            try:
                results = self.yolo_model(frame, verbose=False, conf=self.conf_threshold)
                detections = []
                for r in results:
                    # 1. Bounding Boxes
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            w = max(1, x2 - x1)
                            h = max(1, y2 - y1)
                            conf = float(box.conf[0].cpu().numpy())
                            cls_idx = int(box.cls[0].cpu().numpy())
                            raw_label = r.names.get(cls_idx, "object")
                            
                            # Apply target_class filter if specified
                            if target_filter is not None:
                                if not any(f in raw_label.lower() for f in target_filter):
                                    continue

                            label = f"{raw_label.capitalize()} ({conf:.0%})"
                            cx = x1 + w / 2.0
                            cy = y1 + h / 2.0
                            poly = [(float(x1), float(y1)), (float(x2), float(y1)),
                                    (float(x2), float(y2)), (float(x1), float(y2))]
                            detections.append(DetectionResult(
                                bbox=(x1, y1, w, h),
                                center_of_mass=(cx, cy),
                                confidence=conf,
                                label=label,
                                polygon=poly
                            ))

                    # 2. Keypoints (Pose models, e.g. yolov8n-pose.pt)
                    # COCO Keypoint layout: 0=Nose, 1=Left Eye, 2=Right Eye, 3=Left Ear, 4=Right Ear
                    if hasattr(r, 'keypoints') and r.keypoints is not None:
                        kpts = r.keypoints.xy.cpu().numpy()
                        k_confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
                        for p_idx, person_kpts in enumerate(kpts):
                            # Left Eye (idx 1)
                            if len(person_kpts) > 1 and person_kpts[1][0] > 0 and person_kpts[1][1] > 0:
                                lx, ly = float(person_kpts[1][0]), float(person_kpts[1][1])
                                lconf = float(k_confs[p_idx][1]) if k_confs is not None else 0.85
                                if target_filter is None or any(f in "left eye" for f in target_filter) or any(f in "eye" for f in target_filter):
                                    ew, eh = 36, 26
                                    ebx = max(0, int(lx - ew / 2.0))
                                    eby = max(0, int(ly - eh / 2.0))
                                    detections.append(DetectionResult(
                                        bbox=(ebx, eby, ew, eh),
                                        center_of_mass=(lx, ly),
                                        confidence=lconf,
                                        label=f"Left Eye ({lconf:.0%})"
                                    ))
                            # Right Eye (idx 2)
                            if len(person_kpts) > 2 and person_kpts[2][0] > 0 and person_kpts[2][1] > 0:
                                rx, ry = float(person_kpts[2][0]), float(person_kpts[2][1])
                                rconf = float(k_confs[p_idx][2]) if k_confs is not None else 0.85
                                if target_filter is None or any(f in "right eye" for f in target_filter) or any(f in "eye" for f in target_filter):
                                    ew, eh = 36, 26
                                    ebx = max(0, int(rx - ew / 2.0))
                                    eby = max(0, int(ry - eh / 2.0))
                                    detections.append(DetectionResult(
                                        bbox=(ebx, eby, ew, eh),
                                        center_of_mass=(rx, ry),
                                        confidence=rconf,
                                        label=f"Right Eye ({rconf:.0%})"
                                    ))

                if len(detections) > 0:
                    return detections
            except Exception:
                pass

        # OpenCV Fallback: Detect salient moving or colored object blobs / contours
        all_blobs = self._detect_salient_blobs(frame, target_filter=target_filter)
        if target_filter is not None:
            return [b for b in all_blobs if any(f in b.label.lower() for f in target_filter)]
        return all_blobs


    def _detect_salient_blobs(
        self,
        frame: np.ndarray,
        target_filter: Optional[List[str]] = None
    ) -> List[DetectionResult]:
        """Fallback salient visual contour and eye/pupil detector with heuristics."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 120)
        
        # 1. Check for eye / iris circular features if eye filter is requested
        detections: List[DetectionResult] = []
        if target_filter is not None and any("eye" in f for f in target_filter):
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
                param1=50, param2=28, minRadius=6, maxRadius=40
            )
            if circles is not None:
                for c in circles[0, :4]:
                    cx, cy, r = float(c[0]), float(c[1]), float(c[2])
                    bx = max(0, int(cx - r))
                    by = max(0, int(cy - r))
                    bw = int(r * 2)
                    bh = int(r * 2)
                    detections.append(DetectionResult(
                        bbox=(bx, by, bw, bh),
                        center_of_mass=(cx, cy),
                        confidence=0.88,
                        label="Eye / Pupil",
                        polygon=[(float(bx), float(by)), (float(bx + bw), float(by)),
                                (float(bx + bw), float(by + bh)), (float(bx), float(by + bh))]
                    ))

        # 2. General Contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > (frame.shape[0] * frame.shape[1] * 0.85):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(1.0, float(h))
            
            # Object naming logic
            if target_filter is not None and len(target_filter) > 0:
                lbl = target_filter[0].capitalize()
            elif aspect_ratio >= 1.5:
                lbl = "Wide Object"
            elif aspect_ratio <= 0.65:
                lbl = "Vertical Object / Person"
            elif 0.85 <= aspect_ratio <= 1.15:
                lbl = "Compact Target"
            else:
                lbl = "Visual Object"

            M = cv2.moments(cnt)
            if abs(M["m00"]) > 1e-4:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx = float(x + w / 2.0)
                cy = float(y + h / 2.0)
            
            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            poly = [(float(p[0][0]), float(p[0][1])) for p in approx] if len(approx) >= 3 else None

            detections.append(DetectionResult(
                bbox=(x, y, w, h),
                center_of_mass=(cx, cy),
                confidence=0.90,
                label=lbl,
                polygon=poly
            ))
        return detections


class FastFeatureTracker:
    """Combines FAST Corner Feature Detection + ORB matching with dynamic Center of Mass calculation.
    
    Maintains and dynamically refreshes a set of the 3 most prominent FAST feature points
    inside the tracked target bounding box.
    """

    def __init__(self, fast_threshold: int = 15):
        self.fast_threshold = fast_threshold
        self.fast = cv2.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=True)
        self.orb = cv2.ORB_create(nfeatures=400, edgeThreshold=5, patchSize=15)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.target_descriptors = None
        self.target_template: Optional[np.ndarray] = None
        self.template_size: Tuple[int, int] = (60, 60)
        self.target_label: str = "Target"
        self.fast_points: List[Tuple[float, float]] = []  # Current coordinates of the 3 FAST features
        self.is_initialized: bool = False

    def _extract_top_k_fast_points(
        self,
        roi_gray: np.ndarray,
        origin_x: int,
        origin_y: int,
        k: int = 3
    ) -> List[Tuple[float, float]]:
        """Extract the top-k most prominent FAST corner features within the ROI."""
        rh, rw = roi_gray.shape[:2]
        if rh < 4 or rw < 4:
            return []

        # 1. Detect FAST keypoints
        kp = self.fast.detect(roi_gray, None)
        
        # 2. Fallback to GoodFeaturesToTrack / ORB if fewer than k FAST corners found
        if len(kp) < k:
            corners = cv2.goodFeaturesToTrack(roi_gray, maxCorners=k * 3, qualityLevel=0.01, minDistance=4)
            if corners is not None and len(corners) > 0:
                pts = [(float(origin_x + c[0][0]), float(origin_y + c[0][1])) for c in corners[:k]]
                return pts

        if len(kp) == 0:
            # Synthetic 3-point grid fallback for low-texture patches
            return [
                (float(origin_x + rw * 0.25), float(origin_y + rh * 0.25)),
                (float(origin_x + rw * 0.75), float(origin_y + rh * 0.25)),
                (float(origin_x + rw * 0.50), float(origin_y + rh * 0.75)),
            ]

        # 3. Sort keypoints by corner response score descending
        sorted_kp = sorted(kp, key=lambda p: float(p.response), reverse=True)
        top_k = sorted_kp[:k]

        # Convert to global frame coordinates
        pts = [(float(origin_x + p.pt[0]), float(origin_y + p.pt[1])) for p in top_k]
        
        # If we had 1 or 2 points, pad with remaining positions
        while len(pts) < k:
            if len(pts) == 1:
                pts.append((float(origin_x + rw * 0.75), float(origin_y + rh * 0.50)))
            elif len(pts) == 2:
                pts.append((float(origin_x + rw * 0.50), float(origin_y + rh * 0.75)))

        return pts[:k]

    def refresh_features(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        alpha: float = 0.20
    ) -> List[Tuple[float, float]]:
        """Dynamically refresh the target template and 3 prominent FAST feature points.
        
        Uses an Exponential Moving Average (EMA) for the appearance template to adapt
        to illumination and perspective changes, and re-extracts the 3 sharpest FAST corners.
        """
        bx, by, bw, bh = bbox
        fh, fw = frame.shape[:2]
        x1 = max(0, min(fw - 1, bx))
        y1 = max(0, min(fh - 1, by))
        x2 = max(x1 + 1, min(fw, x1 + bw))
        y2 = max(y1 + 1, min(fh, y1 + bh))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        curr_roi = gray[y1:y2, x1:x2]

        if curr_roi.shape[0] > 4 and curr_roi.shape[1] > 4:
            # 1. Update template using EMA if dimensions match
            if self.target_template is not None and self.target_template.shape == curr_roi.shape:
                self.target_template = cv2.addWeighted(
                    curr_roi, alpha,
                    self.target_template, 1.0 - alpha,
                    0.0
                )
            else:
                self.target_template = curr_roi.copy()
                self.template_size = (x2 - x1, y2 - y1)

            # 2. Refresh the 3 FAST feature points
            self.fast_points = self._extract_top_k_fast_points(curr_roi, x1, y1, k=3)

        return self.fast_points

    def init_target(
        self,
        frame: np.ndarray,
        click_pos: Tuple[int, int],
        window_size: int = 60,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        label: str = "Target"
    ) -> DetectionResult:
        """Initialize tracking with exact bounding box or centered window to ensure coordinate plane alignment."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()

        if bbox is not None:
            bx, by, bw, bh = bbox
            x1 = max(0, min(w - 1, bx))
            y1 = max(0, min(h - 1, by))
            x2 = max(x1 + 1, min(w, x1 + bw))
            y2 = max(y1 + 1, min(h, y1 + bh))
            bw = x2 - x1
            bh = y2 - y1
            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0
        else:
            cx_in, cy_in = click_pos
            x1 = max(0, cx_in - window_size // 2)
            y1 = max(0, cy_in - window_size // 2)
            x2 = min(w, x1 + window_size)
            y2 = min(h, y1 + window_size)
            bw = x2 - x1
            bh = y2 - y1
            cx = float(cx_in)
            cy = float(cy_in)

        roi = gray[y1:y2, x1:x2]

        kp = self.fast.detect(roi, None)
        if len(kp) == 0:
            kp = self.orb.detect(roi, None)
        
        kp, desc = self.orb.compute(roi, kp)
        if desc is None or len(desc) == 0:
            kp, desc = self.orb.detectAndCompute(roi, None)

        self.target_descriptors = desc
        self.target_template = roi.copy()
        self.template_size = (bw, bh)
        self.target_label = label
        self.is_initialized = True

        # Extract initial set of 3 FAST feature points
        self.fast_points = self._extract_top_k_fast_points(roi, x1, y1, k=3)

        poly = [(float(x1), float(y1)), (float(x1 + bw), float(y1)),
                (float(x1 + bw), float(y1 + bh)), (float(x1), float(y1 + bh))]
        return DetectionResult(
            bbox=(x1, y1, bw, bh),
            center_of_mass=(cx, cy),
            confidence=1.0,
            label=self.target_label,
            polygon=poly,
            fast_points=list(self.fast_points)
        )

    def track(
        self,
        frame: np.ndarray,
        expected_pos: Optional[Tuple[float, float]] = None,
        min_confidence: float = 0.48,
        search_margin: int = 80,
        auto_refresh: bool = True,
        enable_full_frame_search: bool = True
    ) -> Optional[DetectionResult]:
        """Track initialized target in new frame with local gating and global full-frame recovery.
        
        1. Localized Search: First searches within the Kalman spatial gate around expected_pos.
        2. Full-Frame Search: If local search fails or target moved across the screen during occlusion,
           scans the entire frame to re-acquire the object anywhere in the image.
        """
        if not self.is_initialized or self.target_template is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        fh, fw = gray.shape[:2]
        tw, th = self.template_size

        if fh < th or fw < tw:
            return None

        # 1. Localized Search Window centered on the Kalman expected position
        if expected_pos is not None:
            epx, epy = float(expected_pos[0]), float(expected_pos[1])
            expected_tl_x = epx - tw / 2.0
            expected_tl_y = epy - th / 2.0
            
            sx1 = max(0, int(expected_tl_x - search_margin))
            sy1 = max(0, int(expected_tl_y - search_margin))
            sx2 = min(fw, int(expected_tl_x + tw + search_margin))
            sy2 = min(fh, int(expected_tl_y + th + search_margin))
            search_roi = gray[sy1:sy2, sx1:sx2]
            
            if search_roi.shape[0] >= th and search_roi.shape[1] >= tw:
                res = cv2.matchTemplate(search_roi, self.target_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                # Check for genuine visual confirmation in local window
                if max_val >= min_confidence:
                    best_x = sx1 + max_loc[0]
                    best_y = sy1 + max_loc[1]
                    cx = best_x + tw / 2.0
                    cy = best_y + th / 2.0
                    poly = [(float(best_x), float(best_y)), (float(best_x + tw), float(best_y)),
                            (float(best_x + tw), float(best_y + th)), (float(best_x), float(best_y + th))]
                    
                    bbox = (best_x, best_y, tw, th)

                    # Dynamically refresh the 3 FAST features on high-confidence frames
                    if auto_refresh and max_val >= 0.65:
                        self.refresh_features(frame, bbox, alpha=0.20)
                    else:
                        if len(self.fast_points) == 3:
                            shift_x = best_x - (cx - tw / 2.0)
                            shift_y = best_y - (cy - th / 2.0)
                            self.fast_points = [(p[0] + shift_x, p[1] + shift_y) for p in self.fast_points]
                        else:
                            self.fast_points = self._extract_top_k_fast_points(
                                gray[best_y:best_y+th, best_x:best_x+tw], best_x, best_y, k=3
                            )

                    return DetectionResult(
                        bbox=bbox,
                        center_of_mass=(cx, cy),
                        confidence=float(max_val),
                        label=self.target_label,
                        polygon=poly,
                        fast_points=list(self.fast_points)
                    )

        # 2. Full-Frame Global Search Fallback (Scans entire picture to recover lost/jumped object)
        if enable_full_frame_search:
            res_full = cv2.matchTemplate(gray, self.target_template, cv2.TM_CCOEFF_NORMED)
            _, g_max_val, _, g_max_loc = cv2.minMaxLoc(res_full)
            
            # Global re-acquisition threshold
            if g_max_val >= max(0.52, min_confidence):
                best_x, best_y = g_max_loc
                cx = best_x + tw / 2.0
                cy = best_y + th / 2.0
                poly = [(float(best_x), float(best_y)), (float(best_x + tw), float(best_y)),
                        (float(best_x + tw), float(best_y + th)), (float(best_x), float(best_y + th))]
                bbox = (best_x, best_y, tw, th)

                # Re-extract the 3 FAST points at the newly discovered location
                self.fast_points = self._extract_top_k_fast_points(
                    gray[best_y:best_y+th, best_x:best_x+tw], best_x, best_y, k=3
                )
                if auto_refresh and g_max_val >= 0.65:
                    self.refresh_features(frame, bbox, alpha=0.25)

                return DetectionResult(
                    bbox=bbox,
                    center_of_mass=(cx, cy),
                    confidence=float(g_max_val),
                    label=self.target_label,
                    polygon=poly,
                    fast_points=list(self.fast_points)
                )

        return None


class HybridTracker:
    """Hybrid Object Tracker synergizing 3-Point FAST Feature Tracking + Full-Frame YOLO Re-acquisition.
    
    Architecture:
    1. Primary Track: High-frequency FAST 3-point corner tracking with dynamic appearance refresh.
    2. Covariance-Adaptive Search Gating: Search radius dynamically expands with Kalman uncertainty.
    3. Full-Frame Scan & YOLO Semantic Recovery: When FAST confidence drops or occlusion occurs,
       scans the entire frame (no artificial radius cutoff) to re-acquire the target anywhere in the picture!
    """

    def __init__(
        self,
        yolo_detector: Optional[YOLODetector] = None,
        fast_threshold: int = 15,
        yolo_model_name: str = "yolov8n.pt",
        dataset: Optional[str] = None
    ):
        self.fast_tracker = FastFeatureTracker(fast_threshold=fast_threshold)
        self.yolo_detector = yolo_detector if yolo_detector is not None else YOLODetector(model_name=yolo_model_name, dataset=dataset)
        self.target_label: str = "Target"
        self.target_class: Optional[str] = None
        self.is_initialized: bool = False
        self.last_track_mode: str = "IDLE"

    def init_target(
        self,
        frame: np.ndarray,
        click_pos: Tuple[int, int],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        label: str = "Target",
        target_class: Optional[str] = None
    ) -> DetectionResult:
        """Initialize hybrid tracking with bounding box and 3 refreshed FAST keypoints."""
        self.target_label = label
        self.target_class = target_class
        det = self.fast_tracker.init_target(
            frame=frame,
            click_pos=click_pos,
            bbox=bbox,
            label=label
        )
        self.is_initialized = True
        self.last_track_mode = "HYBRID_INIT"
        return det

    def track(
        self,
        frame: np.ndarray,
        expected_pos: Tuple[float, float],
        kalman_uncertainty_std: float = 20.0,
        target_class: Optional[str] = None
    ) -> Tuple[Optional[DetectionResult], str]:
        """Execute hybrid tracking step with full-frame recovery.
        
        Returns:
            Tuple of (DetectionResult or None, track_mode_string)
            track_mode_string: 'HYBRID_FAST' | 'HYBRID_FAST_GLOBAL' | 'HYBRID_REACQUIRED_YOLO' | 'DEAD_RECKONING'
        """
        if not self.is_initialized:
            return None, "UNINITIALIZED"

        active_class = target_class if target_class is not None else self.target_class
        pred_x, pred_y = expected_pos

        # 1. Covariance-Adaptive Search Gate radius: expands when uncertainty is high
        adaptive_margin = int(max(70, min(240, 3.0 * kalman_uncertainty_std + 50)))

        # 2. Stage A: Try FAST 3-Point Localized Feature Tracking
        fast_det = self.fast_tracker.track(
            frame=frame,
            expected_pos=expected_pos,
            min_confidence=0.48,
            search_margin=adaptive_margin,
            auto_refresh=True,
            enable_full_frame_search=False
        )

        if fast_det is not None and fast_det.confidence >= 0.52:
            self.last_track_mode = "HYBRID_FAST"
            return fast_det, "HYBRID_FAST"

        # 3. Stage B: Full-Frame YOLO Semantic Re-Acquisition (Scans whole picture!)
        reacq_candidates = self.yolo_detector.detect_all(frame, target_class=active_class)
        best_cand: Optional[DetectionResult] = None
        min_dist = float("inf")

        # Scan ALL candidates across the whole frame, prioritizing the one closest to last known trajectory
        for c in reacq_candidates:
            ccx, ccy = c.center_of_mass
            dist = float(np.hypot(ccx - pred_x, ccy - pred_y))
            if dist < min_dist:
                min_dist = dist
                best_cand = c

        if best_cand is not None:
            # Re-seed FAST features and dynamic template at new YOLO location anywhere in frame
            re_det = self.fast_tracker.init_target(
                frame=frame,
                click_pos=(int(best_cand.center_of_mass[0]), int(best_cand.center_of_mass[1])),
                bbox=best_cand.bbox,
                label=best_cand.label
            )
            re_det.confidence = best_cand.confidence
            self.target_label = best_cand.label
            self.last_track_mode = "HYBRID_REACQUIRED_YOLO"
            return re_det, "HYBRID_REACQUIRED_YOLO"

        # 4. Stage C: Full-Frame FAST Template Search (Scans whole picture if YOLO didn't trigger)
        fast_global = self.fast_tracker.track(
            frame=frame,
            expected_pos=None,
            min_confidence=0.52,
            auto_refresh=True,
            enable_full_frame_search=True
        )
        if fast_global is not None:
            self.last_track_mode = "HYBRID_FAST_GLOBAL"
            return fast_global, "HYBRID_FAST_GLOBAL"

        # 5. Stage D: Neither detector succeeded -> Pure Kalman Dead Reckoning
        self.last_track_mode = "DEAD_RECKONING"
        return None, "DEAD_RECKONING"


