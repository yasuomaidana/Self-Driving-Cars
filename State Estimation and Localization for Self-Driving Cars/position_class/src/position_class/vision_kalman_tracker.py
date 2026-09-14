"""Vision-Based Motion Prediction and Tracking with FAST Feature Extraction and Kalman Filter.

Based on the FAST Algorithm + Kalman Filter motion prediction architecture:
1. Extract FAST corner features and descriptors from target Region of Interest (ROI).
2. For each video frame, extract features and match with target descriptors.
3. Compute matched centroid (x_meas, y_meas).
4. Run Kalman Filter prediction [x_pred, y_pred, vx_pred, vy_pred] and correction.
5. Provide synthetic frame generation and Plotly trajectory visualization.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class FastKalmanVisualTracker:
    """Combines FAST Corner Feature Matching with a 2D Kinematic Kalman Filter for Visual Object Tracking."""

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_std: float = 0.5,
        measurement_noise_std: float = 2.0,
        fast_threshold: int = 10
    ):
        """Initialize FAST detector, descriptor extractor, and Kalman Filter."""
        self.dt = dt
        
        # 1. Initialize FAST Feature Detector & ORB/BRIEF Descriptor Extractor
        self.fast = cv2.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=True)
        self.orb = cv2.ORB_create(nfeatures=500, edgeThreshold=5, patchSize=15)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 2. Initialize 4D Linear Kalman Filter (State: [x, y, vx, vy]^T, Measurement: [x, y]^T)
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        q_pos = (dt**3) / 3.0 * (process_noise_std**2)
        q_vel = dt * (process_noise_std**2)
        q_pos_vel = (dt**2) / 2.0 * (process_noise_std**2)

        self.Q = np.array([
            [q_pos,     0.0,       q_pos_vel, 0.0      ],
            [0.0,       q_pos,     0.0,       q_pos_vel],
            [q_pos_vel, 0.0,       q_vel,     0.0      ],
            [0.0,       q_pos_vel, 0.0,       q_vel    ]
        ], dtype=np.float32)

        self.R = np.eye(2, dtype=np.float32) * (measurement_noise_std**2)
        self.P = np.eye(4, dtype=np.float32) * 100.0
        self.x = np.zeros((4, 1), dtype=np.float32)

        # Template storage
        self.target_descriptors: Optional[np.ndarray] = None
        self.target_keypoints: Optional[List[cv2.KeyPoint]] = None
        self.target_offset_x = 0.0
        self.target_offset_y = 0.0
        self.initialized = False

    def initialize_target_from_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> int:
        """Initialize the target model from a bounding box (x, y, w, h) in the initial frame."""
        x, y, w, h = bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        
        # Crop Region of Interest (ROI)
        roi = gray[y:y+h, x:x+w]
        
        # Detect FAST keypoints inside ROI
        keypoints = self.fast.detect(roi, None)
        if len(keypoints) == 0:
            keypoints = self.orb.detect(roi, None)

        # Compute descriptors
        keypoints, descriptors = self.orb.compute(roi, keypoints)
        
        if descriptors is None or len(descriptors) == 0:
            # Fallback: force detectAndCompute directly with ORB
            keypoints, descriptors = self.orb.detectAndCompute(roi, None)

        if descriptors is None or len(descriptors) == 0:
            raise ValueError("No distinct visual features found in the specified target bounding box.")

        self.target_keypoints = keypoints
        self.target_descriptors = descriptors
        self.target_offset_x = float(x)
        self.target_offset_y = float(y)
        
        # Initialize Kalman state at the center of the bounding box
        center_x = float(x + w / 2.0)
        center_y = float(y + h / 2.0)
        self.x = np.array([[center_x], [center_y], [0.0], [0.0]], dtype=np.float32)
        self.initialized = True
        
        return len(keypoints)

    def predict(self) -> Tuple[float, float, float, float]:
        """Execute Kalman prediction step: returns (pred_x, pred_y, pred_vx, pred_vy)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])

    def update_from_detection(self, measured_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Correct the Kalman state with an observed centroid (meas_x, meas_y)."""
        z = np.array([[measured_pos[0]], [measured_pos[1]]], dtype=np.float32)
        nu = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ nu
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
        
        return float(self.x[0, 0]), float(self.x[1, 0])

    def detect_target_in_frame(self, frame: np.ndarray) -> Optional[Tuple[float, float, int]]:
        """Detect target object in the current frame by matching FAST/ORB descriptors."""
        if not self.initialized or self.target_descriptors is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        
        # Detect keypoints in current frame
        frame_kp = self.fast.detect(gray, None)
        if len(frame_kp) == 0:
            frame_kp = self.orb.detect(gray, None)

        frame_kp, frame_desc = self.orb.compute(gray, frame_kp)
        
        if frame_desc is None or len(frame_desc) == 0:
            return None

        # Match descriptors with target template
        matches = self.matcher.match(self.target_descriptors, frame_desc)
        if len(matches) == 0:
            return None

        # Sort matches by descriptor distance
        matches = sorted(matches, key=lambda m: m.distance)
        good_matches = matches[:min(len(matches), 30)]

        # Compute centroid of matched keypoints
        pts = np.array([frame_kp[m.trainIdx].pt for m in good_matches])
        avg_x = float(np.mean(pts[:, 0]))
        avg_y = float(np.mean(pts[:, 1]))
        
        return avg_x, avg_y, len(good_matches)

    def process_frame(self, frame: np.ndarray) -> Dict[str, Optional[float]]:
        """Run full prediction + detection + correction cycle on a single frame."""
        # 1. Kalman Predict
        pred_x, pred_y, pred_vx, pred_vy = self.predict()

        # 2. Feature Detection & Matching
        detection = self.detect_target_in_frame(frame)
        
        meas_x, meas_y, num_matches = None, None, 0
        if detection is not None:
            meas_x, meas_y, num_matches = detection
            # 3. Kalman Measurement Update
            est_x, est_y = self.update_from_detection((meas_x, meas_y))
        else:
            # Maintain predicted state if occluded or lost
            est_x, est_y = pred_x, pred_y

        return {
            "pred_x": pred_x,
            "pred_y": pred_y,
            "pred_vx": pred_vx,
            "pred_vy": pred_vy,
            "meas_x": meas_x,
            "meas_y": meas_y,
            "est_x": est_x,
            "est_y": est_y,
            "matches": num_matches
        }


def generate_synthetic_tracking_video(
    num_frames: int = 100,
    width: int = 640,
    height: int = 480
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Generates a synthetic video sequence of a textured target moving across the screen."""
    frames = []
    ground_truth = []

    # Create distinct textured target patch (60x60 with high contrast checkerboard corners)
    target_size = 60
    target_patch = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    block_size = 10
    for r in range(0, target_size, block_size):
        for c in range(0, target_size, block_size):
            if (r // block_size + c // block_size) % 2 == 0:
                target_patch[r:r+block_size, c:c+block_size] = [240, 240, 240]
            else:
                target_patch[r:r+block_size, c:c+block_size] = [20, 20, 20]
    cv2.rectangle(target_patch, (0, 0), (target_size-1, target_size-1), (0, 255, 0), 2)

    # Parametric motion trajectory (curved path across canvas)
    t_vals = np.linspace(0, 4 * np.pi, num_frames)
    for t in t_vals:
        cx = int(100 + (width - 200) * (t / (4 * np.pi)))
        cy = int(height / 2 + 100 * np.sin(t))
        ground_truth.append((float(cx), float(cy)))

        # Background frame with subtle noise
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        
        # Overlay moving target patch
        x1 = max(0, cx - target_size // 2)
        y1 = max(0, cy - target_size // 2)
        x2 = min(width, x1 + target_size)
        y2 = min(height, y1 + target_size)
        
        pw = x2 - x1
        ph = y2 - y1
        if pw > 0 and ph > 0:
            frame[y1:y2, x1:x2] = target_patch[:ph, :pw]

        frames.append(frame)

    return frames, ground_truth
