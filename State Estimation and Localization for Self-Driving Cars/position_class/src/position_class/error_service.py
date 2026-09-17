"""Error Publishing and Logging Service for Visual Object Tracking.

Publishes error signals representing the distance and offset of the tracked object
with respect to the center of the image.
"""

from dataclasses import dataclass, asdict
import csv
import json
import math
from pathlib import Path
from queue import Queue
import time
from typing import Callable, Dict, List, Optional, Union
import numpy as np


@dataclass
class TrackingErrorSignal:
    """Error signal message representing deviation from image center."""
    frame_index: int
    timestamp: float
    target_x: float
    target_y: float
    image_center_x: float
    image_center_y: float
    error_x: float          # target_x - image_center_x (pixels)
    error_y: float          # target_y - image_center_y (pixels)
    error_distance: float   # Euclidean distance in pixels
    is_tracked: bool = True
    predicted_x: Optional[float] = None
    predicted_y: Optional[float] = None

    @classmethod
    def compute(
        cls,
        frame_index: int,
        target_pos: tuple[float, float],
        image_shape: tuple[int, int],
        is_tracked: bool = True,
        predicted_pos: Optional[tuple[float, float]] = None,
        timestamp: Optional[float] = None
    ) -> "TrackingErrorSignal":
        """Factory method to calculate error signal from target coordinates and image dimensions."""
        h, w = image_shape[:2]
        cx_img = w / 2.0
        cy_img = h / 2.0
        tx, ty = target_pos
        ex = tx - cx_img
        ey = ty - cy_img
        dist = math.hypot(ex, ey)
        
        pred_x = predicted_pos[0] if predicted_pos is not None else None
        pred_y = predicted_pos[1] if predicted_pos is not None else None

        return cls(
            frame_index=frame_index,
            timestamp=timestamp if timestamp is not None else time.time(),
            target_x=float(tx),
            target_y=float(ty),
            image_center_x=float(cx_img),
            image_center_y=float(cy_img),
            error_x=float(ex),
            error_y=float(ey),
            error_distance=float(dist),
            is_tracked=is_tracked,
            predicted_x=pred_x,
            predicted_y=pred_y
        )


ErrorCallback = Callable[[TrackingErrorSignal], None]


class ErrorPublisherService:
    """Service that publishes tracking error signals to registered listeners, queues, and file sinks."""

    def __init__(self, service_name: str = "VisualTrackingErrorService"):
        self.service_name = service_name
        self._subscribers: List[ErrorCallback] = []
        self._queues: List[Queue] = []
        self._history: List[TrackingErrorSignal] = []
        self._csv_writer: Optional[csv.writer] = None
        self._csv_file = None

    def subscribe(self, callback: ErrorCallback) -> None:
        """Register a subscriber callback function."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: ErrorCallback) -> None:
        """Remove a registered subscriber callback."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def create_queue_subscriber(self, maxsize: int = 100) -> Queue:
        """Create a thread-safe Queue subscriber for consumer workers or ROS bridges."""
        q = Queue(maxsize=maxsize)
        self._queues.append(q)
        return q

    def enable_csv_logging(self, output_path: Union[str, Path]) -> None:
        """Enable automatic continuous logging to a CSV file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(p, "w", newline="", encoding="utf-8")
        fieldnames = [
            "frame_index", "timestamp", "target_x", "target_y",
            "image_center_x", "image_center_y", "error_x", "error_y",
            "error_distance", "is_tracked", "predicted_x", "predicted_y"
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

    def publish(self, signal: TrackingErrorSignal) -> None:
        """Publish an error signal to all active subscribers, queues, and loggers."""
        self._history.append(signal)

        # 1. Notify callback listeners
        for cb in self._subscribers:
            try:
                cb(signal)
            except Exception as e:
                print(f"[{self.service_name}] Callback error: {e}")

        # 2. Push to queues
        for q in self._queues:
            if not q.full():
                q.put_nowait(signal)

        # 3. Write to CSV sink if enabled
        if self._csv_writer is not None:
            self._csv_writer.writerow(asdict(signal))
            self._csv_file.flush()

    def get_history(self) -> List[TrackingErrorSignal]:
        """Return full published history."""
        return list(self._history)

    def get_latest(self) -> Optional[TrackingErrorSignal]:
        """Return the most recently published error signal."""
        return self._history[-1] if self._history else None

    def compute_metrics(self) -> Dict[str, float]:
        """Compute statistical error metrics (MAE, RMSE, Max Error) over the session."""
        if not self._history:
            return {"mean_error": 0.0, "rmse": 0.0, "max_error": 0.0, "std_error": 0.0}

        distances = np.array([s.error_distance for s in self._history])
        return {
            "mean_error": float(np.mean(distances)),
            "rmse": float(np.sqrt(np.mean(distances**2))),
            "max_error": float(np.max(distances)),
            "std_error": float(np.std(distances)),
            "total_samples": len(distances)
        }

    def close(self) -> None:
        """Close loggers and clean up resources."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
