"""Input and Dataset IO Module for Visual Tracking Exercises.

Handles reading JSON datasets containing picture paths with bounding boxes
or enclosing polygon points, calculating object centroids/centers of mass,
and generating synthetic datasets for student exercises.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class TrackedAnnotation:
    """Represents a visual object annotation in a single frame."""
    frame_index: int
    image_path: str
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    polygon: Optional[List[Tuple[float, float]]] = None  # [(x1, y1), (x2, y2), ...]
    label: str = "object"
    center_of_mass: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        if self.center_of_mass is None:
            self.center_of_mass = self.calculate_center_of_mass()

    def calculate_center_of_mass(self) -> Optional[Tuple[float, float]]:
        """Calculate center of mass from bounding box or polygon."""
        if self.polygon is not None and len(self.polygon) >= 3:
            pts = np.array(self.polygon, dtype=np.float32)
            # Use OpenCV contour moments for accurate geometric centroid
            moments = cv2.moments(pts)
            if abs(moments["m00"]) > 1e-5:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
                return cx, cy
            # Fallback to mean of vertices
            return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        elif self.bbox is not None:
            x, y, w, h = self.bbox
            return float(x + w / 2.0), float(y + h / 2.0)
        return None


@dataclass
class TrackingDataset:
    """Dataset container holding sequence of frames and object annotations."""
    name: str
    frames: List[TrackedAnnotation] = field(default_factory=list)
    image_shape: Optional[Tuple[int, int, int]] = None

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> TrackedAnnotation:
        return self.frames[idx]

    def load_frame_image(self, idx: int) -> np.ndarray:
        """Load OpenCV BGR image for the given frame index."""
        item = self.frames[idx]
        if not os.path.exists(item.image_path):
            raise FileNotFoundError(f"Image not found at path: {item.image_path}")
        img = cv2.imread(item.image_path)
        if img is None:
            raise ValueError(f"OpenCV failed to read image at: {item.image_path}")
        return img


class TrackingDatasetLoader:
    """Loads tracking datasets from JSON files."""

    @staticmethod
    def load_from_json(json_path: Union[str, Path]) -> TrackingDataset:
        """Load dataset from a JSON annotation file.

        Supported JSON formats:
        1. Object-list style:
           {
               "dataset_name": "car_tracking",
               "frames": [
                   {"frame_index": 0, "image_path": "frames/000.png", "bbox": [100, 150, 60, 40]},
                   {"frame_index": 1, "image_path": "frames/001.png", "polygon": [[100, 150], [160, 150], [160, 190], [100, 190]]}
               ]
           }

        2. Key-path style dictionary:
           {
               "frames/000.png": {"bbox": [100, 150, 60, 40]},
               "frames/001.png": {"points": [[100, 150], [160, 150], [160, 190], [100, 190]]}
           }
        """
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {json_path}")

        base_dir = json_file.parent

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_name = data.get("dataset_name", json_file.stem) if isinstance(data, dict) else "custom_dataset"
        frames: List[TrackedAnnotation] = []

        if isinstance(data, dict) and "frames" in data and isinstance(data["frames"], list):
            # Format 1
            for idx, item in enumerate(data["frames"]):
                raw_path = item["image_path"]
                # Resolve relative paths relative to the JSON file location
                full_path = raw_path if os.path.isabs(raw_path) else str((base_dir / raw_path).resolve())
                
                bbox = None
                if "bbox" in item and item["bbox"] is not None:
                    b = item["bbox"]
                    bbox = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                elif "rect" in item and item["rect"] is not None:
                    b = item["rect"]
                    bbox = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

                polygon = None
                if "polygon" in item and item["polygon"] is not None:
                    polygon = [(float(p[0]), float(p[1])) for p in item["polygon"]]
                elif "points" in item and item["points"] is not None:
                    polygon = [(float(p[0]), float(p[1])) for p in item["points"]]

                ann = TrackedAnnotation(
                    frame_index=item.get("frame_index", idx),
                    image_path=full_path,
                    bbox=bbox,
                    polygon=polygon,
                    label=item.get("label", "object")
                )
                frames.append(ann)

        elif isinstance(data, dict):
            # Format 2 (Dictionary map)
            for idx, (img_key, ann_info) in enumerate(data.items()):
                if img_key in ("dataset_name", "metadata"):
                    continue
                full_path = img_key if os.path.isabs(img_key) else str((base_dir / img_key).resolve())
                
                bbox = None
                if isinstance(ann_info, dict) and ("bbox" in ann_info or "rect" in ann_info):
                    b = ann_info.get("bbox") or ann_info.get("rect")
                    bbox = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

                polygon = None
                if isinstance(ann_info, dict) and ("polygon" in ann_info or "points" in ann_info):
                    pts = ann_info.get("polygon") or ann_info.get("points")
                    polygon = [(float(p[0]), float(p[1])) for p in pts]

                ann = TrackedAnnotation(
                    frame_index=idx,
                    image_path=full_path,
                    bbox=bbox,
                    polygon=polygon,
                    label=ann_info.get("label", "object") if isinstance(ann_info, dict) else "object"
                )
                frames.append(ann)

        return TrackingDataset(name=dataset_name, frames=frames)

    @staticmethod
    def save_to_json(dataset: TrackingDataset, json_path: Union[str, Path]) -> None:
        """Serialize a TrackingDataset instance to JSON format."""
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)

        frames_data = []
        for f in dataset.frames:
            item: Dict[str, Any] = {
                "frame_index": f.frame_index,
                "image_path": f.image_path,
                "label": f.label,
            }
            if f.bbox is not None:
                item["bbox"] = list(f.bbox)
            if f.polygon is not None:
                item["polygon"] = [list(p) for p in f.polygon]
            if f.center_of_mass is not None:
                item["center_of_mass"] = list(f.center_of_mass)
            frames_data.append(item)

        data = {
            "dataset_name": dataset.name,
            "total_frames": len(dataset.frames),
            "frames": frames_data
        }

        with open(json_file, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)


class KITTITrackingLoader:
    """Parser and loader for the official KITTI 2D Tracking Benchmark dataset format."""

    @staticmethod
    def parse_kitti_label_file(label_path: Union[str, Path]) -> Dict[int, List[Dict[str, Any]]]:
        """Parse KITTI label_02 text file.

        Format per line:
        frame track_id type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom height width length x y z rotation_y
        """
        label_file = Path(label_path)
        if not label_file.exists():
            raise FileNotFoundError(f"KITTI label file not found: {label_path}")

        records_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                frame_idx = int(parts[0])
                track_id = int(parts[1])
                obj_type = parts[2]
                truncated = float(parts[3])
                occluded = int(parts[4])
                alpha = float(parts[5])
                bbox_left = float(parts[6])
                bbox_top = float(parts[7])
                bbox_right = float(parts[8])
                bbox_bottom = float(parts[9])

                w = max(1.0, bbox_right - bbox_left)
                h = max(1.0, bbox_bottom - bbox_top)
                cx = bbox_left + w / 2.0
                cy = bbox_top + h / 2.0

                item = {
                    "frame": frame_idx,
                    "track_id": track_id,
                    "type": obj_type,
                    "truncated": truncated,
                    "occluded": occluded,
                    "alpha": alpha,
                    "bbox": (int(bbox_left), int(bbox_top), int(w), int(h)),
                    "center_of_mass": (float(cx), float(cy)),
                }

                if frame_idx not in records_by_frame:
                    records_by_frame[frame_idx] = []
                records_by_frame[frame_idx].append(item)

        return records_by_frame

    @staticmethod
    def load_kitti_sequence(
        sequence_dir: Union[str, Path],
        label_file: Optional[Union[str, Path]] = None,
        track_id: Optional[int] = None,
        class_type: str = "Car"
    ) -> TrackingDataset:
        """Load a KITTI tracking image sequence and match ground-truth labels."""
        seq_path = Path(sequence_dir)
        if not seq_path.exists():
            raise FileNotFoundError(f"KITTI sequence directory not found: {sequence_dir}")

        image_files = sorted(list(seq_path.glob("*.png")) + list(seq_path.glob("*.jpg")))
        if not image_files:
            raise ValueError(f"No image files found in KITTI sequence directory: {sequence_dir}")

        labels_by_frame = {}
        if label_file is not None and Path(label_file).exists():
            labels_by_frame = KITTITrackingLoader.parse_kitti_label_file(label_file)

        frames: List[TrackedAnnotation] = []
        for idx, img_path in enumerate(image_files):
            # Search for matching object in this frame
            matched_obj = None
            if idx in labels_by_frame:
                candidates = labels_by_frame[idx]
                for c in candidates:
                    if class_type and c["type"].lower() != class_type.lower():
                        continue
                    if track_id is not None and c["track_id"] != track_id:
                        continue
                    matched_obj = c
                    break

            bbox = matched_obj["bbox"] if matched_obj else None
            com = matched_obj["center_of_mass"] if matched_obj else None
            lbl = matched_obj["type"] if matched_obj else "Unknown"

            polygon = None
            if bbox is not None:
                bx, by, bw, bh = bbox
                polygon = [
                    (float(bx), float(by)),
                    (float(bx + bw), float(by)),
                    (float(bx + bw), float(by + bh)),
                    (float(bx), float(by + bh))
                ]

            ann = TrackedAnnotation(
                frame_index=idx,
                image_path=str(img_path.resolve()),
                bbox=bbox,
                polygon=polygon,
                label=lbl,
                center_of_mass=com
            )
            frames.append(ann)

        return TrackingDataset(name=f"kitti_{seq_path.name}", frames=frames)

    @staticmethod
    def convert_kitti_to_json(
        sequence_dir: Union[str, Path],
        label_file: Union[str, Path],
        output_json: Union[str, Path],
        track_id: Optional[int] = None,
        class_type: str = "Car"
    ) -> Path:
        """Convert a KITTI tracking sequence to standard JSON format for student exercises."""
        dataset = KITTITrackingLoader.load_kitti_sequence(
            sequence_dir=sequence_dir,
            label_file=label_file,
            track_id=track_id,
            class_type=class_type
        )
        out_json_path = Path(output_json)
        TrackingDatasetLoader.save_to_json(dataset, out_json_path)
        return out_json_path


def create_kitti_sample_sequences(
    output_base_dir: Union[str, Path],
    num_frames: int = 60,
    width: int = 1242,
    height: int = 375
) -> Dict[str, Path]:
    """Generates a realistic KITTI-format driving tracking dataset.

    Creates driving street scenes matching standard KITTI dimensions (1242x375),
    with multiple moving cars, ego-motion road perspective, road markings,
    horizon, buildings, official KITTI label file format (0000.txt), and JSON annotations.

    Returns:
        Dictionary with paths to generated sequences and annotations.
    """
    base_path = Path(output_base_dir)
    seq_name = "0000"
    images_dir = base_path / "data_tracking_image_2" / "training" / "image_02" / seq_name
    labels_dir = base_path / "data_tracking_label_2" / "training" / "label_02"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    label_txt_path = labels_dir / f"{seq_name}.txt"
    json_path = base_path / f"kitti_{seq_name}_car_tracking.json"

    label_lines = []
    dataset_frames: List[TrackedAnnotation] = []

    # Two distinct cars:
    # Car 0 (Primary target): Moving ahead in our lane, slight sway and scaling
    # Car 1 (Opposing traffic): Moving across in left lane
    for i in range(num_frames):
        # Create road perspective background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sky & Horizon
        horizon_y = int(height * 0.45)
        frame[:horizon_y, :] = (210, 180, 140)  # Light sky
        # Road & Ground
        frame[horizon_y:, :] = (50, 50, 50)     # Asphalt road

        # Background urban buildings & trees
        cv2.rectangle(frame, (50, horizon_y - 80), (250, horizon_y), (100, 110, 120), -1)
        cv2.rectangle(frame, (300, horizon_y - 120), (500, horizon_y), (85, 95, 105), -1)
        cv2.rectangle(frame, (800, horizon_y - 90), (1100, horizon_y), (110, 120, 130), -1)

        # Perspective Road Markings (vanishing point at center horizon)
        vp_x, vp_y = width // 2, horizon_y
        cv2.line(frame, (vp_x, vp_y), (width // 4, height), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(frame, (vp_x, vp_y), (3 * width // 4, height), (255, 255, 255), 3, cv2.LINE_AA)
        
        # Center dashed lane divider with perspective spacing
        dash_offset = (i * 12) % 60
        for y_dash in range(horizon_y + 10, height, 40):
            yd = y_dash + dash_offset
            if yd >= height:
                continue
            t_dash = (yd - horizon_y) / (height - horizon_y)
            dash_x = int(vp_x + (width * 0.05) * (t_dash - 0.5))
            dash_len = int(10 + 25 * t_dash)
            cv2.line(frame, (dash_x, yd), (dash_x, min(height, yd + dash_len)), (0, 220, 255), max(1, int(3 * t_dash)), cv2.LINE_AA)

        # --- Object 0 (Lead Car in front of ego-vehicle) ---
        # Kinematics: Driving ahead with sinusoidal lane weaving
        t_prog = i / max(1, num_frames - 1)
        c0_scale = 1.0 + 0.3 * np.sin(t_prog * np.pi)  # Distance scaling
        c0_w = int(140 * c0_scale)
        c0_h = int(90 * c0_scale)
        c0_cx = int(width * 0.58 + 90.0 * np.sin(t_prog * 2.5 * np.pi))
        c0_cy = int(horizon_y + 80 + 30 * c0_scale)

        c0_x1 = c0_cx - c0_w // 2
        c0_y1 = c0_cy - c0_h // 2
        c0_x2 = c0_x1 + c0_w
        c0_y2 = c0_y1 + c0_h

        # Draw realistic Car 0 (Metallic Blue Sedan)
        # Shadow
        cv2.ellipse(frame, (c0_cx, c0_y2 + 2), (c0_w // 2 + 5, 8), 0, 0, 360, (20, 20, 20), -1)
        # Body
        cv2.rectangle(frame, (c0_x1, c0_y1 + c0_h // 3), (c0_x2, c0_y2), (180, 70, 30), -1)
        # Roof & Rear Windshield
        roof_inset = int(c0_w * 0.15)
        cv2.rectangle(frame, (c0_x1 + roof_inset, c0_y1), (c0_x2 - roof_inset, c0_y1 + c0_h // 2), (140, 50, 20), -1)
        cv2.rectangle(frame, (c0_x1 + roof_inset + 6, c0_y1 + 8), (c0_x2 - roof_inset - 6, c0_y1 + c0_h // 2 - 4), (40, 40, 40), -1)
        # Taillights
        light_w = int(c0_w * 0.18)
        light_h = int(c0_h * 0.16)
        cv2.rectangle(frame, (c0_x1 + 6, c0_y1 + c0_h // 2), (c0_x1 + 6 + light_w, c0_y1 + c0_h // 2 + light_h), (0, 0, 240), -1)
        cv2.rectangle(frame, (c0_x2 - 6 - light_w, c0_y1 + c0_h // 2), (c0_x2 - 6, c0_y1 + c0_h // 2 + light_h), (0, 0, 240), -1)
        # License Plate
        cv2.rectangle(frame, (c0_cx - 18, c0_y2 - 18), (c0_cx + 18, c0_y2 - 6), (230, 230, 230), -1)

        # Save KITTI label entry for Car 0
        # Format: frame track_id type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom h w l x y z rot_y
        label_lines.append(
            f"{i} 0 Car 0.00 0 1.57 {c0_x1:.2f} {c0_y1:.2f} {c0_x2:.2f} {c0_y2:.2f} 1.60 1.80 4.20 1.50 1.20 18.50 0.05"
        )

        # Save frame image
        img_name = f"{i:06d}.png"
        img_file = images_dir / img_name
        cv2.imwrite(str(img_file), frame)

        # Add to dataset frame list
        dataset_frames.append(TrackedAnnotation(
            frame_index=i,
            image_path=str(img_file.resolve()),
            bbox=(c0_x1, c0_y1, c0_w, c0_h),
            polygon=[
                (float(c0_x1), float(c0_y1)),
                (float(c0_x2), float(c0_y1)),
                (float(c0_x2), float(c0_y2)),
                (float(c0_x1), float(c0_y2))
            ],
            label="Car",
            center_of_mass=(float(c0_cx), float(c0_cy))
        ))

    # Write official KITTI label_02 text file
    with open(label_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines) + "\n")

    # Write corresponding JSON format
    kitti_dataset = TrackingDataset(name=f"kitti_seq_{seq_name}_cars", frames=dataset_frames)
    TrackingDatasetLoader.save_to_json(kitti_dataset, json_path)

    return {
        "sequence_dir": images_dir,
        "label_file": label_txt_path,
        "json_path": json_path,
        "dataset": kitti_dataset
    }


def generate_sample_tracking_dataset(
    output_dir: Union[str, Path],
    num_frames: int = 50,
    width: int = 640,
    height: int = 480,
    target_type: str = "car",
    include_polygons: bool = True
) -> Tuple[Path, TrackingDataset]:
    """Generates an image sequence and a JSON ground-truth annotation file for exercises."""
    out_path = Path(output_dir)
    frames_dir = out_path / "images"
    frames_dir.mkdir(parents=True, exist_ok=True)

    dataset_frames: List[TrackedAnnotation] = []
    t = np.linspace(0.0, 3.0 * np.pi, num_frames)
    target_w, target_h = 70, 40

    for i in range(num_frames):
        cx = 100.0 + (width - 200.0) * (i / max(1, num_frames - 1))
        cy = (height / 2.0) + 110.0 * np.sin(t[i])
        x = int(cx - target_w / 2.0)
        y = int(cy - target_h / 2.0)

        frame = np.full((height, width, 3), (35, 35, 35), dtype=np.uint8)
        cv2.line(frame, (0, height // 2), (width, height // 2), (180, 180, 180), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + target_w, y + target_h), (0, 0, 220), -1)

        img_filename = f"frame_{i:04d}.png"
        img_path = frames_dir / img_filename
        cv2.imwrite(str(img_path), frame)

        bbox = (x, y, target_w, target_h)
        polygon = None
        if include_polygons:
            polygon = [
                (float(x), float(y)),
                (float(x + target_w), float(y)),
                (float(x + target_w), float(y + target_h)),
                (float(x), float(y + target_h))
            ]

        ann = TrackedAnnotation(
            frame_index=i,
            image_path=str(img_path.resolve()),
            bbox=bbox,
            polygon=polygon,
            label=target_type,
            center_of_mass=(float(cx), float(cy))
        )
        dataset_frames.append(ann)

    dataset = TrackingDataset(name="sample_car_tracking", frames=dataset_frames)
    json_path = out_path / "tracking_annotations.json"
    TrackingDatasetLoader.save_to_json(dataset, json_path)

    return json_path, dataset


