"""Configuration and CLI parameter parsing for Visual Object Tracking exercises.

Centralizes CLI options, hyperparameters, and default settings for both
Student and Instructor tracking programs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
import argparse


@dataclass
class TrackingConfig:
    """Dataclass holding all runtime configuration and hyperparameters."""
    # Data source settings
    source: Optional[Union[str, int]] = None
    camera_id: Optional[int] = None
    kitti_dir: Optional[Union[str, Path]] = None
    kitti_label: Optional[Union[str, Path]] = None
    use_kitti_demo: bool = False
    output_dir: Path = field(default_factory=lambda: Path("tracking_output_interactive"))

    # AI Detection / Vision settings
    model_name: str = "yolov8n.pt"
    dataset: Optional[str] = None
    target_class: Optional[str] = None
    feature_algorithm: str = "fast"
    num_points: int = 3
    hybrid_mode: bool = False
    feature_only: bool = False
    disable_helper: bool = False

    # Kalman Filter Hyperparameters (Day 01 & Day 02)
    dt: float = 1.0 / 30.0
    process_noise: float = 1.5       # sigma_a (px/s^2 acceleration noise)
    measurement_noise: float = 2.0   # sigma_meas (pixels detector noise)
    initial_covariance: float = 50.0 # sigma_0^2 (initial uncertainty)

    # UI / Execution controls
    auto_lock: bool = False
    start_paused: bool = False
    interactive_gui: bool = True
    save_output: bool = True
    auto_click_frame: int = 5
    auto_click_coords: Optional[Tuple[int, int]] = None
    max_live_frames: Optional[int] = None
    service_name: str = "TrackingVisualErrorService"

    def __post_init__(self):
        """Sanitize and resolve paths and flags."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Normalize feature algorithm and helper flag
        raw_algo = str(self.feature_algorithm).lower().strip()
        if self.disable_helper or raw_algo in ("none", "off", "disabled", "no", "false", ""):
            self.disable_helper = True
            self.feature_algorithm = "none"
        else:
            self.feature_algorithm = raw_algo

        self.num_points = max(1, int(self.num_points))

        # Check camera mode
        if self.camera_id is not None:
            self.source = self.camera_id
        elif isinstance(self.source, str) and self.source.isdigit():
            self.camera_id = int(self.source)
            self.source = int(self.source)


def build_tracking_arg_parser(description: str = "Visual Object Tracking Exercise") -> argparse.ArgumentParser:
    """Constructs the standard CLI ArgumentParser for tracking programs."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", type=str, default=None, help="Input video file, image folder, camera index, or JSON dataset")
    parser.add_argument("--camera", action="store_true", help="Use live webcam / camera stream (default device 0)")
    parser.add_argument("--camera-id", type=int, default=None, help="Camera device index (e.g. 0, 1)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model name or weights file (e.g. yolov8n.pt, yolo26n.pt)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset preset for YOLO classes (e.g. 'kitti', 'coco', 'visdrone')")
    parser.add_argument("--class", "--target-class", dest="target_class", type=str, default=None,
                        help="Filter target class (e.g. 'person', 'car', 'cup', 'bottle'). Default: all classes")
    parser.add_argument("--hybrid", action="store_true", help="Enable Hybrid Mode: feature tracking with dynamic refresh & YOLO re-acquisition")
    parser.add_argument("--feature-only", "--no-yolo", action="store_true", help="Run strictly in Feature-Only mode (FAST/SIFT/ORB + Kalman)")
    parser.add_argument("--algorithm", "--feature-algo", "--algo", dest="algorithm", type=str, default="fast",
                        choices=["fast", "sift", "orb", "none", "off", "disabled"],
                        help="Feature algorithm: 'fast', 'sift', 'orb', or 'none' (default: 'fast')")
    parser.add_argument("--disable-helper", "--no-helper", action="store_true", help="Disable visual feature helper algorithm")
    parser.add_argument("--num-points", "--points", "-k", dest="num_points", type=int, default=3,
                        help="Number of feature keypoints to track and refresh (default: 3)")
    parser.add_argument("--sift", action="store_true", help="Shortcut to run with SIFT feature extractor")
    parser.add_argument("--orb", action="store_true", help="Shortcut to run with ORB feature extractor")
    parser.add_argument("--fast", action="store_true", help="Shortcut to run with FAST feature extractor")
    parser.add_argument("--auto-lock", action="store_true", help="Automatically lock target upon first detection")
    parser.add_argument("--paused", "--pause", dest="paused", action="store_true", help="Start in paused mode to select target easily")
    parser.add_argument("--kitti-dir", type=str, default=None, help="Path to KITTI image_02 sequence folder")
    parser.add_argument("--kitti-label", type=str, default=None, help="Path to KITTI label_02 text file")
    parser.add_argument("--kitti", action="store_true", help="Run with realistic KITTI driving street scene demo")
    parser.add_argument("--dt", type=float, default=1.0 / 30.0, help="Delta time per frame in seconds (default: 1/30)")
    parser.add_argument("--process-noise", "--q-noise", dest="process_noise", type=float, default=1.5,
                        help="Process acceleration noise standard deviation sigma_a (default: 1.5)")
    parser.add_argument("--measurement-noise", "--r-noise", dest="measurement_noise", type=float, default=2.0,
                        help="Measurement noise standard deviation sigma_meas in pixels (default: 2.0)")
    parser.add_argument("--out", type=str, default="tracking_output_interactive", help="Output directory for saved artifacts")
    parser.add_argument("--no-gui", action="store_true", help="Run without interactive GUI window (headless/test mode)")
    return parser


def parse_tracking_cli_args(cli_args=None, description: str = "Visual Object Tracking Exercise") -> TrackingConfig:
    """Parses command line arguments into a structured TrackingConfig instance."""
    parser = build_tracking_arg_parser(description=description)
    args = parser.parse_args(cli_args)

    cam_id = args.camera_id if args.camera_id is not None else (0 if args.camera else None)
    is_helper_disabled = args.disable_helper or (args.algorithm.lower().strip() in ["none", "off", "disabled"])

    selected_algo = args.algorithm
    if args.sift:
        selected_algo = "sift"
        is_helper_disabled = False
    elif args.orb:
        selected_algo = "orb"
        is_helper_disabled = False
    elif args.fast:
        selected_algo = "fast"
        is_helper_disabled = False
    elif is_helper_disabled:
        selected_algo = "none"

    is_kitti = bool(args.kitti or args.kitti_dir is not None)
    if is_kitti and not args.hybrid and not is_helper_disabled:
        is_feat_only = True
    else:
        is_feat_only = (args.feature_only or ((args.sift or args.orb or args.fast) and not args.hybrid)) and not is_helper_disabled

    effective_dataset = args.dataset if args.dataset is not None else ("kitti" if is_kitti else None)

    return TrackingConfig(
        source=args.source if cam_id is None else cam_id,
        camera_id=cam_id,
        kitti_dir=args.kitti_dir,
        kitti_label=args.kitti_label,
        use_kitti_demo=args.kitti,
        output_dir=Path(args.out),
        model_name=args.model,
        dataset=effective_dataset,
        target_class=args.target_class,
        feature_algorithm=selected_algo,
        num_points=args.num_points,
        hybrid_mode=args.hybrid,
        feature_only=is_feat_only,
        disable_helper=is_helper_disabled,
        dt=args.dt,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
        auto_lock=args.auto_lock,
        start_paused=args.paused,
        interactive_gui=not args.no_gui
    )
