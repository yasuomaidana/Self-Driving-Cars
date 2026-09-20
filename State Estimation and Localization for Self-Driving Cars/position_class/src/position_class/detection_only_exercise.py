"""Program: AI Vision & Keypoint Detection Visualizer (No Kalman Filter).

Dedicated standalone tool for inspecting visual detectors, bounding boxes,
camera streams, and object candidate selection without filtering.
"""

import argparse
from typing import Optional, Union
from pathlib import Path

from position_class import parse_tracking_cli_args, TrackingConfig
from position_class import show_detection_only_stream


def run_detection_only_exercise(
    source: Optional[Union[str, int]] = None,
    camera_id: Optional[int] = None,
    model_name: str = "yolov8n.pt",
    dataset: Optional[str] = None,
    target_class: Optional[str] = None,
    interactive_gui: bool = True,
    max_frames: Optional[int] = None,
) -> None:
    """Entrypoint function to run visual detection only."""
    config = TrackingConfig(
        source=source if camera_id is None else camera_id,
        camera_id=camera_id,
        model_name=model_name,
        dataset=dataset,
        target_class=target_class,
        interactive_gui=interactive_gui,
        max_live_frames=max_frames
    )
    show_detection_only_stream(config=config)


def main():
    config = parse_tracking_cli_args(description="AI & Vision Detection Visualizer (Detection Only)")
    show_detection_only_stream(config=config)


if __name__ == "__main__":
    main()
