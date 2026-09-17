"""Program 1: Visual Tracking Exercise using JSON Annotations and Kalman Filtering.

Reads an image sequence with bounding boxes or polygon coordinates from a JSON file,
estimates motion and predicts future positions with a 2D Kalman Filter,
and continuously publishes error signals relative to the image center.
"""

import argparse
from pathlib import Path
from typing import Optional, Union
import cv2

from .tracking_io import (
    TrackingDatasetLoader,
    generate_sample_tracking_dataset,
    TrackingDataset,
    KITTITrackingLoader,
    create_kitti_sample_sequences,
)
from .tracking_kalman import KalmanTracker2D
from .error_service import ErrorPublisherService, TrackingErrorSignal
from .tracking_visualizer import TrackingVisualizer


def run_json_tracking_exercise(
    json_path: Optional[Union[str, Path]] = None,
    kitti_dir: Optional[Union[str, Path]] = None,
    kitti_label: Optional[Union[str, Path]] = None,
    track_id: Optional[int] = 0,
    output_dir: Optional[Union[str, Path]] = None,
    use_kitti_demo: bool = False,
    save_video: bool = False,
    display: bool = False,
    class_type: str = "Car",
    dt: float = 1.0 / 30.0,
    process_noise: float = 0.8,
    measurement_noise: float = 2.0,
) -> ErrorPublisherService:
    """Run Program 1: JSON-driven Visual Object Tracking Exercise (KITTI or custom dataset).

    Args:
        json_path: Path to JSON annotations file.
        kitti_dir: Path to KITTI image_02 sequence folder (e.g. data_tracking_image_2/training/image_02/0000).
        kitti_label: Path to KITTI ground-truth label file (e.g. 0000.txt).
        track_id: KITTI track ID to isolate (e.g. 0 for primary target car).
        output_dir: Directory where annotated frames and CSV logs will be saved.
        use_kitti_demo: If True, generates a realistic KITTI-format driving car dataset.
        save_video: If True, writes an output MP4 video of the tracking run.
        display: If True, displays OpenCV GUI window during execution.
        class_type: Object class to track for KITTI datasets (e.g. "Car", "Pedestrian", "Cyclist", "Van").
        dt: Delta time between frames in seconds.
        process_noise: Process noise standard deviation.
        measurement_noise: Measurement noise standard deviation.

    Returns:
        ErrorPublisherService containing published error signals and computed metrics.
    """
    if output_dir is None:
        output_dir = Path("tracking_output_json")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset (KITTI sequence, JSON file, or synthetic demo)
    if kitti_dir is not None and Path(kitti_dir).exists():
        print(f"[Program 1] Loading KITTI tracking sequence from: {kitti_dir} (Track ID: {track_id}, Class: {class_type})...")
        dataset = KITTITrackingLoader.load_kitti_sequence(
            sequence_dir=kitti_dir,
            label_file=kitti_label,
            track_id=track_id,
            class_type=class_type
        )
    elif use_kitti_demo:
        print("[Program 1] Generating realistic KITTI-format driving car dataset (1242x375)...")
        kitti_info = create_kitti_sample_sequences(out_path / "kitti_demo", num_frames=60)
        dataset = kitti_info["dataset"]
        print(f"[Program 1] KITTI dataset generated at: {kitti_info['sequence_dir']}")
    elif json_path is not None and Path(json_path).exists():
        print(f"[Program 1] Loading dataset from JSON: {json_path}")
        dataset = TrackingDatasetLoader.load_from_json(json_path)
    else:
        print("[Program 1] No input provided. Generating sample driving tracking dataset...")
        sample_dir = out_path / "sample_data"
        json_path, dataset = generate_sample_tracking_dataset(sample_dir, num_frames=60)
        print(f"[Program 1] Sample dataset created at: {json_path}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. No frames to track.")

    # 2. Initialize Kalman Filter Module
    kalman = KalmanTracker2D(
        dt=dt,
        process_noise_std=process_noise,
        measurement_noise_std=measurement_noise
    )

    # 3. Initialize Error Publisher Service (Error Output Module)
    error_service = ErrorPublisherService(service_name="JSONTrackingErrorService")
    csv_log_path = out_path / "tracking_error_log.csv"
    error_service.enable_csv_logging(csv_log_path)

    # Simple console listener for student visibility
    def on_error_published(sig: TrackingErrorSignal) -> None:
        if sig.frame_index % 10 == 0 or sig.frame_index == len(dataset) - 1:
            print(f"[Frame {sig.frame_index:03d}] CoM: ({sig.target_x:.1f}, {sig.target_y:.1f}) | "
                  f"Center Error: {sig.error_distance:.2f} px (dx={sig.error_x:+.1f}, dy={sig.error_y:+.1f}) | "
                  f"KF Pred: ({sig.predicted_x:.1f}, {sig.predicted_y:.1f})")

    error_service.subscribe(on_error_published)

    # 4. Initialize Image Visualizer Module (Image Output Module)
    visualizer = TrackingVisualizer()
    annotated_frames_dir = out_path / "annotated_frames"
    annotated_frames_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None

    print(f"\n[Program 1] Starting tracking over {len(dataset)} frames...")
    print(f"{'='*75}")

    # Process all frames in the dataset
    for i in range(len(dataset)):
        ann = dataset[i]
        frame = dataset.load_frame_image(i)
        h, w = frame.shape[:2]

        if save_video and video_writer is None:
            video_file = str(out_path / "tracking_result.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_file, fourcc, 30.0, (w, h))

        # Measured Center of Mass from JSON annotation
        meas_com = ann.center_of_mass

        # Kalman Filter Predict & Update Cycle
        track_state = kalman.step(meas_com, image_shape=(h, w))

        # Compute & Publish Error Signal
        error_signal = TrackingErrorSignal.compute(
            frame_index=i,
            target_pos=(track_state.est_x, track_state.est_y),
            image_shape=(h, w),
            is_tracked=track_state.is_detected,
            predicted_pos=(track_state.pred_x, track_state.pred_y)
        )
        error_service.publish(error_signal)

        # Render Visual Output
        vis_frame = visualizer.draw_frame(
            frame=frame,
            frame_idx=i,
            bbox=ann.bbox,
            polygon=ann.polygon,
            meas_point=meas_com,
            est_point=(track_state.est_x, track_state.est_y),
            pred_point=(track_state.pred_x, track_state.pred_y),
            velocity=(track_state.est_vx, track_state.est_vy),
            error_distance=error_signal.error_distance,
            status_text="TRACKING (JSON)",
            label=ann.label
        )

        # Save annotated image
        cv2.imwrite(str(annotated_frames_dir / f"vis_{i:04d}.png"), vis_frame)

        if video_writer is not None:
            video_writer.write(vis_frame)

        if display:
            cv2.imshow("Program 1: JSON Tracking Exercise", vis_frame)
            if cv2.waitKey(int(dt * 1000)) & 0xFF == ord('q'):
                break

    if video_writer is not None:
        video_writer.release()

    if display:
        cv2.destroyAllWindows()

    error_service.close()

    # Summary Metrics
    metrics = error_service.compute_metrics()
    print(f"{'='*75}")
    print("[Program 1] Tracking Complete!")
    print(f"  • Total Frames Processed: {metrics.get('total_samples', len(dataset))}")
    print(f"  • Mean Distance Error to Center: {metrics['mean_error']:.2f} px")
    print(f"  • Root Mean Squared Error (RMSE): {metrics['rmse']:.2f} px")
    print(f"  • Maximum Distance Error:        {metrics['max_error']:.2f} px")
    print(f"  • CSV Error Log Saved:            {csv_log_path}")
    print(f"  • Annotated Frames Saved to:      {annotated_frames_dir}")
    print(f"{'='*75}\n")

    return error_service


def main():
    parser = argparse.ArgumentParser(description="Program 1: JSON & KITTI Kalman Visual Tracking Exercise")
    parser.add_argument("--json", type=str, default=None, help="Path to JSON annotation file")
    parser.add_argument("--kitti-dir", type=str, default=None, help="Path to KITTI image_02 sequence directory")
    parser.add_argument("--kitti-label", type=str, default=None, help="Path to KITTI label_02 text file")
    parser.add_argument("--track-id", type=int, default=0, help="KITTI car track ID (default: 0)")
    parser.add_argument("--class", "--target-class", dest="target_class", type=str, default="Car",
                        help="Target object class for KITTI datasets (e.g. 'Car', 'Pedestrian', 'Cyclist', 'Van'). Default: Car")
    parser.add_argument("--kitti", action="store_true", help="Run with realistic KITTI driving car sequence")
    parser.add_argument("--out", type=str, default="tracking_output_json", help="Output directory")
    parser.add_argument("--display", action="store_true", help="Display OpenCV window")
    parser.add_argument("--video", action="store_true", help="Save output MP4 video")
    args = parser.parse_args()

    run_json_tracking_exercise(
        json_path=args.json,
        kitti_dir=args.kitti_dir,
        kitti_label=args.kitti_label,
        track_id=args.track_id,
        class_type=args.target_class,
        use_kitti_demo=args.kitti,
        output_dir=args.out,
        save_video=args.video,
        display=args.display
    )


if __name__ == "__main__":
    main()
