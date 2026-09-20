"""Position Class - State Estimation & Localization Library for Self-Driving Cars."""

from .least_squares import (
    BatchLeastSquares,
    RecursiveLeastSquares,
    LeastSquaresResult,
    ohms_law_example,
    wheel_odometry_calibration_example,
    kinematic_position_velocity_example,
    lidar_plane_fitting_example,
    fit_lidar_ground_plane,
)
from .linear_kalman_filter import (
    LinearKalmanFilter,
    KalmanState,
    create_1d_constant_velocity_tracker,
    create_2d_constant_velocity_tracker,
    compute_nees,
)
from .extended_kalman_filter import (
    ExtendedKalmanFilter,
    LandmarkBearingEKF,
    Radar2DTargetTrackerEKF,
    wraptopi,
)
from .unscented_kalman_filter import (
    UnscentedKalmanFilter,
    LandmarkBearingUKF,
)
from .vision_kalman_tracker import (
    FastKalmanVisualTracker,
    generate_synthetic_tracking_video,
)
from .tracking_kalman import (
    KalmanTracker2D,
    KalmanTrackState,
    build_cv_matrices_2d,
)
from .tracking_io import (
    TrackedAnnotation,
    TrackingDataset,
    TrackingDatasetLoader,
    KITTITrackingLoader,
    generate_sample_tracking_dataset,
    create_kitti_sample_sequences,
)
from .error_service import (
    TrackingErrorSignal,
    ErrorPublisherService,
)
from .tracking_visualizer import (
    TrackingVisualizer,
)
from .ai_detectors import (
    DetectionResult,
    BaseDetector,
    YOLODetector,
    VisualFeatureTracker,
    FastFeatureTracker,
    HybridTracker,
)
from .tracking_exercise_json import (
    run_json_tracking_exercise,
)
from .interactive_tracking_exercise import (
    InteractiveKalmanAITracker,
    run_interactive_tracking_exercise,
)
from .base_kalman_tracker import (
    BaseKalmanTracker2D,
)
from .tracking_config import (
    TrackingConfig,
    parse_tracking_cli_args,
    build_tracking_arg_parser,
)
from .camera_stream import (
    CameraStream,
    InteractiveMouseHandler,
    show_detection_only_stream,
)
from .detection_only_exercise import (
    run_detection_only_exercise,
)
from .student_tracking_exercise import (
    StudentKalmanTracker2D,
    run_student_tracking_exercise,
)
try:
    from .instructor_tracking_exercise import (
        InstructorKalmanTracker2D,
        run_instructor_tracking_exercise,
    )
except ImportError:
    pass
from .inertial_navigation import (
    Quaternion,
    StrapdownIMUIntegrator,
    solve_gnss_trilateration,
    skew_symmetric,
)
from .es_ekf import (
    ErrorStateEKF,
    ErrorStateKalmanFilter,
)

__all__ = [
    "BatchLeastSquares",
    "RecursiveLeastSquares",
    "LeastSquaresResult",
    "ohms_law_example",
    "wheel_odometry_calibration_example",
    "kinematic_position_velocity_example",
    "lidar_plane_fitting_example",
    "fit_lidar_ground_plane",
    "LinearKalmanFilter",
    "KalmanState",
    "create_1d_constant_velocity_tracker",
    "create_2d_constant_velocity_tracker",
    "compute_nees",
    "ExtendedKalmanFilter",
    "LandmarkBearingEKF",
    "Radar2DTargetTrackerEKF",
    "wraptopi",
    "UnscentedKalmanFilter",
    "LandmarkBearingUKF",
    "FastKalmanVisualTracker",
    "generate_synthetic_tracking_video",
    "BaseKalmanTracker2D",
    "KalmanTracker2D",
    "KalmanTrackState",
    "build_cv_matrices_2d",
    "TrackingConfig",
    "parse_tracking_cli_args",
    "build_tracking_arg_parser",
    "CameraStream",
    "InteractiveMouseHandler",
    "show_detection_only_stream",
    "run_detection_only_exercise",
    "StudentKalmanTracker2D",
    "run_student_tracking_exercise",
    "InstructorKalmanTracker2D",
    "run_instructor_tracking_exercise",
    "TrackedAnnotation",
    "TrackingDataset",
    "TrackingDatasetLoader",
    "KITTITrackingLoader",
    "generate_sample_tracking_dataset",
    "create_kitti_sample_sequences",
    "TrackingErrorSignal",
    "ErrorPublisherService",
    "TrackingVisualizer",
    "DetectionResult",
    "BaseDetector",
    "YOLODetector",
    "VisualFeatureTracker",
    "FastFeatureTracker",
    "HybridTracker",
    "InteractiveKalmanAITracker",
    "run_json_tracking_exercise",
    "run_interactive_tracking_exercise",
    "Quaternion",
    "StrapdownIMUIntegrator",
    "solve_gnss_trilateration",
    "skew_symmetric",
    "ErrorStateEKF",
    "ErrorStateKalmanFilter",
]


