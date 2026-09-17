"""Unit Tests for Tracking IO & Annotation Module."""

import json
import math
from pathlib import Path
import numpy as np
import pytest

from position_class.tracking_io import (
    TrackedAnnotation,
    TrackingDatasetLoader,
    generate_sample_tracking_dataset,
)


class TestTrackingIO:
    """Test suite for tracking input and JSON dataset operations."""

    def test_center_of_mass_from_bbox(self):
        """Test Center of Mass calculation from bounding box."""
        ann = TrackedAnnotation(
            frame_index=0,
            image_path="test.png",
            bbox=(100, 200, 40, 60)
        )
        assert ann.center_of_mass is not None
        assert ann.center_of_mass == (120.0, 230.0)

    def test_center_of_mass_from_polygon(self):
        """Test Center of Mass calculation from polygon coordinates."""
        polygon = [
            (10.0, 10.0),
            (30.0, 10.0),
            (30.0, 50.0),
            (10.0, 50.0)
        ]
        ann = TrackedAnnotation(
            frame_index=0,
            image_path="test.png",
            polygon=polygon
        )
        assert ann.center_of_mass is not None
        assert math.isclose(ann.center_of_mass[0], 20.0, abs_tol=1e-3)
        assert math.isclose(ann.center_of_mass[1], 30.0, abs_tol=1e-3)

    def test_json_save_and_load_roundtrip(self, tmp_path):
        """Test generating, saving to JSON, and loading back tracking dataset."""
        json_path, dataset = generate_sample_tracking_dataset(
            output_dir=tmp_path / "dataset",
            num_frames=10,
            width=320,
            height=240,
            include_polygons=True
        )

        assert json_path.exists()
        assert len(dataset) == 10

        loaded_dataset = TrackingDatasetLoader.load_from_json(json_path)
        assert len(loaded_dataset) == 10
        assert loaded_dataset[0].bbox is not None
        assert loaded_dataset[0].polygon is not None
        assert loaded_dataset[0].center_of_mass is not None

        # Verify image reading with OpenCV
        img = loaded_dataset.load_frame_image(0)
        assert img is not None
        assert img.shape == (240, 320, 3)

    def test_kitti_generation_and_loading(self, tmp_path):
        """Test generating, parsing, and loading KITTI benchmark format sequences."""
        from position_class.tracking_io import create_kitti_sample_sequences, KITTITrackingLoader

        kitti_info = create_kitti_sample_sequences(tmp_path / "kitti_test", num_frames=10)
        assert kitti_info["sequence_dir"].exists()
        assert kitti_info["label_file"].exists()
        assert kitti_info["json_path"].exists()

        # Test parsing raw KITTI label file
        parsed_labels = KITTITrackingLoader.parse_kitti_label_file(kitti_info["label_file"])
        assert len(parsed_labels) == 10
        assert 0 in parsed_labels
        assert parsed_labels[0][0]["type"] == "Car"
        assert parsed_labels[0][0]["track_id"] == 0

        # Test loading KITTI sequence directly
        dataset = KITTITrackingLoader.load_kitti_sequence(
            sequence_dir=kitti_info["sequence_dir"],
            label_file=kitti_info["label_file"],
            track_id=0,
            class_type="Car"
        )
        assert len(dataset) == 10
        assert dataset[0].center_of_mass is not None
        assert dataset[0].label == "Car"

        img = dataset.load_frame_image(0)
        assert img.shape == (375, 1242, 3)

