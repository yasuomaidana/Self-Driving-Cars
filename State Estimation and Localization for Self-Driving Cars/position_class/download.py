"""Download KITTI tracking dataset from Kaggle using kagglehub
and target the image_02 directory.
"""

from pathlib import Path
import kagglehub


def download_kitti_image_02() -> Path:
    """Downloads the 'leducnhuan/kitti-tracking' dataset via kagglehub
    and locates/returns the path to the training image_02 directory.
    """
    print("Downloading 'leducnhuan/kitti-tracking' dataset via kagglehub...")
    dataset_path = kagglehub.dataset_download("leducnhuan/kitti-tracking")
    dataset_root = Path(dataset_path)
    print(f"Dataset root path: {dataset_root}")

    # Search for image_02 directories inside the downloaded dataset
    image_02_dirs = list(dataset_root.rglob("image_02"))

    if not image_02_dirs:
        # Fallback to standard relative path if not directly matched
        target_dir = dataset_root / "training" / "image_02"
    else:
        # Prefer the training set image_02 if both training and testing are present
        training_image_02 = [d for d in image_02_dirs if "training" in str(d)]
        target_dir = training_image_02[0] if training_image_02 else image_02_dirs[0]

    print(f"\nTargeted image_02 directory:\n  {target_dir}")
    return target_dir


if __name__ == "__main__":
    download_kitti_image_02()
