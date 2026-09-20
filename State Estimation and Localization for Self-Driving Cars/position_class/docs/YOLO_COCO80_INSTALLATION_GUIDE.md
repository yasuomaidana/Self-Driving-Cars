# Complete Guide: Installing & Using COCO 80-Class Deep Learning Neural Networks

This guide provides step-by-step instructions to install, configure, and run **COCO 80-class deep learning object detection neural networks** (YOLOv8, YOLOv9, YOLOv10, YOLO11, and YOLO-Pose keypoints) with hardware acceleration (**Apple Silicon MPS / NVIDIA CUDA / CPU**) for the tracking system.

---

## 1. Quick Installation (Recommended)

In this project, package management is handled by `uv`. You can install the complete **Ultralytics YOLO + PyTorch** deep learning stack with a single command:

```bash
# Navigate to the project folder
cd "/Users/yasuomaidana/Projects/classes/Self-Driving-Cars/State Estimation and Localization for Self-Driving Cars/position_class"

# Install Ultralytics and PyTorch via uv
uv add ultralytics
```

> [!NOTE]
> If you are using standard `pip` in a Python virtual environment:
> ```bash
> pip install ultralytics torch torchvision
> ```

---

## 2. Hardware Acceleration Support

Ultralytics and PyTorch automatically detect and leverage the best available hardware accelerator:

| Hardware Platform | Acceleration Backend | Configuration | Expected Latency |
| :--- | :--- | :--- | :--- |
| **Mac (Apple Silicon M1/M2/M3/M4)** | **Metal Performance Shaders (MPS)** | Automatic | **4 – 8 ms / frame** |
| **Linux / Windows (NVIDIA GPU)** | **NVIDIA CUDA + TensorRT** | `torch.cuda.is_available()` | **1 – 4 ms / frame** |
| **CPU (Intel / AMD / Mac x86)** | **OpenMP / AVX2 / NEON** | Default fallback | **15 – 35 ms / frame** |

### Verify Hardware Acceleration
To confirm PyTorch and MPS/CUDA are working:
```bash
uv run python -c "import torch, ultralytics; print(f'PyTorch: {torch.__version__} | MPS (Apple Silicon): {torch.backends.mps.is_available()} | CUDA: {torch.cuda.is_available()}')"
```

---

## 3. Available Model Sizes & Weights

Ultralytics automatically downloads pretrained model weights on the first run. You can select the model that fits your speed vs. accuracy requirement:

| Model File | Parameters | mAP50-95 (COCO) | Inference Speed (MPS) | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `yolov8n.pt` *(Default)* | 3.2M | 37.3 | **~5 ms** (Ultra-Fast) | Live Webcam, Real-Time Tracking |
| `yolov8s.pt` | 11.2M | 44.9 | **~9 ms** (Balanced) | KITTI Driving Scenes |
| `yolov8m.pt` | 25.9M | 50.2 | **~18 ms** (High Accuracy) | Complex Traffic / Small Objects |
| `yolov8l.pt` / `yolov8x.pt` | 43.7M+ | 52.9+ | **~35 ms** (Maximum Accuracy) | High-Resolution Offline Analysis |
| `yolov8n-pose.pt` | 3.3M | 50.4 (Pose) | **~6 ms** (Keypoints) | **Eye, Face & Human Body Tracking** |

---

## 4. The 80 MS COCO Object Classes Reference

When `ultralytics` is installed, the AI detector recognizes all 80 standard categories:

```
┌─────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Category Group  │ COCO 80 Class Names                                                    │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Vehicles        │ person, bicycle, car, motorcycle, airplane, bus, train, truck, boat    │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Traffic & City  │ traffic light, fire hydrant, stop sign, parking meter, bench           │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Animals         │ bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe       │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Accessories     │ backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard   │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Sports Items    │ sports ball, kite, baseball bat, baseball glove, skateboard, surfboard │
│                 │ tennis racket                                                          │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Kitchenware     │ bottle, wine glass, cup, fork, knife, spoon, bowl                      │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Food            │ banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza      │
│                 │ donut, cake                                                            │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Indoor Furniture│ chair, couch, potted plant, bed, dining table, toilet                  │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Electronics     │ tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven       │
│                 │ toaster, sink, refrigerator                                            │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Household / Misc│ book, clock, vase, scissors, teddy bear, hair drier, toothbrush        │
└─────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Running the Tracking System with COCO Models

### A. Live Webcam with General COCO Detection
```bash
# Detects all 80 object classes in real time
uv run track-interactive --camera --hybrid --model yolov8n.pt
```

### B. Filter for Specific COCO Classes
```bash
# Filter for personal desktop items
uv run track-interactive --camera --hybrid --class "cup, bottle, cell phone, laptop"

# Filter for human tracking
uv run track-interactive --camera --hybrid --class person
```

### C. Human Body & Eye Keypoint Tracking (YOLOv8-Pose)
```bash
# Tracks human body keypoints (Left Eye = Keypoint 1, Right Eye = Keypoint 2)
uv run track-interactive --camera --hybrid --model yolov8n-pose.pt --class eye
```

### D. Self-Driving Driving Dataset (KITTI)
```bash
# Automatically loads and tracks cars on realistic street scenes
uv run track-interactive --kitti --hybrid --model yolov8s.pt --class car
```

---

## 6. Python API Usage

You can also use the neural network directly in Python:

```python
import cv2
from position_class import YOLODetector, HybridTracker, KalmanTracker2D

# 1. Initialize YOLO detector with COCO 80 weights
detector = YOLODetector(model_name="yolov8n.pt", conf_threshold=0.35)
print("Active Model:", detector.get_model_info())

# 2. Read frame
img = cv2.imread("frame.jpg")

# 3. Detect all COCO objects or specific classes
detections = detector.detect_all(img, target_class="car, person, cup")

for det in detections:
    print(f"Detected: {det.label} at CoM: {det.center_of_mass}, BBox: {det.bbox}")
```

---

## 7. Troubleshooting

* **Issue: `ModuleNotFoundError: No module named 'ultralytics'`**
  * **Fix**: Run `uv add ultralytics`. The tracking script will automatically detect the package upon restart.
* **Issue: First-time run is slow**
  * **Explanation**: On the first execution, Ultralytics downloads the `.pt` weight file (~6 MB for `yolov8n.pt`) and caches it in `~/.config/Ultralytics`. Subsequent runs load instantaneously from disk cache.
* **Issue: Tracking a custom non-COCO object (e.g. specialized parts, eye iris)**
  * **Fix**: Press <kbd>Space</kbd> in the program to freeze the video, then click and drag a green box over the feature. The FAST 3-point feature tracker will extract localized texture gradients and track it with Kalman filtering.
