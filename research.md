# Research Avenues

A living document of research material supporting development of the football kick tracker pipeline. This will be revisited and expanded as the project progresses.

> Initially drafted with AI assistance and will be updated throughout the project.

---

## Contents

- [**Pose Estimation**](#pose-estimation)
  - [Human Pose Estimation - Background](#human-pose-estimation---background)
  - [Ultralytics YOLO POSE](#ultralytics-yolo-pose)
  - [ViTPose](#vitpose)
  - [RTMPose](#rtmpose)
- [**Pose Based Action Recognition**](#pose-based-action-recognition)
- [**Inference Smoothing**](#inference-smoothing)
- [**OpenCV - Video and Calibration**](#opencv---video-and-calibration)
- [**Sports Analytics and Kick Detection - Applied Work**](#sports-analytics-and-kick-detection---applied-work)
- [**Annotation - CVAT**](#annotation---cvat)
- [**Conventional Commits**](#conventional-commits)
- [**Ball Detection and Tracking**](#ball-detection-and-tracking)
- [**Later - Park for Now**](#later---park-for-now)
---

## Pose Estimation

### Human Pose Estimation - Background

Background on how pose estimation models work under the hood. Useful for debugging unexpected keypoint outputs and for understanding model limitations in the field.

### Papers
- **OpenPose** - *Realtime Multi-Person 2D Pose Estimation* (Cao et al., 2017) - the paper that made multi-person pose estimation practical. Foundational reading.
  - [arXiv:1611.08050](https://arxiv.org/abs/1611.08050)
- **HRNet** - *Deep High-Resolution Representation Learning for Visual Recognition* (Wang et al., 2019) - the architecture behind many modern pose estimators including those in MMPose.
  - [arXiv:1908.07919](https://arxiv.org/abs/1908.07919)

### Videos
- Search YouTube: **"human pose estimation deep learning"** - good coverage exists from university CV courses
- Search YouTube: **"YOLO pose estimation tutorial"** - practical walkthroughs of the Ultralytics pipeline

---

### Ultralytics YOLO Pose

The immediate pipeline dependency for Phase 1. Understanding the output format is required before any detection logic can be designed around it. Current working default is YOLO11l-pose (see `notebooks/pose/YOLO_candidate_comparison.ipynb` for benchmarking notes and decision log).

**Goal:** given a frame, what does the keypoint tensor look like and how does one index into it for the leg joints?

### Docs
- [Ultralytics YOLO Pose - official docs](https://docs.ultralytics.com/tasks/pose/)
- [Keypoint output format reference](https://docs.ultralytics.com/reference/engine/results/)

### Key concepts to understand
- The 17-point COCO keypoint schema - which indices map to hips, knees, ankles, and feet
- Per-keypoint confidence scores and how to threshold them
- Difference between `Results.keypoints.xy` (pixel coords) and `Results.keypoints.xyn` (normalised)
- How to run inference on a video vs a single frame

### Practical task
Run quickstart inference on any video clip and print the raw keypoint tensor to the terminal. This is the most valuable first step before starting Phase 1.

---

### ViTPose

A strong non-YOLO alternative for pose estimation, and a genuine candidate to replace or supplement in a future phase. ViTPose uses a plain, non-hierarchical vision transformer as a backbone with a lightweight decoder for keypoint estimation. Its key strengths are scalability across model sizes (ViT-S through ViT-H) and strong performance on the COCO benchmark. ViTPose++ extends the original to handle heterogeneous body keypoint categories across multiple pose estimation tasks simultaneously.
Goal: understand how ViTPose's architecture differs from YOLO's approach and evaluate whether it is worth integrating into the pipeline as a candidate model.

### Papers

- **ViTPose** - *ViTPose: Simple Vision Transformer Baseline for Human Pose Estimation*
  - [arXiv:2204.12484](https://arxiv.org/abs/2204.12484)
- **ViTPose++** - *ViTPose++: Vision Transformer for Generic Body Pose Estimation*
  - [arXiv:2212.04246](https://arxiv.org/abs/2212.04246)

### Code

- [GitHub: ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose) - official repo, requires mmcv/mmpose
- [GitHub: Tau-J/rtmlib](https://github.com/Tau-J/rtmlib) - lightweight wrapper that runs ViTPose++ models without the full MMPose stack; far easier to integrate

### Key concepts to understand

- How plain vision transformers differ from CNN-based architectures like HRNet (used by YOLO's pose head)
- Top-down vs bottom-up pose estimation paradigms and the trade-offs for a single-player setting
Knowledge distillation from large to small ViTPose models via knowledge tokens - relevant for future fine-tuning
- How `rtmlib` abstracts the inference pipeline and whether it can be slotted into the existing `pose/inference.py` structure

---

### RTMPose

A high-performance real-time multi-person pose estimation framework built on MMPose. RTMPose is explicitly designed for practical deployment - it achieves strong COCO accuracy at high inference speeds across CPU, GPU, and mobile hardware, and supports ONNX, TensorRT, and ncnn backends. Its inference pipeline includes built-in pose NMS and smoothing filtering, which is directly relevant to the smoothing problem described in the [Inference Smoothing](#inference-smoothing) section.

**Goal:** evaluate RTMPose as a cadidate, with particular attention to inference speed and the built-in smoothing pipeline.

### Papers
- **RTMPose** - *Real-Time Multi-Person Pose Estimation based on MMPose* (Jiang et al., 2023)
  - [arXiv:2303.07399](https://arxiv.org/abs/2303.07399)

### Code
- [GitHub: open-mmlab/mmpose RTMPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) - official implementation
- [GitHub: Tau-J/rtmlib](https://github.com/Tau-J/rtmlib) - same lightweight wrapper as for ViTPose; supports RTMPose models without mmcv/mmpose dependencies

---

## Pose-Based Action Recognition

The academic framing of the core detection problem. Understanding this landscape informs the rule-based detector design and the eventual neural approach.

### Papers
- **ST-GCN** - *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition* (Yan et al., 2018) - foundational paper explaining why pose sequences are naturally a graph problem. Worth skimming for the mental model.
  - [arXiv:1801.07455](https://arxiv.org/abs/1801.07455)
- **PoseConv3D** - *Revisiting Skeleton-based Action Recognition* (Duan et al., 2021) - more recent approach using 3D heatmap volumes instead of graph networks; strong benchmark results.
  - [arXiv:2104.13586](https://arxiv.org/abs/2104.13586)
- **PYSKL** - a skeleton-based action recognition toolbox built on PoseConv3D and ST-GCN variants; useful to browse for reference.
  - [GitHub: kennymckormick/pyskl](https://github.com/kennymckormick/pyskl)

### Key concepts to understand
- Sliding window classification over pose sequences - the pattern the temporal classifier will follow
- Computing joint angles and angular velocities from keypoint coordinates
- Why temporal context matters - a single frame is ambiguous, a window of frames is not
- The difference between detection (did a kick happen?) and localisation (exactly when?)

### Videos
- Search YouTube: **"skeleton action recognition explained"** - several lecture-style walkthroughs exist from university CV courses
- Search YouTube: **"pose estimation sports analytics"** - useful for seeing applied work in a similar domain

---

## Inference Smoothing

Noisy or jittery pose detections are an expected failure mode when using YOLO11l-pose on training footage - particularly at distance and at non-standard camera angles. This section covers approaches to smooth and stabilise keypoint predictions across frames, both classical and learned.

### Approaches to understand

**Kalman filtering** is the standard classical approach for smoothing noisy position estimates across frames. It models the expected motion of a tracked object and uses a predict-correct cycle to reduce noise. For pose estimation, this means treating each keypoint independently as a 2D position to be filtered. OpenCV has a built-in `cv2.KalmanFilter` implementation.

- Search: *"Kalman filter pose estimation Python"* and *"Kalman filter object tracking OpenCV"*
- Key concepts: process noise vs measurement noise, the predict/update cycle, when Kalman filtering degrades (fast, non-linear motion)

**Temporal averaging / neighbour interpolation** is a simpler frame-level approach. The core idea is: given a keypoint detection in frame N, check whether frames N-1 and N+1 also contain a detection of the same person (identified by spatial proximity of keypoints). If all three frames agree on the rough position of a keypoint, the frame N value can be replaced by the mean of all three. This artificially generated pose is more stable than the raw detection alone. Key challenges are identity matching across frames (ensuring N-1 and N+1 are the same person) and handling missing detections gracefully.

**RTMPose's built-in smoothing** - RTMPose's inference pipeline includes pose NMS and smoothing filtering as first-class features (see [RTMPose](#rtmpose)). Worth understanding what it does before building something custom, as it may be sufficient.

**One Euro Filter** - a low-latency, parameter-tunable signal filter designed for real-time tracking. Trades off smoothing strength against lag dynamically based on the speed of the signal. A good alternative to Kalman when motion is irregular.

- Search: *"One Euro Filter pose estimation"*
- [Original One Euro Filter paper: hal.inria.fr](https://hal.inria.fr/hal-00670496/document)

### Key concepts to understand
- The difference between smoothing in post-processing (offline, full-clip context available) vs smoothing in real-time (causal, only past frames available)
- Identity matching across frames - how to deduce that the same person was detected in two adjacent frames
- How smoothing interacts with kick detection: over-smoothing may blur the sharp motion signal of a kick

---

## OpenCV - Video and Calibration

Used from Phase 1 onwards for video I/O and visualisation, and for camera calibration in Phase 2.

### Docs
- [OpenCV Python tutorials - official](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [ChArUco board calibration](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html)

### Phase 1 priorities
- `cv2.VideoCapture` - reading video files frame by frame
- `cv2.VideoWriter` - writing annotated output video
- Drawing utilities: `cv2.circle`, `cv2.line`, `cv2.putText` - for overlaying keypoints and kick event markers

### Phase 2 priorities (park for now)
- Camera intrinsics and extrinsics - the conceptual model of what calibration is computing
- `cv2.calibrateCamera` and `cv2.solvePnP`
- Triangulation with `cv2.triangulatePoints`

---

## Sports Analytics and Kick Detection - Applied Work

Relevant to understand the existing landscape of applied work in the space before implementing.

### Papers
- **SoccerNet** - *A Scalable Dataset for Action Spotting in Soccer Videos* (Giancola et al., 2018) - the main academic football video dataset. Action spotting is closely related to the kick detection problem.
  - [arXiv:1804.04527](https://arxiv.org/abs/1804.04527)
- **Action Spotting in Soccer** - search arXiv for recent SoccerNet challenge papers (2021–2024); the challenge has pushed the state of the art on temporal event detection in football footage.
- **Inertial Sensor Kick Detection** - search Google Scholar: *"football kick detection IMU"* or *"soccer kick detection inertial sensor"* - relevant for Phase 5 IMU integration.

### Videos
- Search YouTube: **"SoccerNet action spotting"** - workshop presentations give a good overview of the problem space
- Search YouTube: **"football player tracking computer vision"** - useful for understanding multi-player tracking approaches

---

## Annotation - CVAT

Needed for Phase 4 dataset construction. Worth a short exploration before reaching that phase.

### Docs
- [CVAT.ai - getting started](https://docs.cvat.ai/docs/getting-started/)
- [Video annotation guide](https://docs.cvat.ai/docs/manual/basics/create-annotation-job/)

### Key things to understand
- How to set up a video annotation task
- Timeline-based event labelling - marking the frame range of a kick event
- Exporting annotations in a usable format (COCO, CSV, or custom)

*Note: CVAT runs in the browser at cvat.ai - no installation needed to explore it.*

---

## Conventional Commits

The commit message specification used in this project. The full spec is short and worth reading once.

- [conventionalcommits.org - full specification](https://www.conventionalcommits.org/en/v1.0.0/)

### Types relevant to this project
| Type | When to use |
|---|---|
| `feat` | A new capability added to the pipeline |
| `fix` | A bug fix |
| `chore` | Tooling, config, maintenance - no production code change |
| `docs` | README, comments, docstrings |
| `refactor` | Code restructured without behaviour change |
| `test` | Adding or updating tests |
| `experiment` | Non-standard but useful for notebook/research commits |

---

## Ball Detection and Tracking

Needed for Phase 1 onwards. The ball is small, fast-moving, and subject to motion blur. No pretrained model exists with weights specifically trained on close-range training footage, so fine-tuning will be required at some point. The baseline candidates below are ordered by integration effort.

#### Worth a read:
- **DeepBall** - *DeepBall: Deep Neural-Network Ball Detector* - football specific ball detector. although it does not have a high chance of being used in the project, it is a great and easy enough read to understand some concepts of ball detection.
  - [arXiv:1902.07304](https://arxiv.org/abs/1902.07304)

### Baseline candidates

**Fine-tuned YOLO on football data** - fits the existing stack exactly, lowest friction. Roboflow Universe has pre-labelled football datasets in YOLO format ready to fine-tune with. Limitation: YOLO is single-frame and will struggle with motion blur and fast ball movement without temporal context.

**FootAndBall** - a dedicated football detector for ball and player detection, purpose-built for football broadcast footage. Uses a Feature Pyramid Network architecture to improve discriminability of small objects (the ball) by incorporating larger visual context. PyTorch-based, pretrained weights available. Limitation: trained on broadcast (long-shot, overhead) footage - expect degraded performance on close-range training drill angles without fine-tuning.
- [GitHub: jac99/FootAndBall](https://github.com/jac99/FootAndBall)
- [arXiv:1912.05445](https://arxiv.org/abs/1912.05445)

*Fine-tuned `yolo11n` has been chosen as the baseline model over `footandball` (see `notebooks/ball/ball_detector_baseline_shooting_drill.ipynb` for the full evaluation experiment and conclusion).*

### Sequence-based approaches (post-baseline)

**TrackNetV2** - designed specifically for small fast sports ball tracking using sequences of frames rather than single frames. More robust to motion blur than single-frame detectors. Originally trained on tennis and badminton.
- Search: *"TrackNetV2 football"* and *"BallTrack soccer"*

**TrackNetV4** - extends TrackNetV2 with motion attention maps that give the model explicit context of the ball's position in previous frames. TensorFlow-based, no pretrained football weights. The motion attention mechanism is conceptually simple enough to reconstruct on top of a different backbone at a later phase.
- [arXiv:2409.14543](https://arxiv.org/abs/2409.14543)
- [GitHub: TrackNetV4/TrackNetV4](https://github.com/TrackNetV4/TrackNetV4)

### Datasets
- [Roboflow Universe](https://universe.roboflow.com/) (search "soccer ball detection") - pre-labelled datasets in YOLO format

### Key concepts to understand
- Temporal ball tracking vs single-frame detection - why sequences matter for fast-moving objects
- Kalman filtering for smoothing noisy ball position estimates across frames (see also [Inference Smoothing](#inference-smoothing))
- Ball velocity estimation from position deltas across frames
- Spatial-temporal proximity of ball to foot keypoints as a kick detection signal

---

## Later - Park for Now

Relevant in later phases. To be returned to when needed.

- **Hungarian algorithm** - `scipy.optimize.linear_sum_assignment` - needed for player assignment in Phase 5. The Wikipedia page covers the conceptual model well.
- **PyTorch LSTM / 1D-CNN on sequences** - needed for Phase 7. Search: *"1D CNN time series classification PyTorch"*.
- **ONNX and model export** - relevant for real-time deployment in Phase 8. [onnxruntime.ai](https://onnxruntime.ai)
- **Google Colab + Google Drive pipeline** - setting up a clean data loading workflow from Drive for GPU training sessions. Worth reading before Phase 7.
