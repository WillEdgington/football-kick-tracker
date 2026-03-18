# Research Avenues

A living document of research material supporting development of the football kick tracker pipeline. Ordered by relevance to current phase. This will be revisited and expanded as the project progresses.

> Initially drafted with AI assistance and will be updated throughout the project.

---

## 1. YOLOv8-Pose - Ultralytics

The immediate pipeline dependency for Phase 1. Understanding the output format is required before any detection logic can be designed around it.

**Goal:** given a frame, what does the keypoint tensor look like and how does one index into it for the leg joints?

### Docs
- [Ultralytics YOLOv8 Pose - official docs](https://docs.ultralytics.com/tasks/pose/)
- [Keypoint output format reference](https://docs.ultralytics.com/reference/engine/results/)

### Key concepts to understand
- The 17-point COCO keypoint schema - which indices map to hips, knees, ankles, and feet
- Per-keypoint confidence scores and how to threshold them
- Difference between `Results.keypoints.xy` (pixel coords) and `Results.keypoints.xyn` (normalised)
- How to run inference on a video vs a single frame

### Practical task
Run quickstart inference on any video clip and print the raw keypoint tensor to the terminal. This is the most valuable first step before starting Phase 1.

---

## 2. Pose-Based Action Recognition

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

## 3. OpenCV - Video and Calibration

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

## 4. Human Pose Estimation - Background

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

## 5. Sports Analytics and Kick Detection - Applied Work

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

## 6. Annotation - CVAT

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

## 7. Conventional Commits

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

## 8. Later - Park for Now

Relevant in later phases. To be returned to when needed.

- **Hungarian algorithm** - `scipy.optimize.linear_sum_assignment` - needed for player assignment in Phase 5. The Wikipedia page covers the conceptual model well.
- **Kalman filtering** - useful for smoothing noisy pose detections and tracking players across frames. Search: *"Kalman filter object tracking OpenCV"*.
- **PyTorch LSTM / 1D-CNN on sequences** - needed for Phase 7. Search: *"1D CNN time series classification PyTorch"*.
- **ONNX and model export** - relevant for real-time deployment in Phase 8. [onnxruntime.ai](https://onnxruntime.ai)
- **Google Colab + Google Drive pipeline** - setting up a clean data loading workflow from Drive for GPU training sessions. Worth reading before Phase 7.
