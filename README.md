# Football Kick Tracker

A computer vision pipeline for detecting and tracking kicks performed by football players. Built as a non-invasive, automated alternative for logging kick frequency in training sessions.

---

## Project Phases

### Phase 1 - Single-Camera Kick Detector
2D pose estimation and rule-based kick detection on single-camera footage. Single ball, single player kick count.

### Phase 2 - Multi-Camera Kick Detector
Extend to 3D pose estimation across multiple cameras. Rule-based kick detection with duplicate kick removal across views.

### Phase 3 - Multiple Ball Detection
Handle real training session conditions with multiple balls in frame simultaneously.

### Phase 4 - Dataset Construction
Use the rule-based pipeline to generate candidate kick events from footage. Manually verify and label events to build a ground-truth dataset for model training.

### Phase 5 - Player Assignment
Assign detected kicks to individual players using IMU wearable data and/or computer vision methods.

### Phase 6 - Session Output
Attribute kicks to players and produce a per-session kick frequency table.

### Phase 7 - Train / Fine-Tune Pipeline
Use the labelled dataset to train a lightweight neural network for kick detection, replacing or supplementing the rule-based approach.

### Phase 8 - Real-Time Tracker
Extend the pipeline to support real-time kick detection and live session feedback.

### Phase 9 - Extended Metrics
Track additional metrics such as kick velocity and intensity classification (soft / medium / hard) to build a richer picture of training load and player output.

---

## Project Status

This section serves as a living development log.

### Current Focus
- **Phase 1 - Single-camera kick detector:** YOLO candidate model evaluation across multiple training drill clips

### Recently Completed
<!-- Latest first. Maximum 10 items. Older entries belong in the git log. -->
- Refactored relevant methods from `notebooks/YOLO_candidate_comparison.ipynb` into the main repo with full test coverage (methods in: `pose/annotate.py`, `pose/inference.py`, `utils/io.py`, `utils/metrics.py`)
- YOLO pose model comparison (YOLO11(m)(s)(l)(n) vs YOLO26m) - see `notebooks/YOLO_candidate_comparison.ipynb`
- `pose/visualise.py` - `drawKeypoints` with full test coverage
- Initial two-model comparison (YOLO11m vs YOLO26m) - see `notebooks/2D_pose_model_comparison.ipynb`
- Repo scaffolding - pre-commit hooks, commitizen, pytest, ruff, black

---

## Author

Created by [**WillEdgington**](https://github.com/WillEdgington)

📧 [willedge037@gmail.com](mailto:willedge037@gmail.com)

🔗 [LinkedIn](https://www.linkedin.com/in/williamedgington/)

## License

This project is licensed under the [GPL-3.0 License](LICENSE).
