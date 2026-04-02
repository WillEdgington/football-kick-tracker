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
- `chore/update-build-system`: Integrated GitHub Actions (CI), automated dependency management via pyproject.toml, and established a Branch & PR development protocol.
- Update to research.md to include Inference Smoothing, ViTPose, RTMPose sections and updated the Ball Tracking section
- YOLO pose model tested on high quality footage of shooting training drill - see `notebooks/pose/YOLO_pose_shooting_drills.ipynb`
- Refactored relevant methods from `notebooks/pose/YOLO_candidate_comparison.ipynb` into the main repo with full test coverage (methods in: `pose/annotate.py`, `pose/inference.py`, `utils/io.py`, `utils/metrics.py`)
- YOLO pose model comparison (YOLO11(m)(s)(l)(n) vs YOLO26m) - see `notebooks/pose/YOLO_candidate_comparison.ipynb`
- `pose/visualise.py` - `drawKeypoints` with full test coverage
- Initial two-model comparison (YOLO11m vs YOLO26m) - see `notebooks/pose/2D_pose_model_comparison.ipynb`
- Repo scaffolding - pre-commit hooks, commitizen, pytest, ruff, black

---

## Development Setup

### 1.Environment & Dependencies

This project requires **Python 3.10**. Using a virtual environment is highly recommended.
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install project and dependencies
# The '-e' flag installs the project in 'editable' mode
pip install -e .
pip install -r requirements-dev.txt # install dev dependencies
```
*Note: Runtime dependencies in `requirements.txt` are handled automatically by the editable install.*

### 2. Enable Quality Automation

We use `pre-commit` to ensure code remains clean and follows professional standards before it reaches the repository.
```bash
pre-commit install
pre-commit install --hook-type commit-msg # activate Commitizen checks
```

### 3. Workflow: Branch & Pull Request

All development should occur in descriptively named branches to keep the `main` branch stable.
  1. **Create a branch:** `git checkout -b <type>/<description>` (e.g. `feature/yolo-pose-pipeline` or `experiment/ball-tracking-comparison`).
  2. **Commit:** Ensure your message follows the [Commitizen](#4-commit-prefixes) format. Run `pytest` locally before committing code changes.
  3. **Sync:** Before finishing, pull the latest changes from `main` to ensure your branch is up-to-date and resolve any conflicts.
  4. **Push & PR:** `git push -u origin <branch name>`. Open a **Pull Request** on GitHub.
  5. **CI Check:** Automated **GitHub Actions** will run `Ruff`, `Black`, and `PyTest` on your PR. All checks must pass before merging.

### 4. Commit Prefixes

We follow the **Conventional Commits** standard using `Commitizen`. Every commit must start with a lowercase prefix. We encourage using bracketed scopes to specify the module being updated.

|Prefix|Use Case|Example|
|---|---|---|
|`feat:`|A new feature|`feat: (pose) add keypoint confidence filtering`|
|`fix:`|Fixing a bug|`fix: (utils) fix to normMethod in metrics.py`|
|`docs:`|Documentation updates|`docs: add dev setup to README.md`|
|`experiment:`|**(Custom)** Notebook research|`experiment: evaluate yolo11l-pose on shooting footage`|
|`refactor:`|Code structural changes|`refactor: (ball) optimise inference loop efficiency`|
|`chore:`|Maintenance (CI/CD, dependencies, build configs)|`chore: add ultralytics to requirements.txt`|
|`test:`|Adding or updating `pytest` files|`test: (pose) add tests for annotate.py`|

*For more details, see the [Conventional Commits Specification](https://www.conventionalcommits.org/en/v1.0.0/) and [Pre-commit Documentation](https://pre-commit.com/).*

### 5. Notebook Best Practices

- **Isolation:** Conduct all notebook research on an `experiment/` branch. This branch should primarily contain `experiment:` (logic/testing) and `docs:` (findings/observations) commits.

- **Clean State:** Clear all cell outputs before committing. This keeps the repository size manageable and prevents Git from tracking non-essential metadata. Observations and decisions should be documented in Markdown cells.

- **Refactoring:** Once an experiment is successful, refactor the core logic into the main library (`.py` modules). This should be done on a fresh `feature/` or `refactor/` branch to ensure full test coverage and architectural alignment.

---

## Author

Created by [**WillEdgington**](https://github.com/WillEdgington)

📧 [willedge037@gmail.com](mailto:willedge037@gmail.com)

🔗 [LinkedIn](https://www.linkedin.com/in/williamedgington/)

## License

This project is licensed under the [GPL-3.0 License](LICENSE).
