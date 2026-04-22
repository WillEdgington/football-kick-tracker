# Football Kick Tracker

A computer vision pipeline for detecting and tracking kicks performed by football players. Built as a non-invasive, automated alternative for logging kick frequency in training sessions.

---

## Pipeline Components

### Pose Estimator

`yolo11l-pose`. Benchmarked against YOLO11m/s/n and YOLO26m (see `notebooks/pose/YOLO_candidate_comparison.ipynb`). Some failure modes at some problem specific camera angles (see `notebooks/pose/YOLO_pose_shooting_drills.ipynb`).

**Planned:**
- Problem specific labelled data for fine-tuning and proper evaluation.
- Inference smoothing.
- Candidate evaluation: RTMPose, ViTPose

### Ball Tracker

Fine-tuned `yolo11n` single-frame detector, trained on the Roboflow `football-ball-detection` dataset (v4). Chosen over `footandball` as the baseline (see `notebooks/pose/ball_detector_baseline_shooting_drills.ipynb`).

**Planned:**
- Close-range labelled data for fine-tuning and proper evaluation
- Inference smoothing
- Sequence-based tracking with motion attention maps for robustness against motion blur and track loss

### Kick Detector

Not yet implemented.

**Planned:**
- Rule-based detector using pose and ball signal (ankle velocity, foot-to-ball proximity, ball velocity change)
- Ground-truth dataset construction from labelled footage
- Learned detector trained on ground-truth dataset

### Player Assignment

Not yet implemented.

**Planned:**
- GPS wearable data for player-to-kick attribution
- Computer vision fallback

### Multi-Camera Support

Not yet implemented. Single-camera only.

**Planned:**
- Extend pipeline to handle multiple simultaneous camera feeds
- Duplicate kick removal across views

### Session Output

Not yet implemented.

**Planned:**
- Per-player kick count
- Kick velocity
- Per-session summary report

---

## Project Status

This section serves as a living development log.

### Current Focus
- **Data labelling:** annotating existing shooting drill footage in CVAT. Ball bounding boxes and pose keypoints. This will define the ground truth format for future fine-tuning and evaluation.
- **Data collection:** searching for and collecting vaired problem specific footage across different environments, angles, balls and people to address the current data distribution problem.
- **Kick detection logic:** beginning design of the rule-based kick detector using existing pose output.

### Recently Completed
<!-- Latest first. Maximum 10 items. Older entries belong in the git log. -->
- `experiment/preannotation-tool-notebooks`: created `notebooks/tools/preannotate_pose_script.ipynb`, `notebooks/tools/preannotate_ball_script.ipynb` notebooks to provide a way to run `tools/preannotate_pose.py`, `tools/preannotate_ball.py` on a non-local kernel (i.e. Colab GPU).
- `feat/preannotation-CLI-root-arg`: created `resolvePath` (`utils/io.py`) method for resolving a path from a given `root` and `path` input. Refactored `tools/preannotate_pose.py`, `tools/preannotate_ball.py` CLI scripts with addition of `--root` input argument to allow for user to input a custom project root directory rather than the default assumption (`.`).
- `feat/preannotation-batch-processing`: `getAllVideoPaths` (`utils/video.py`) method for getting video paths from a directory, `batchCVATYOLOPosePreannotation` (`pose/preannotate.py`) method for batch CVAT pose pre-annotation, `batchCVATYOLOBallPreannotation` (`ball/preannotate.py`) method for batch CVAT ball pre-annotation. Refactored `tools/preannotate_pose.py`, `tools/preannotate_ball.py` CLI scripts for batch processing.
- `feat/cvat-preannotation-tooling`: created methods to convert raw model inference into CVAT compatible XML for ball (`ball/cvat.py`) and pose (`pose/cvat.py`) models with test coverage. Created CLI scripts for the pre-annotation pipeline for ball (`tools/preannotate_ball.py`) and pose (`tools/preannotate_pose.py) YOLO-based models.
- `feat/common-utils-overhaul`: `saveText` method (in `utils/io.py`) for saving a string object to a file, `getVideoInfo` method (in `utils/video.py`) that returns the common video metadata info, `loadYOLOModel` method (in `utils/yolo.py`) for loading YOLO models. All new utility methods have full test coverage.
- `feat/raw-inference-pipeline`: renamed test scripts from test_{module}.py to test_{library}_{module}.py to avoid collision errors. expanded pose constants to include more COCO keypoints (face, upper, all, body). Created YOLO compatible raw video inference methods for pose and ball models (this included the creation of `ball/inference.py`). Full test coverage for new methods.
- `chore/docs-restructuring`: `/docs` directory created for any secondary documentation, `research.md` renamed to `RESEARCH.md` and relocated to inside `docs/`. Checked for any internal links that may be affected from restructure, found None.
- `docs/update-readme-phases`: old "Project Phase" section of README replaced with new "[Pipeline Components](#pipeline-components)" section. Has a more clear direction as to what needs to be built for each component of the project. Updates to `research.md` to replace phase references and remove commitizen section.
- `experiment/ball-detection-baseline`: evaluated a fine-tuned `yolo11n` and pre-trained `footandball` model for ball tracking on high quality shooting drill footage. Concluded that the fine-tuned `yolo11n` was the better baseline model.
- `chore/update-build-system`: Integrated GitHub Actions (CI), automated dependency management via pyproject.toml, and established a Branch & PR development protocol.

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

📧 [**willedge037@gmail.com**](mailto:willedge037@gmail.com) &nbsp;|&nbsp; 🔗 [**LinkedIn**](https://www.linkedin.com/in/williamedgington/)

## License

This project is licensed under the [GPL-3.0 License](LICENSE).
