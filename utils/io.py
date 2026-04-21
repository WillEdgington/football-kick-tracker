import json
import os
from pathlib import Path
from typing import Dict, List


def saveJSON(data: List[Dict] | Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmpPath = path.with_suffix(".tmp")
    try:
        with open(tmpPath, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmpPath, path)
    except Exception as e:
        if tmpPath.exists():
            tmpPath.unlink()
        raise e


def loadJSON(path: Path) -> List[Dict] | Dict | None:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def saveText(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmpPath = path.with_suffix(".tmp")
    try:
        with open(tmpPath, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmpPath, path)
    except Exception as e:
        if tmpPath.exists():
            tmpPath.unlink()
        raise e


def resolvePath(root: str | Path, path: str | Path) -> Path:
    path = path if isinstance(path, Path) else Path(path)
    if path.is_absolute() or (isinstance(root, str) and root in {"", "."}):
        return path
    root = root if isinstance(root, Path) else Path(root)

    return root / path
