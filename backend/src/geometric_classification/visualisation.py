"""Diagnostic visualisations for the isolated Module 3 workflow."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cloudchamber-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def diagnostic_figure(record: dict[str, Any]):
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    image = np.asarray(record["image"])
    annotated = image.copy()
    rectangle = record["rectangle"]
    cv2.rectangle(annotated, (rectangle["x"], rectangle["y"]), (rectangle["x2"], rectangle["y2"]), (0, 255, 0), 2)
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Original annotation")
    axes[0, 1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)); axes[0, 1].set_title("Detected rectangle")
    axes[0, 2].imshow(cv2.cvtColor(record["roi"], cv2.COLOR_BGR2RGB)); axes[0, 2].set_title("Interior ROI")
    axes[1, 0].imshow(record["mask"], cmap="gray"); axes[1, 0].set_title("Temporary track mask")
    axes[1, 1].imshow(record["skeleton"], cmap="gray"); axes[1, 1].set_title("Skeleton")
    axes[1, 2].imshow(record["roi"][:, :, ::-1])
    path = np.asarray(record["principal_path"])
    axes[1, 2].plot(path[:, 1], path[:, 0], color="cyan", linewidth=1.5)
    axes[1, 2].scatter(path[[0, -1], 1], path[[0, -1], 0], c=["lime", "yellow"], s=35)
    axes[1, 2].set_title("Principal path and endpoints")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    return figure


def save_diagnostic_figure(record: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = diagnostic_figure(record)
    figure.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(figure)
