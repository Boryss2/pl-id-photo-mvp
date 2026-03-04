from pathlib import Path
import json
import re

import cv2
import numpy as np
from PIL import Image

from backend.app.pipeline import process_image_with_diagnostics
from backend.app.postprocess import adjust_zoom, finalize_output
from backend.app.validators import ProcessingError

input_dir = Path("input")


def _next_output_dir(base: Path) -> Path:
    existing = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("output_")]
    max_index = 0
    for folder in existing:
        match = re.fullmatch(r"output_(\d+)", folder.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return base / f"output_{max_index + 1}"


def _write_diagnostics(
    input_path: Path, output_path: Path, out_dir: Path, diagnostics: dict
) -> None:
    inp = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    out = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    if inp is None or out is None:
        return

    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    input_hist = _histogram(inp)
    output_hist = _histogram(out)
    hist_path = diag_dir / f"{output_path.stem}_hist.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"input": input_hist, "output": output_hist},
            f,
            indent=2,
        )

    input_stats = _lab_stats(inp)
    output_stats = _lab_stats(out)
    stats_path = diag_dir / f"{output_path.stem}_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input": input_stats,
                "output": output_stats,
                "pipeline": diagnostics,
            },
            f,
            indent=2,
        )


def _histogram(img: np.ndarray) -> dict:
    hist = {}
    clipped = {}
    for idx, channel in enumerate(["b", "g", "r"]):
        h = cv2.calcHist([img], [idx], None, [256], [0, 256]).flatten()
        hist[channel] = h.astype(int).tolist()
        clipped[channel] = {
            "0": float(h[0]) / float(h.sum()),
            "255": float(h[-1]) / float(h.sum()),
        }
    return {"histogram": hist, "clipped_ratio": clipped}


def _lab_stats(img: np.ndarray) -> dict:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    x0, x1 = int(w * 0.3), int(w * 0.7)
    y0, y1 = int(h * 0.25), int(h * 0.7)
    roi = lab[y0:y1, x0:x1]
    means = roi.reshape(-1, 3).mean(axis=0)
    stds = roi.reshape(-1, 3).std(axis=0)
    return {
        "mean": {"L": float(means[0]), "A": float(means[1]), "B": float(means[2])},
        "std": {"L": float(stds[0]), "A": float(stds[1]), "B": float(stds[2])},
    }


output_dir = _next_output_dir(Path("."))
output_dir.mkdir(exist_ok=False)

for path in input_dir.iterdir():
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    for preset in ["id", "passport"]:
        out_path = output_dir / f"{path.stem}_{preset}.jpg"
        try:
            data = path.read_bytes()
            result, diagnostics = process_image_with_diagnostics(data, preset)
            out_path.write_bytes(result)
            _write_diagnostics(path, out_path, output_dir, diagnostics)
            try:
                zoomed_dir = output_dir / "zoomed"
                zoomed_dir.mkdir(exist_ok=True)
                zoomed_path = zoomed_dir / f"{out_path.stem}_zoom.jpg"
                zoomed_img, zoom_stats = adjust_zoom(
                    Image.open(out_path).convert("RGB")
                )
                zoomed_bytes = finalize_output(zoomed_img, preset)
                zoomed_path.write_bytes(zoomed_bytes)
                _ = zoom_stats
            except Exception as err:
                print(f"ZOOM ERR {out_path.name}: {err}")
        except ProcessingError as err:
            print(f"ERR {path.name} preset={preset} {err.code}: {err.message}")
        except Exception as err:
            print(f"ERR {path.name} preset={preset} UNEXPECTED: {err}")
