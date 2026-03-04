from __future__ import annotations

import io
import math
from typing import Dict, Tuple

import mediapipe as mp
import numpy as np
from PIL import Image

from .validators import ProcessingError

FACE_RATIO_MIN = 0.50
FACE_RATIO_MAX = 0.80
CHIN_TO_BOTTOM_MIN = 0.08
CHIN_TO_BOTTOM_MAX = 0.22
TARGETS = {
    "id": (492, 633),
    "passport": (781, 1004),
}
MAX_OUTPUT_BYTES = 2_500_000


def adjust_zoom(image: Image.Image) -> Tuple[Image.Image, Dict[str, float]]:
    rgb = np.array(image.convert("RGB"))
    landmarks = _detect_landmarks(rgb)
    if landmarks is None:
        raise ProcessingError(code="FACE_NOT_DETECTED", message="Nie wykryto twarzy.")

    height, width = rgb.shape[:2]
    head_top_y, chin_y, eye_y = _extract_vertical_refs(landmarks, height)

    face_height = chin_y - head_top_y
    if face_height <= 0:
        raise ProcessingError(code="FACE_GEOMETRY_INVALID", message="Nieprawidłowa geometria twarzy.")

    d_eye_chin = chin_y - eye_y
    if d_eye_chin <= 0:
        raise ProcessingError(code="FACE_GEOMETRY_INVALID", message="Nieprawidłowa geometria twarzy.")

    face_ratio_target = 0.70
    down_attempts = 0
    up_attempts = 0
    max_down = 20
    max_up = 10
    s = FACE_RATIO_MIN * height / face_height

    while True:
        if face_ratio_target < FACE_RATIO_MIN or face_ratio_target > FACE_RATIO_MAX:
            raise ProcessingError(
                code="ZOOM_CONSTRAINTS_UNSATISFIABLE",
                message="Nie można dopasować powiększenia do zadanych warunków.",
            )

        s = face_ratio_target * height / face_height
        scaled = image.resize((int(width * s), int(height * s)), Image.LANCZOS)
        scaled_w, scaled_h = scaled.size
        eye_y_scaled = eye_y * s

        offset_y = int(round(0.5 * height - eye_y_scaled))
        offset_x = int(round((width - scaled_w) / 2))

        chin_y_scaled = chin_y * s + offset_y
        face_height_scaled = face_height * s
        face_ratio = face_height_scaled / height
        chin_to_bottom_ratio = (height - chin_y_scaled) / height

        if chin_to_bottom_ratio < CHIN_TO_BOTTOM_MIN:
            down_attempts += 1
            if down_attempts > max_down:
                raise ProcessingError(
                    code="ZOOM_CONSTRAINTS_UNSATISFIABLE",
                    message="Nie można dopasować powiększenia do zadanych warunków.",
                )
            face_ratio_target = round(face_ratio_target - 0.01, 2)
            continue
        if chin_to_bottom_ratio > CHIN_TO_BOTTOM_MAX:
            up_attempts += 1
            if up_attempts > max_up:
                raise ProcessingError(
                    code="ZOOM_CONSTRAINTS_UNSATISFIABLE",
                    message="Nie można dopasować powiększenia do zadanych warunków.",
                )
            face_ratio_target = round(face_ratio_target + 0.01, 2)
            continue

        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(scaled, (offset_x, offset_y))
        return canvas, {
            "scale": float(s),
            "face_ratio": float(face_ratio),
            "chin_to_bottom_ratio": float(chin_to_bottom_ratio),
            "face_ratio_target": float(face_ratio_target),
        }

    raise ProcessingError(
        code="ZOOM_CONSTRAINTS_UNSATISFIABLE",
        message="Nie można dopasować powiększenia do zadanych warunków.",
    )


def finalize_output(image: Image.Image, preset: str) -> bytes:
    if preset not in TARGETS:
        raise ProcessingError(code="INVALID_PRESET", message="Nieznany preset.")
    min_w, min_h = TARGETS[preset]
    ratio = min_w / min_h

    current_w, current_h = image.size
    current_ratio = current_w / current_h
    if abs(current_ratio - ratio) > 0.01:
        target_h = int(round(current_w / ratio))
        if target_h <= 0:
            target_h = min_h
            target_w = int(round(target_h * ratio))
        else:
            target_w = int(round(target_h * ratio))
        image = image.resize((target_w, target_h), Image.LANCZOS)
        current_w, current_h = image.size

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95, optimize=True)
    data = buf.getvalue()
    if len(data) <= MAX_OUTPUT_BYTES:
        return data

    resized = image
    w, h = current_w, current_h
    while len(data) > MAX_OUTPUT_BYTES and (w > min_w or h > min_h):
        scale = max(0.85, (MAX_OUTPUT_BYTES / len(data)) ** 0.5 * 0.98)
        w = max(min_w, int(round(w * scale)))
        h = max(min_h, int(round(w / ratio)))
        resized = resized.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=92, optimize=True)
        data = buf.getvalue()

    if len(data) <= MAX_OUTPUT_BYTES:
        return data

    for quality in range(90, 69, -4):
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= MAX_OUTPUT_BYTES:
            return data

    raise ProcessingError(
        code="OUTPUT_FILE_TOO_LARGE",
        message="Nie udało się zmieścić pliku w limicie 2,5 MB.",
    )


def _detect_landmarks(rgb: np.ndarray) -> np.ndarray | None:
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(
        static_image_mode=True, refine_landmarks=True, max_num_faces=1
    ) as mesh:
        results = mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        face = results.multi_face_landmarks[0]
        return np.array([(lm.x, lm.y) for lm in face.landmark], dtype=np.float32)


def _extract_vertical_refs(landmarks: np.ndarray, height: int) -> Tuple[float, float, float]:
    head_indices = [10, 9, 67, 109, 338, 297, 151]
    ys = landmarks[:, 1] * height
    head_top_y = float(ys[head_indices].min())
    chin_y = float(ys[152])
    left_eye_y = float(ys[[33, 133]].mean())
    right_eye_y = float(ys[[362, 263]].mean())
    eye_y = (left_eye_y + right_eye_y) / 2
    return head_top_y, chin_y, eye_y
