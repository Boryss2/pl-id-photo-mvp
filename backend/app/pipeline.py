import io
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

from .validators import (
    ProcessingError,
    ensure_chin_to_bottom,
    ensure_face_detected,
    ensure_face_ratio,
    ensure_output_file_size,
    ensure_output_ratio,
    ensure_output_resolution,
)

TARGETS = {
    "id": (492, 633),
    "passport": (781, 1004),
}

MAX_OUTPUT_BYTES = 2_500_000
FACE_RATIO_TARGET = 0.7
FACE_RATIO_MIN = 0.50
FACE_RATIO_MAX = 0.80
CHIN_TO_BOTTOM_MIN = 0.08
CHIN_TO_BOTTOM_MAX = 0.22
MIN_FACE_WIDTH_RATIO = 1.05
BACKGROUND_WHITE = (248, 248, 248)
LUMA_RANGE_LOW = 95.0
LUMA_RANGE_HIGH = 155.0
LUMA_OFFSET_MAX = 6.0
CHROMA_DELTA_WARN = 6.0
BG_STD_WARN = 6.0


def process_image(content: bytes, preset: str) -> bytes:
    output, _ = process_image_with_diagnostics(content, preset)
    return output


def process_image_variants(content: bytes, preset: str) -> dict:
    if preset not in TARGETS:
        raise ProcessingError(code="INVALID_PRESET", message="Nieznany preset.")

    pil = _load_image(content)
    rgb = np.array(pil)

    landmarks = _detect_landmarks(rgb)
    ensure_face_detected(landmarks is not None)

    zoom_crop, zoom_face_ratio, _ = _crop_to_spec(
        rgb, landmarks, use_mask_bounds=False, use_shoulders=False
    )
    ensure_face_ratio(zoom_face_ratio)

    wide_crop, wide_face_ratio = _crop_wide(rgb, landmarks)

    zoom_bytes = _process_crop(zoom_crop, preset)
    wide_bytes = _process_crop(wide_crop, preset)

    return {
        "zoom": {"bytes": zoom_bytes, "face_ratio": zoom_face_ratio},
        "wide": {"bytes": wide_bytes, "face_ratio": wide_face_ratio},
    }


def process_image_with_diagnostics(content: bytes, preset: str) -> tuple[bytes, dict]:
    if preset not in TARGETS:
        raise ProcessingError(code="INVALID_PRESET", message="Nieznany preset.")

    pil = _load_image(content)
    rgb = np.array(pil)

    landmarks = _detect_landmarks(rgb)
    ensure_face_detected(landmarks is not None)
    crop, face_ratio, allow_low_face_ratio = _crop_to_spec(rgb, landmarks)
    if not allow_low_face_ratio:
        ensure_face_ratio(face_ratio)

    cutout, fg_mask = _remove_background(crop)
    corrected = _luminance_match(cutout, crop, fg_mask)
    enhanced = _denoise_sharpen(corrected)

    height, width = enhanced.shape[:2]
    ensure_output_ratio(width, height)
    fg_mask_resized = _resize_mask(fg_mask, enhanced.shape[:2])
    diagnostics = _qa_metrics(crop, enhanced, fg_mask_resized)
    _warn_if_chroma_shifted(diagnostics)

    output_bytes = _encode_jpeg_high_quality(enhanced)
    return output_bytes, diagnostics


def _load_image(content: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(content))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


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


def _crop_to_spec(
    rgb: np.ndarray,
    landmarks: np.ndarray,
    use_mask_bounds: bool = True,
    use_shoulders: bool = True,
) -> Tuple[np.ndarray, float, bool]:
    height, width = rgb.shape[:2]
    xs = landmarks[:, 0] * width
    ys = landmarks[:, 1] * height

    min_x, max_x = xs.min(), xs.max()
    face_left = float(min_x)
    face_right = float(max_x)
    face_width = face_right - face_left

    top_idx = 10
    bottom_idx = 152
    top_y = float(ys[top_idx])
    bottom_y = float(ys[bottom_idx])
    face_height = bottom_y - top_y

    left_eye = _avg_points(xs, ys, [33, 133])
    right_eye = _avg_points(xs, ys, [362, 263])
    eyes_center_x = (left_eye[0] + right_eye[0]) / 2
    eyes_center_y = (left_eye[1] + right_eye[1]) / 2
    shoulders = _detect_pose_shoulders(rgb) if use_shoulders else None
    mask_bounds = _mask_bounds(rgb) if use_mask_bounds else None
    allow_low_face_ratio = False

    face_ratio_target = FACE_RATIO_TARGET
    max_iters = 20
    for _ in range(max_iters):
        crop_height = face_height / face_ratio_target
        top = eyes_center_y - 0.5 * crop_height
        bottom = top + crop_height
        chin_to_bottom = bottom - bottom_y
        chin_to_bottom_ratio = chin_to_bottom / crop_height

        if chin_to_bottom_ratio < CHIN_TO_BOTTOM_MIN:
            face_ratio_target -= 0.01
        elif chin_to_bottom_ratio > CHIN_TO_BOTTOM_MAX:
            face_ratio_target += 0.01
        else:
            break

    if face_ratio_target < FACE_RATIO_MIN or face_ratio_target > FACE_RATIO_MAX:
        raise ProcessingError(
            code="FACE_RATIO_TARGET_INVALID",
            message="Nieprawidłowa proporcja twarzy do kadru.",
        )

    crop_height = face_height / face_ratio_target
    top = eyes_center_y - 0.5 * crop_height
    bottom = top + crop_height
    chin_to_bottom = bottom - bottom_y
    chin_to_bottom_ratio = chin_to_bottom / crop_height

    crop_width = crop_height * (35 / 45)
    min_crop_width = face_width * MIN_FACE_WIDTH_RATIO
    if shoulders:
        shoulder_span = shoulders["right_x"] - shoulders["left_x"]
        min_crop_width = max(min_crop_width, shoulder_span * 1.05)
        shoulder_margin = shoulder_span * 0.1
        min_crop_width = max(min_crop_width, shoulder_span + 2 * shoulder_margin)
    if mask_bounds:
        mask_span = mask_bounds["right_x"] - mask_bounds["left_x"]
        mask_margin = mask_span * 0.05
        min_crop_width = max(min_crop_width, mask_span + 2 * mask_margin)
    if crop_width < min_crop_width:
        crop_width = min_crop_width
        crop_height = crop_width * (45 / 35)
        top = eyes_center_y - 0.5 * crop_height
        bottom = top + crop_height
        chin_to_bottom = bottom - bottom_y
        chin_to_bottom_ratio = chin_to_bottom / crop_height
        if shoulders or mask_bounds:
            allow_low_face_ratio = True

    face_ratio = face_height / crop_height
    if face_ratio < FACE_RATIO_MIN:
        allow_low_face_ratio = True

    top = eyes_center_y - 0.5 * crop_height
    center_x = eyes_center_x
    if shoulders:
        center_x = shoulders["center_x"]
    left = center_x - crop_width / 2
    right = left + crop_width
    if shoulders:
        shoulder_span = shoulders["right_x"] - shoulders["left_x"]
        margin = shoulder_span * 0.1
        min_left = shoulders["left_x"] - margin
        min_right = shoulders["right_x"] + margin
        shoulder_width = min_right - min_left
        if crop_width < shoulder_width:
            crop_width = shoulder_width
            crop_height = crop_width * (45 / 35)
            top = eyes_center_y - 0.5 * crop_height
            bottom = top + crop_height
        left = min_left
        right = left + crop_width
    if mask_bounds:
        mask_span = mask_bounds["right_x"] - mask_bounds["left_x"]
        mask_margin = mask_span * 0.05
        min_left = mask_bounds["left_x"] - mask_margin
        min_right = mask_bounds["right_x"] + mask_margin
        mask_width = min_right - min_left
        if crop_width < mask_width:
            crop_width = mask_width
            crop_height = crop_width * (45 / 35)
            top = eyes_center_y - 0.5 * crop_height
            bottom = top + crop_height
        left = min_left
        right = left + crop_width
    bottom = top + crop_height

    pad_top = max(0, int(-top))
    padded, left, top, right, bottom = _pad_if_needed(
        rgb, left, top, right, bottom
    )
    crop = padded[int(top) : int(bottom), int(left) : int(right)]
    face_ratio = face_height / crop_height

    eyes_offset = (eyes_center_y + pad_top - top) / crop_height
    if abs(eyes_offset - 0.5) > 0.01:
        raise ProcessingError(
            code="EYES_NOT_CENTERED",
            message="Linia oczu nie znajduje się w połowie wysokości zdjęcia.",
        )
    if not allow_low_face_ratio:
        ensure_chin_to_bottom(chin_to_bottom_ratio)

    ratio = crop_width / crop_height
    if abs(ratio - (35 / 45)) > 0.01:
        raise ProcessingError(
            code="CROP_RATIO_INVALID",
            message="Nieprawidłowe proporcje kadru.",
        )

    return crop, float(face_ratio), allow_low_face_ratio


def _avg_points(xs: np.ndarray, ys: np.ndarray, indices: list[int]) -> Tuple[float, float]:
    return float(xs[indices].mean()), float(ys[indices].mean())


def _detect_pose_shoulders(rgb: np.ndarray) -> Optional[dict]:
    height, width = rgb.shape[:2]
    mp_pose = mp.solutions.pose
    try:
        with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            results = pose.process(rgb)
            if not results.pose_landmarks:
                return None
            landmarks = results.pose_landmarks.landmark
            left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            if left.visibility < 0.5 or right.visibility < 0.5:
                return None
            left_x = left.x * width
            right_x = right.x * width
            if right_x <= left_x:
                return None
            return {
                "left_x": float(left_x),
                "right_x": float(right_x),
                "center_x": float((left_x + right_x) / 2),
            }
    except Exception:
        return None


def _mask_bounds(rgb: np.ndarray) -> Optional[dict]:
    try:
        from rembg import remove
    except Exception:
        return None

    try:
        pil = Image.fromarray(rgb)
        cutout = remove(pil)
        if cutout.mode != "RGBA":
            cutout = cutout.convert("RGBA")
        alpha = np.array(cutout.split()[3]).astype(np.float32) / 255.0
        mask = alpha > 0.25
        if not np.any(mask):
            return None
        ys, xs = np.where(mask)
        left_x = float(xs.min())
        right_x = float(xs.max())
        return {"left_x": left_x, "right_x": right_x}
    except Exception:
        return None


def _pad_if_needed(
    rgb: np.ndarray, left: float, top: float, right: float, bottom: float
) -> Tuple[np.ndarray, float, float, float, float]:
    height, width = rgb.shape[:2]
    pad_left = max(0, int(-left))
    pad_top = max(0, int(-top))
    pad_right = max(0, int(right - width))
    pad_bottom = max(0, int(bottom - height))

    if pad_left or pad_top or pad_right or pad_bottom:
        rgb = cv2.copyMakeBorder(
            rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
    return rgb, left, top, right, bottom


def _remove_background(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from rembg import remove
    except Exception:
        return _remove_background_grabcut(rgb)

    try:
        pil = Image.fromarray(rgb)
        cutout = remove(pil)
        if cutout.mode != "RGBA":
            cutout = cutout.convert("RGBA")
        alpha = np.array(cutout.split()[3]).astype(np.float32) / 255.0
        alpha = _feather_alpha(alpha, 0.8)
        rgb = np.array(cutout.convert("RGB")).astype(np.float32)
        white = np.full_like(rgb, BACKGROUND_WHITE, dtype=np.float32)
        composite = rgb * alpha[..., None] + white * (1 - alpha[..., None])
        composite = _edge_darkening(composite, alpha, strength=0.025)
        return np.clip(composite, 0, 255).astype(np.uint8), alpha
    except Exception:
        return _remove_background_grabcut(rgb)


def _remove_background_grabcut(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    margin = int(min(height, width) * 0.05)
    rect = (margin, margin, width - 2 * margin, height - 2 * margin)

    mask = np.zeros((height, width), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(rgb, mask, rect, bg_model, fg_model, 3, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0
    ).astype(np.float32)

    fg_mask = _feather_alpha(fg_mask, 0.8)
    white = np.full_like(rgb, BACKGROUND_WHITE, dtype=np.uint8)
    composite = rgb * fg_mask[..., None] + white * (1 - fg_mask[..., None])
    composite = _edge_darkening(composite, fg_mask, strength=0.025)
    return np.clip(composite, 0, 255).astype(np.uint8), fg_mask


def _luminance_match(
    rgb: np.ndarray, reference: np.ndarray, fg_mask: np.ndarray
) -> np.ndarray:
    ref_stats = _lab_roi_stats(reference)
    ref_l = ref_stats["mean_l"]
    if LUMA_RANGE_LOW <= ref_l <= LUMA_RANGE_HIGH:
        return rgb

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    l_norm = np.clip(l / 255.0, 0.0, 1.0)

    out_stats = _lab_roi_stats(rgb)
    out_l = out_stats["mean_l"]
    target_l = float(np.clip(out_l, LUMA_RANGE_LOW, LUMA_RANGE_HIGH))
    delta = np.clip(target_l - out_l, -LUMA_OFFSET_MAX, LUMA_OFFSET_MAX)

    mid_weight = 1.0 - 4.0 * (l_norm - 0.5) ** 2
    mid_weight = np.clip(mid_weight, 0.0, 1.0)
    l_corrected = np.clip(l + delta * mid_weight, 0, 255)

    w = _smoothstep(fg_mask, 0.4, 0.9)
    l = l * (1 - w) + l_corrected * w
    merged = cv2.merge([l, a, b]).astype(np.uint8)
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def _denoise_sharpen(rgb: np.ndarray) -> np.ndarray:
    return rgb


def _feather_alpha(alpha: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.clip(alpha, 0.0, 1.0)
    blurred = cv2.GaussianBlur(alpha, (0, 0), sigma)
    return np.clip(blurred, 0.0, 1.0)


def _edge_darkening(rgb: np.ndarray, alpha: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return rgb
    alpha = np.clip(alpha, 0.0, 1.0)
    edge = alpha * (1.0 - alpha)
    edge = cv2.GaussianBlur(edge, (0, 0), 1.0)
    edge = edge / (edge.max() + 1e-6)
    factor = 1.0 - strength * edge
    return rgb * factor[..., None]


def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _resize(rgb: np.ndarray, preset: str) -> np.ndarray:
    target_w, target_h = TARGETS[preset]
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)


def _qa_metrics(
    input_rgb: np.ndarray, output_rgb: np.ndarray, fg_mask: np.ndarray
) -> dict:
    in_stats = _lab_roi_stats(input_rgb)
    out_stats = _lab_roi_stats(output_rgb)
    bg_stats = _background_stats(output_rgb, fg_mask)
    return {
        "input_face_lab": in_stats,
        "output_face_lab": out_stats,
        "background_lab": bg_stats,
    }


def _warn_if_chroma_shifted(diagnostics: dict) -> None:
    in_stats = diagnostics["input_face_lab"]
    out_stats = diagnostics["output_face_lab"]
    delta_a = abs(out_stats["mean_a"] - in_stats["mean_a"])
    delta_b = abs(out_stats["mean_b"] - in_stats["mean_b"])
    if delta_a > CHROMA_DELTA_WARN or delta_b > CHROMA_DELTA_WARN:
        print(
            f"WARNING: chroma shift detected (Δa={delta_a:.2f}, Δb={delta_b:.2f})"
        )
    bg_std = diagnostics["background_lab"]["std_l"]
    if bg_std > BG_STD_WARN:
        print(f"WARNING: background L std high ({bg_std:.2f})")


def _lab_roi_stats(rgb: np.ndarray) -> dict:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    x0, x1 = int(w * 0.3), int(w * 0.7)
    y0, y1 = int(h * 0.25), int(h * 0.7)
    roi = lab[y0:y1, x0:x1]
    mean = roi.reshape(-1, 3).mean(axis=0)
    std = roi.reshape(-1, 3).std(axis=0)
    return {
        "mean_l": float(mean[0]),
        "mean_a": float(mean[1]),
        "mean_b": float(mean[2]),
        "std_l": float(std[0]),
        "std_a": float(std[1]),
        "std_b": float(std[2]),
    }


def _background_stats(rgb: np.ndarray, fg_mask: np.ndarray) -> dict:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg = fg_mask < 0.1
    if not np.any(bg):
        return {"mean_l": 0.0, "mean_a": 0.0, "mean_b": 0.0, "std_l": 0.0}
    values = lab[bg]
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return {
        "mean_l": float(mean[0]),
        "mean_a": float(mean[1]),
        "mean_b": float(mean[2]),
        "std_l": float(std[0]),
    }


def _encode_jpeg_high_quality(rgb: np.ndarray) -> bytes:
    image = Image.fromarray(rgb)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95, optimize=True)
    return buf.getvalue()
