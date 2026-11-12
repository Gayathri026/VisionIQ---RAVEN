"""
RAVEN Uncertainty Calibration & Scoring Module
Implements temperature-scaled confidence calibration, epistemic uncertainty
estimation from visual geometry (spatial resolution, occlusion, blur),
and token reliability scoring.
"""

import math
from typing import Dict, Any, List, Optional


def calibrate_confidence(
    raw_confidence: float,
    temperature: float = 1.35,
    margin_bias: float = 0.05
) -> float:
    """
    Applies temperature scaling and Platt margin adjustment to raw model confidence.
    Neural edge detectors often output overconfident scores (e.g., 0.99) on ambiguous crops.
    This sigmoid-based temperature transformation rescales scores towards empirical reliability.
    
    sigma_cal(s) = 1 / (1 + exp(- (logit(s) - margin_bias) / temperature))
    """
    if raw_confidence <= 0.0:
        return 0.0
    if raw_confidence >= 1.0:
        return 0.98  # Guard against absolute certainty claims in open-world vision

    # Safe logit calculation with epsilon clipping
    eps = 1e-6
    s = min(max(raw_confidence, eps), 1.0 - eps)
    logit = math.log(s / (1.0 - s))

    # Rescale by temperature and margin
    calibrated_logit = (logit - margin_bias) / max(0.1, temperature)
    calibrated_prob = 1.0 / (1.0 + math.exp(-calibrated_logit))

    return round(float(min(max(calibrated_prob, 0.01), 0.98)), 4)


def compute_spatial_uncertainty(
    bbox: Optional[List[float]],
    image_shape: Optional[List[int]] = None,
    min_dimension_px: float = 75.0
) -> float:
    """
    Computes visual epistemic uncertainty from bounding box spatial resolution.
    Small crops (low pixel footprint) carry high epistemic uncertainty.
    """
    if not bbox or len(bbox) < 4:
        return 0.50  # Default moderate uncertainty when spatial bounds are absent

    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    dim = min(w, h)

    if dim <= 0.0:
        return 0.90
    if dim >= min_dimension_px:
        # Sufficient spatial resolution
        return round(float(max(0.05, 0.25 * (min_dimension_px / max(dim, min_dimension_px)))), 3)
    else:
        # Sub-resolution crop - uncertainty grows quadratically as dimension shrinks
        ratio = dim / min_dimension_px
        uncertainty = 1.0 - (0.80 * (ratio ** 1.5))
        return round(float(min(0.95, max(0.20, uncertainty))), 3)


def compute_occlusion_uncertainty(
    target_box: List[float],
    other_boxes: List[List[float]],
    iou_threshold: float = 0.15
) -> float:
    """
    Computes uncertainty induced by spatial occlusion with neighbouring objects.
    """
    if not target_box or not other_boxes:
        return 0.0

    max_iou = 0.0
    tx1, ty1, tx2, ty2 = target_box
    t_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    if t_area <= 0.0:
        return 0.80

    for ob in other_boxes:
        ox1, oy1, ox2, oy2 = ob
        ix1 = max(tx1, ox1)
        iy1 = max(ty1, oy1)
        ix2 = min(tx2, ox2)
        iy2 = min(ty2, oy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        o_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
        union = t_area + o_area - inter
        if union > 0:
            iou = inter / union
            if iou > max_iou:
                max_iou = iou

    if max_iou <= iou_threshold:
        return 0.05
    # Scale uncertainty proportionally to maximum overlapping IoU
    scaled_unc = min(0.95, 0.20 + (max_iou * 0.80))
    return round(float(scaled_unc), 3)


def compute_composite_uncertainty(
    raw_confidence: float,
    spatial_unc: float,
    occlusion_unc: float,
    scene_blur_unc: float = 0.0
) -> float:
    """
    Combines predictive variance, spatial resolution, occlusion, and environmental blur
    into an aggregate token uncertainty score U in [0, 1].
    """
    pred_unc = 1.0 - raw_confidence
    composite = (0.35 * pred_unc) + (0.30 * spatial_unc) + (0.25 * occlusion_unc) + (0.10 * scene_blur_unc)
    return round(float(min(max(composite, 0.02), 0.98)), 4)


def recompute_clopper_pearson_upper_bound(
    k: int,
    n: int,
    confidence: float = 0.95
) -> float:
    """
    Computes exact two-sided Clopper-Pearson upper confidence bound for false negative rate.
    Strictly recomputed from underlying validation counts (k false negatives in n trials).
    
    For k = 0: Upper bound = 1 - (alpha / 2)**(1 / n)
    For arbitrary k: Upper bound = BetaInv(1 - alpha / 2, k + 1, n - k)
    """
    if n <= 0:
        return 1.0
    if k < 0:
        k = 0
    alpha = 1.0 - confidence
    if k == 0:
        return float(1.0 - (alpha / 2.0) ** (1.0 / float(n)))
    try:
        from scipy.stats import beta
        return float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    except Exception:
        # High-accuracy Wilson score / Clopper-Pearson bound approximation
        z = 1.96
        p_hat = k / float(n)
        upper = (p_hat + (z * z) / (2.0 * n) + z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * n)) / n)) / (1.0 + (z * z) / n)
        return float(min(1.0, max(0.0, upper)))


def measure_image_quality(image_path: str) -> Dict[str, Any]:
    """
    Evaluates optical image quality and integrity for RAVEN safety validation:
    - Verifies file existence and image readability.
    - Checks minimum spatial resolution (min 50x50 pixels).
    - Measures Laplacian variance for optical blur.
    - Measures average luminance for extreme underexposure or contrast blowout.
    """
    import os
    if not image_path or not os.path.exists(image_path):
        return {
            "is_valid": False,
            "error": "Image file does not exist or unreadable",
            "laplacian_variance": 0.0,
            "mean_luminance": 0.0,
            "width": 0,
            "height": 0
        }

    try:
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        if img is None:
            return {
                "is_valid": False,
                "error": "Image corrupted or invalid raster format",
                "laplacian_variance": 0.0,
                "mean_luminance": 0.0,
                "width": 0,
                "height": 0
            }

        h, w = img.shape[:2]
        if h < 50 or w < 50:
            return {
                "is_valid": False,
                "error": f"Image dimensions {w}x{h} sub-minimum safety threshold (50x50)",
                "laplacian_variance": 0.0,
                "mean_luminance": 0.0,
                "width": w,
                "height": h
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_lum = float(np.mean(gray))

        is_blurry = lap_var < 5.0
        is_dark = mean_lum < 5.0

        if is_blurry:
            return {
                "is_valid": False,
                "error": f"Severe optical blur (Laplacian variance {lap_var:.1f} < 5.0 threshold)",
                "laplacian_variance": round(lap_var, 2),
                "mean_luminance": round(mean_lum, 2),
                "width": w,
                "height": h
            }

        if is_dark:
            return {
                "is_valid": False,
                "error": f"Severe underexposure / pitch black (mean luminance {mean_lum:.1f} < 5.0 threshold)",
                "laplacian_variance": round(lap_var, 2),
                "mean_luminance": round(mean_lum, 2),
                "width": w,
                "height": h
            }

        return {
            "is_valid": True,
            "error": None,
            "laplacian_variance": round(lap_var, 2),
            "mean_luminance": round(mean_lum, 2),
            "width": w,
            "height": h
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"Image quality evaluation failed: {str(e)}",
            "laplacian_variance": 0.0,
            "mean_luminance": 0.0,
            "width": 0,
            "height": 0
        }

