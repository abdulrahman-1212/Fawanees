"""
Colour Smoothing Stage
======================
Produces a painterly / flat-colour version of the input image by
iteratively applying bilateral filtering, and optionally OpenCV's
built-in stylization and detail-enhancement passes.

Design notes
------------
* Purely functional — no hidden state.
* Accepts SmoothConfig so all parameters are caller-controlled.
* Both colour channels and luminance are smoothed (works in BGR).
"""

import cv2
import numpy as np

from config.settings import SmoothConfig


def smooth_colors(image: np.ndarray, cfg: SmoothConfig) -> np.ndarray:
    """
    Smooth an image to produce cartoon-like flat colour regions.

    Strategy
    --------
    1. (Optional) cv2.stylization — OpenCV's edge-preserving, oil-paint filter.
    2. Iterative bilateral filter — preserves edges while blurring interiors.
    3. (Optional) cv2.detailEnhance — sharpens fine structure after smoothing.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image (H × W × 3, uint8).
    cfg : SmoothConfig
        Tunable parameters for this stage.

    Returns
    -------
    np.ndarray
        Smoothed BGR image with the same dtype and shape as input.
    """
    result = image.copy()

    if cfg.use_stylization:
        result = cv2.stylization(
            result,
            sigma_s=cfg.stylization_sigma_s,
            sigma_r=cfg.stylization_sigma_r,
        )
    else:
        for _ in range(cfg.bilateral_iterations):
            result = cv2.bilateralFilter(
                result,
                d=cfg.bilateral_d,
                sigmaColor=cfg.bilateral_sigma_color,
                sigmaSpace=cfg.bilateral_sigma_space,
            )

    if cfg.detail_enhance:
        result = cv2.detailEnhance(
            result,
            sigma_s=cfg.detail_sigma_s,
            sigma_r=cfg.detail_sigma_r,
        )

    return result


def smooth_colors_lab(image: np.ndarray, cfg: SmoothConfig) -> np.ndarray:
    """
    Alternative: smooth only the luminance channel (L* in L*a*b*).
    Preserves colour saturation while reducing texture noise.
    Useful when you want vivid colours with smooth shading.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_ch, a_ch, b_ch = cv2.split(lab)

    for _ in range(cfg.bilateral_iterations):
        l_ch = cv2.bilateralFilter(
            l_ch,
            d=cfg.bilateral_d,
            sigmaColor=cfg.bilateral_sigma_color,
            sigmaSpace=cfg.bilateral_sigma_space,
        )

    merged = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_Lab2BGR)


__all__ = ["smooth_colors", "smooth_colors_lab"]