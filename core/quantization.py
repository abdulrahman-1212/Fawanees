"""
Colour Quantisation Stage
=========================
Reduces the palette of a smoothed image to K representative colours,
reinforcing the flat-fill look of cartoons.

Design notes
------------
* Uses k-means clustering in BGR space (fast, broadly available).
* An optional LAB-space variant is provided for perceptually uniform clusters.
* Purely functional — returns a new array, never mutates the input.
"""

import cv2
import numpy as np

from config.settings import QuantizeConfig


def quantize_colors(image: np.ndarray, cfg: QuantizeConfig) -> np.ndarray:
    """
    Reduce an image's palette via k-means clustering (BGR space).

    Parameters
    ----------
    image : np.ndarray
        Smoothed BGR image (H × W × 3, uint8).
    cfg : QuantizeConfig
        Tunable parameters for this stage.

    Returns
    -------
    np.ndarray
        Quantised BGR image (same H × W × 3, uint8).
    """
    if not cfg.enabled:
        return image

    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        cfg.max_iter,
        1.0,
    )
    _, labels, centres = cv2.kmeans(
        pixels,
        cfg.k,
        None,
        criteria,
        cfg.attempts,
        cv2.KMEANS_PP_CENTERS,
    )

    centres = np.uint8(centres)
    quantised = centres[labels.flatten()]
    return quantised.reshape(h, w, 3)


def quantize_colors_lab(image: np.ndarray, cfg: QuantizeConfig) -> np.ndarray:
    """
    Palette reduction in CIE L*a*b* space for perceptually equal cluster sizes.
    Converts to LAB, clusters, maps back to BGR.
    """
    if not cfg.enabled:
        return image

    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        cfg.max_iter,
        1.0,
    )
    _, labels, centres = cv2.kmeans(
        pixels,
        cfg.k,
        None,
        criteria,
        cfg.attempts,
        cv2.KMEANS_PP_CENTERS,
    )

    centres = np.uint8(np.clip(centres, 0, 255))
    quantised_lab = centres[labels.flatten()].reshape(h, w, 3)
    return cv2.cvtColor(quantised_lab, cv2.COLOR_Lab2BGR)


__all__ = ["quantize_colors", "quantize_colors_lab"]