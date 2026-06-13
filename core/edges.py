"""
Edge Detection Stage
====================
Produces a binary edge mask from a BGR image.

Design notes
------------
* Purely functional — no state, easy to swap or test in isolation.
* Accepts an EdgeConfig so callers can tune without touching this file.
* Returns a single-channel uint8 mask (0 = no edge, 255 = edge).
"""

import cv2
import numpy as np

from config.settings import EdgeConfig


def detect_edges(image: np.ndarray, cfg: EdgeConfig) -> np.ndarray:
    """
    Convert a BGR image to a cartoon-style edge mask.

    Pipeline
    --------
    BGR → grayscale → median blur → Canny → optional dilation → optional invert

    Parameters
    ----------
    image : np.ndarray
        Input BGR image (H × W × 3, uint8).
    cfg : EdgeConfig
        Tunable parameters for this stage.

    Returns
    -------
    np.ndarray
        Single-channel uint8 edge mask (H × W).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Median blur suppresses noise without blurring edges as much as Gaussian
    blurred = cv2.medianBlur(gray, cfg.blur_kernel)

    edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)

    if cfg.dilate_kernel > 0:
        kernel = np.ones((cfg.dilate_kernel, cfg.dilate_kernel), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    if cfg.invert:
        edges = cv2.bitwise_not(edges)

    return edges


def detect_edges_adaptive(image: np.ndarray, cfg: EdgeConfig) -> np.ndarray:
    """
    Alternative edge detector using adaptive thresholding.
    Useful for images with highly variable lighting conditions.

    Returns the same uint8 mask contract as detect_edges().
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, cfg.blur_kernel)

    edges = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=9,
        C=2,
    )
    # adaptiveThreshold already produces inverted-style output (white bg, black edges)
    if not cfg.invert:
        edges = cv2.bitwise_not(edges)

    if cfg.dilate_kernel > 0:
        kernel = np.ones((cfg.dilate_kernel, cfg.dilate_kernel), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)  # erode thins black lines

    return edges


__all__ = ["detect_edges", "detect_edges_adaptive"]