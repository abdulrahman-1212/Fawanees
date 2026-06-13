"""
Composite Stage
===============
Overlays the edge mask onto the smoothed / quantised colour image
to produce the final cartoon frame.

Design notes
------------
* Three blend modes give very different visual outcomes — swap via config.
* edge_strength lets callers dial in how prominent the ink lines are.
* The edge mask is expected in the inverted convention (255 = no edge, 0 = edge)
  that matches EdgeConfig.invert = True.  If you use a raw Canny mask flip it
  before passing in, or set cfg.composite.blend_mode = 'and'.
"""

import cv2
import numpy as np

from config.settings import CompositeConfig


def composite(
    color_image: np.ndarray,
    edge_mask: np.ndarray,
    cfg: CompositeConfig,
) -> np.ndarray:
    """
    Blend edge mask onto colour image.

    Parameters
    ----------
    color_image : np.ndarray
        Smoothed (and optionally quantised) BGR image (H × W × 3, uint8).
    edge_mask : np.ndarray
        Single-channel uint8 mask (H × W).
        Convention: 255 = background, 0 = edge line  (inverted Canny).
    cfg : CompositeConfig
        Blend mode and edge strength.

    Returns
    -------
    np.ndarray
        Final cartoon BGR image (H × W × 3, uint8).
    """
    # Expand mask to 3 channels for per-pixel operations
    mask_3ch = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    if cfg.edge_strength != 1.0:
        # Lerp: full-white where no edge (mask==1), partial-dark where edge
        mask_3ch = 1.0 - (1.0 - mask_3ch) * cfg.edge_strength

    color_f = color_image.astype(np.float32) / 255.0

    if cfg.blend_mode == "multiply":
        # Darkens colour image where mask is dark (edge pixels)
        result_f = color_f * mask_3ch

    elif cfg.blend_mode == "and":
        # Hard black lines — bitwise AND on uint8, then apply strength via lerp
        mask_u8 = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(color_image, mask_u8)
        if cfg.edge_strength < 1.0:
            result = cv2.addWeighted(result, cfg.edge_strength,
                                     color_image, 1.0 - cfg.edge_strength, 0)
        return result

    elif cfg.blend_mode == "overlay":
        # Photoshop-style overlay: darkens darks, lightens lights
        low = 2.0 * color_f * mask_3ch
        high = 1.0 - 2.0 * (1.0 - color_f) * (1.0 - mask_3ch)
        result_f = np.where(mask_3ch <= 0.5, low, high)

    else:
        raise ValueError(
            f"Unknown blend_mode '{cfg.blend_mode}'. "
            "Choose 'multiply', 'and', or 'overlay'."
        )

    result_f = np.clip(result_f, 0.0, 1.0)
    return (result_f * 255).astype(np.uint8)


__all__ = ["composite"]