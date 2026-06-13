"""
Image Cartoonization Pipeline
==============================
Orchestrates core stages into a single callable.

Extend this file (or subclass ImagePipeline) to plug in CartoonGAN or any
other DL-based stage — just override `_apply_dl_stage()`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from config.settings import CartoonConfig
from core.edges import detect_edges, detect_edges_adaptive
from core.smoothing import smooth_colors, smooth_colors_lab
from core.quantization import quantize_colors, quantize_colors_lab
from core.composite import composite


# ---------------------------------------------------------------------------
# Diagnostic output
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Everything the pipeline produces in a single run."""
    cartoon: np.ndarray                          # Final output (BGR)
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def stage_names(self) -> list[str]:
        return list(self.intermediates.keys())

    def total_ms(self) -> float:
        return sum(self.timings.values())


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ImagePipeline:
    """
    Classic (non-DL) image cartoonizer.

    Usage
    -----
    pipe = ImagePipeline()                           # default config
    pipe = ImagePipeline(CartoonConfig.preset("comic"))
    result = pipe.run(bgr_image, debug=True)
    cv2.imshow("cartoon", result.cartoon)

    Extension point for CartoonGAN
    --------------------------------
    Subclass and override `_apply_dl_stage`:

        class GanPipeline(ImagePipeline):
            def __init__(self, model_path, cfg=None):
                super().__init__(cfg)
                self.model = load_cartoongan(model_path)

            def _apply_dl_stage(self, image):
                return run_cartoongan(self.model, image)
    """

    def __init__(self, cfg: Optional[CartoonConfig] = None) -> None:
        self.cfg = cfg or CartoonConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image: np.ndarray,
        *,
        debug: bool = False,
        edge_detector: str = "canny",   # "canny" | "adaptive"
        color_space: str = "bgr",       # "bgr" | "lab"  (smoothing + quant)
    ) -> PipelineResult:
        """
        Cartoonize a single BGR frame.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image.
        debug : bool
            When True, intermediate stages are stored in result.intermediates.
        edge_detector : str
            Which edge algorithm to use ('canny' or 'adaptive').
        color_space : str
            Which colour pipeline to use ('bgr' or 'lab').

        Returns
        -------
        PipelineResult
        """
        intermediates: dict[str, np.ndarray] = {}
        timings: dict[str, float] = {}

        def timed(name: str, fn: Callable[[], np.ndarray]) -> np.ndarray:
            t0 = time.perf_counter()
            out = fn()
            timings[name] = (time.perf_counter() - t0) * 1_000
            if debug:
                intermediates[name] = out.copy()
            return out

        # 1 — Colour smoothing
        smooth_fn = smooth_colors_lab if color_space == "lab" else smooth_colors
        smoothed = timed("smooth", lambda: smooth_fn(image, self.cfg.smooth))

        # 2 — Colour quantisation
        quant_fn = quantize_colors_lab if color_space == "lab" else quantize_colors
        quantised = timed("quantize", lambda: quant_fn(smoothed, self.cfg.quantize))

        # 3 — Edge detection
        edge_fn = detect_edges_adaptive if edge_detector == "adaptive" else detect_edges
        edges = timed("edges", lambda: edge_fn(image, self.cfg.edge))

        # 4 — (Hook) Optional DL stage — no-op in base class
        dl_out = timed("dl_stage", lambda: self._apply_dl_stage(quantised))

        # 5 — Composite
        cartoon = timed("composite", lambda: composite(dl_out, edges, self.cfg.composite))

        return PipelineResult(cartoon=cartoon, intermediates=intermediates, timings=timings)

    # ------------------------------------------------------------------
    # Extension hook
    # ------------------------------------------------------------------

    def _apply_dl_stage(self, image: np.ndarray) -> np.ndarray:
        """
        No-op in the base class.

        Override in a subclass to inject CartoonGAN or any other DL model
        between quantisation and compositing.  Must return a BGR image of
        the same spatial dimensions.
        """
        return image

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def resize_for_processing(
        image: np.ndarray,
        max_dim: int = 1024,
    ) -> tuple[np.ndarray, float]:
        """
        Downscale if the image exceeds max_dim on either axis.
        Returns (resized_image, scale_factor).
        Bilateral filters are O(H·W) — keeping images ≤1024 px keeps them fast.
        """
        h, w = image.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image, scale


__all__ = ["ImagePipeline", "PipelineResult"]