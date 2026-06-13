"""
Cartoonizer Configuration
=========================
Central place for all tunable parameters.
Swap profiles or override per-call — nothing is hard-coded in the pipeline.
"""

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Edge-detection profiles
# ---------------------------------------------------------------------------

@dataclass
class EdgeConfig:
    """Parameters for the edge-detection stage."""
    blur_kernel: int = 7          # Median blur before edge detection (odd int)
    canny_low: int = 75           # Canny lower threshold
    canny_high: int = 200         # Canny upper threshold
    dilate_kernel: int = 1        # Dilation size for thickening edges (0 = off)
    invert: bool = True           # True → black lines on white; False → raw mask


# ---------------------------------------------------------------------------
# Colour-smoothing profiles
# ---------------------------------------------------------------------------

@dataclass
class SmoothConfig:
    """Parameters for the colour-smoothing (bilateral / stylization) stage."""
    # Bilateral filter (applied iteratively for a painting effect)
    bilateral_d: int = 9          # Neighbourhood diameter
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    bilateral_iterations: int = 4  # More passes → more painterly, slower

    # Optional OpenCV stylization (replaces bilateral when enabled)
    use_stylization: bool = False
    stylization_sigma_s: float = 60.0   # Spatial spread  [0–200]
    stylization_sigma_r: float = 0.45   # Color sensitivity [0–1]

    # Optional detail-enhance pass after bilateral
    detail_enhance: bool = False
    detail_sigma_s: float = 10.0
    detail_sigma_r: float = 0.15


# ---------------------------------------------------------------------------
# Colour-quantisation profiles
# ---------------------------------------------------------------------------

@dataclass
class QuantizeConfig:
    """Parameters for the colour-quantisation stage."""
    enabled: bool = True
    k: int = 8                    # Number of colour clusters
    attempts: int = 3             # k-means attempts (higher → more stable)
    max_iter: int = 10            # k-means max iterations per attempt


# ---------------------------------------------------------------------------
# Composite stage
# ---------------------------------------------------------------------------

@dataclass
class CompositeConfig:
    """How the edge mask is blended onto the smoothed image."""
    # 'multiply' | 'and' | 'overlay'
    blend_mode: str = "multiply"
    edge_strength: float = 1.0    # 0.0 = no edges, 1.0 = full edges


# ---------------------------------------------------------------------------
# Video-specific settings
# ---------------------------------------------------------------------------

@dataclass
class VideoConfig:
    """Settings used only by the video pipeline."""
    output_fps: float = 0.0       # 0 → copy source FPS
    fourcc: str = "mp4v"          # Codec for output file
    max_frames: int = 0           # 0 → process entire video
    show_preview: bool = False    # Display frame while processing
    temporal_smooth: bool = True  # Blend adjacent frames to reduce flicker
    temporal_alpha: float = 0.35  # Weight of previous cartoon frame [0–1]


# ---------------------------------------------------------------------------
# Full pipeline profile
# ---------------------------------------------------------------------------

@dataclass
class CartoonConfig:
    """
    Top-level config that groups all sub-configs.

    Usage
    -----
    cfg = CartoonConfig()                   # defaults
    cfg = CartoonConfig.preset("sketch")    # named preset
    cfg.edge.canny_low = 50                 # per-call override
    """
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    smooth: SmoothConfig = field(default_factory=SmoothConfig)
    quantize: QuantizeConfig = field(default_factory=QuantizeConfig)
    composite: CompositeConfig = field(default_factory=CompositeConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    @classmethod
    def preset(cls, name: str) -> "CartoonConfig":
        """Return a named configuration preset."""
        presets = {
            "default": cls(),

            "sketch": cls(
                smooth=SmoothConfig(bilateral_iterations=1, use_stylization=False),
                quantize=QuantizeConfig(enabled=False),
                edge=EdgeConfig(canny_low=50, canny_high=150, dilate_kernel=2),
                composite=CompositeConfig(blend_mode="and"),
            ),

            "watercolor": cls(
                smooth=SmoothConfig(
                    use_stylization=True,
                    stylization_sigma_s=80,
                    stylization_sigma_r=0.5,
                    detail_enhance=True,
                ),
                quantize=QuantizeConfig(k=12),
                edge=EdgeConfig(canny_low=30, canny_high=100, dilate_kernel=1),
                composite=CompositeConfig(edge_strength=0.6),
            ),

            "comic": cls(
                smooth=SmoothConfig(bilateral_iterations=6, bilateral_d=9),
                quantize=QuantizeConfig(k=6),
                edge=EdgeConfig(canny_low=80, canny_high=220, dilate_kernel=2),
                composite=CompositeConfig(blend_mode="multiply", edge_strength=1.0),
            ),

            "fast": cls(
                smooth=SmoothConfig(bilateral_iterations=2, bilateral_d=7),
                quantize=QuantizeConfig(k=6, attempts=1, max_iter=5),
            ),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(presets)}")
        return presets[name]


__all__ = [
    "EdgeConfig",
    "SmoothConfig",
    "QuantizeConfig",
    "CompositeConfig",
    "VideoConfig",
    "CartoonConfig",
]