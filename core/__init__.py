from .edges import detect_edges, detect_edges_adaptive
from .smoothing import smooth_colors, smooth_colors_lab
from .quantization import quantize_colors, quantize_colors_lab
from .composite import composite

__all__ = [
    "detect_edges", "detect_edges_adaptive",
    "smooth_colors", "smooth_colors_lab",
    "quantize_colors", "quantize_colors_lab",
    "composite",
]