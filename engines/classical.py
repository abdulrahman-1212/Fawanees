import cv2
import numpy as np

# ─── Core Helpers ──────────────────────────────────────────────────────────────

def _optimize_resolution(img: np.ndarray, max_dim: int = 1024) -> tuple[np.ndarray, tuple[int, int]]:
    """Downscale large images for heavy processing, return original shape for upscaling."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), (h, w)
    return img, (h, w)

def _bilateral_smooth(img: np.ndarray, rounds: int = 2, d: int = 9, sc: float = 75, ss: float = 75) -> np.ndarray:
    out = img.copy()
    for _ in range(rounds):
        out = cv2.bilateralFilter(out, d, sc, ss)
    return out

def _quantise_colours(img: np.ndarray, k: int = 10) -> np.ndarray:
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centres = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    return np.uint8(centres)[labels.flatten()].reshape(img.shape)

def _adaptive_edge_mask(gray: np.ndarray, blur_k: int = 5, dilation_iter: int = 1) -> np.ndarray:
    """Canny edge mask with adaptive thresholds based on image median."""
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    median = np.median(blurred)
    sigma = 0.33
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))
    
    edges = cv2.Canny(blurred, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(edges, kernel, iterations=dilation_iter)

def _overlay_edges(colour: np.ndarray, edges: np.ndarray, edge_colour=(0, 0, 0)) -> np.ndarray:
    out = colour.copy()
    mask = edges == 255
    out[mask] = edge_colour
    return out

def _boost_saturation(img: np.ndarray, scale: float = 1.4) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ─── Style Implementations ─────────────────────────────────────────────────────

def comic(img: np.ndarray) -> np.ndarray:
    smooth = _bilateral_smooth(img, rounds=2, d=9, sc=80, ss=80)
    quant = _quantise_colours(smooth, k=12)
    vibrant = _boost_saturation(quant, 1.5)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = _adaptive_edge_mask(gray, blur_k=5, dilation_iter=1)
    return _overlay_edges(vibrant, edges, (0, 0, 0))

def anime(img: np.ndarray) -> np.ndarray:
    smooth = _bilateral_smooth(img, rounds=3, d=9, sc=60, ss=60)
    pastel = cv2.addWeighted(smooth, 0.75, np.full_like(smooth, 245), 0.25, 0)
    boosted = _boost_saturation(pastel, 1.25)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = _adaptive_edge_mask(gray, blur_k=7, dilation_iter=1)
    edges_soft = cv2.GaussianBlur(edges, (3, 3), 0)
    result = _overlay_edges(boosted, edges_soft, (40, 30, 20))
    bloom = cv2.GaussianBlur(result, (15, 15), 0)
    return cv2.addWeighted(result, 0.85, bloom, 0.15, 0)

def sketch(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur_inv = cv2.GaussianBlur(inv, (21, 21), 0)
    dodge = cv2.divide(gray, 255.0 - blur_inv, scale=256.0)
    dodge = np.clip(dodge, 0, 255).astype(np.uint8)
    sketch_bgr = cv2.cvtColor(dodge, cv2.COLOR_GRAY2BGR)
    tint = np.zeros_like(sketch_bgr)
    tint[..., 0] = 10  
    tint[..., 2] = 20
    return cv2.add(sketch_bgr, tint)

def neon(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = _adaptive_edge_mask(gray, blur_k=3, dilation_iter=1)
    edge_mask_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    colour_edges = cv2.bitwise_and(img, edge_mask_3ch)
    colour_edges = _boost_saturation(colour_edges, 3.0)
    
    glow_tight = cv2.GaussianBlur(colour_edges, (5, 5), 0)
    glow_wide = cv2.GaussianBlur(colour_edges, (21, 21), 0)
    glow = cv2.addWeighted(glow_tight, 1.0, glow_wide, 0.6, 0)
    
    background = (img.astype(np.float32) * 0.08).astype(np.uint8)
    result = cv2.add(background, glow)
    return np.clip(cv2.add(result, colour_edges), 0, 255).astype(np.uint8)

def stained_glass(img: np.ndarray) -> np.ndarray:
    smooth = img.copy()
    for _ in range(3): 
        smooth = cv2.pyrMeanShiftFiltering(smooth, sp=10, sr=40)
    vibrant = _boost_saturation(smooth, 2.0)
    quant = _quantise_colours(vibrant, k=8)
    
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, thick_edges = cv2.threshold(grad, 12, 255, cv2.THRESH_BINARY)
    thick_edges = cv2.dilate(thick_edges, kernel, iterations=1)
    
    return _overlay_edges(quant, thick_edges, (0, 0, 0))

# ─── Public API ───────────────────────────────────────────────────────────────

STYLES = {
    "comic": comic,
    "anime": anime,
    "sketch": sketch,
    "neon": neon,
    "stained_glass": stained_glass,
}

def cartoonize(img_bgr: np.ndarray, style: str = "comic") -> np.ndarray:
    if style not in STYLES:
        raise ValueError(f"Unknown style '{style}'. Choose from: {list(STYLES)}")
    
    # 1. Optimize resolution for heavy CV operations
    processed_img, original_shape = _optimize_resolution(img_bgr, max_dim=1024)
    
    # 2. Apply style
    result = STYLES[style](processed_img)
    
    # 3. Upscale back to original resolution if it was downscaled
    if original_shape != result.shape[:2]:
        result = cv2.resize(result, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        
    return result