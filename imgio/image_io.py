import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Union

def read_image(path: Union[str, Path]) -> np.ndarray:
    """
    Read an image from the given path using OpenCV.
    
    Args:
        path: Path to the image file.
        
    Returns:
        A numpy array representing the image in BGR format.
        
    Raises:
        FileNotFoundError: If the image cannot be read or does not exist.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image from {path}")
    return img

def write_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """
    Write an image to the given path using OpenCV.
    
    Args:
        image: A numpy array representing the image.
        path: Path where the image should be saved.
        
    Raises:
        IOError: If the image cannot be written to disk.
    """
    path = Path(path)
    # Ensure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Failed to write image to {path}")

def build_debug_grid(stages: Dict[str, np.ndarray], cols: int = 3) -> np.ndarray:
    """
    Build a grid of images for debugging intermediate pipeline stages.
    
    Args:
        stages: A dictionary mapping stage names (str) to images (np.ndarray).
        cols: Number of columns in the grid.
        
    Returns:
        A single numpy array representing the combined, labeled grid.
    """
    if not stages:
        raise ValueError("No stages provided to build debug grid")
    
    names = list(stages.keys())
    images = list(stages.values())
    
    # Use the first image as the reference for dimensions
    ref_h, ref_w = images[0].shape[:2]
    processed_images = []
    
    for img in images:
        # Convert to 3-channel BGR if the image is grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Resize to match reference dimensions for a uniform grid
        if img.shape[:2] != (ref_h, ref_w):
            img = cv2.resize(img, (ref_w, ref_h))
            
        processed_images.append(img)
        
    rows = (len(processed_images) + cols - 1) // cols
    
    # Pad with black images if the last row is not completely full
    while len(processed_images) % cols != 0:
        processed_images.append(np.zeros_like(processed_images[0]))
        names.append("")
        
    row_images = []
    for r in range(rows):
        row_imgs = processed_images[r * cols : (r + 1) * cols]
        labeled_imgs = []
        
        for i, img in enumerate(row_imgs):
            name = names[r * cols + i]
            if name:
                img_with_text = img.copy()
                
                # Dynamically scale font size and thickness based on image height
                font_scale = max(0.4, ref_h / 1000.0)
                thickness = max(1, int(ref_h / 200))
                y_pos = min(30, max(20, int(ref_h * 0.05)))
                
                cv2.putText(
                    img_with_text, 
                    name, 
                    (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (0, 255, 0),  # Green text
                    thickness, 
                    cv2.LINE_AA
                )
            else:
                img_with_text = img
                
            labeled_imgs.append(img_with_text)
            
        # Concatenate images horizontally for the current row
        row_concat = np.hstack(labeled_imgs)
        row_images.append(row_concat)
        
    # Concatenate all rows vertically to form the final grid
    grid = np.vstack(row_images)
    return grid