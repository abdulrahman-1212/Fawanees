"""
Video Cartoonization Pipeline
==============================
Wraps ImagePipeline to handle video I/O, frame iteration, temporal smoothing,
progress reporting, and graceful interruption.

Design notes
------------
* The heavy lifting stays in ImagePipeline — this file is pure orchestration.
* Temporal smoothing (exponential moving average) reduces inter-frame flicker.
* The pipeline is easily parallelisable: replace the frame loop with a
  ProcessPoolExecutor over frame batches when speed matters more than flicker.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from config.settings import CartoonConfig
from pipeline.image_pipeline import ImagePipeline, PipelineResult


# ---------------------------------------------------------------------------
# Progress callback type alias
# ---------------------------------------------------------------------------
ProgressCallback = Callable[[int, int, float], None]   # (frame_idx, total, fps)


def _default_progress(idx: int, total: int, fps: float) -> None:
    pct = (idx + 1) / max(total, 1) * 100
    bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
    print(f"\r[{bar}] {pct:5.1f}%  frame {idx+1}/{total}  {fps:.1f} fps", end="")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class VideoPipeline:
    """
    Frame-by-frame video cartoonizer.

    Usage
    -----
    vp = VideoPipeline()
    vp.process("input.mp4", "output.mp4")

    # Custom config
    cfg = CartoonConfig.preset("comic")
    vp = VideoPipeline(cfg)
    vp.process("clip.mp4", "cartoon_clip.mp4", max_frames=300)
    """

    def __init__(
        self,
        cfg: Optional[CartoonConfig] = None,
        image_pipeline: Optional[ImagePipeline] = None,
    ) -> None:
        self.cfg = cfg or CartoonConfig()
        # Allow injecting a pre-built pipeline (e.g. GAN subclass)
        self.image_pipeline = image_pipeline or ImagePipeline(self.cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        max_frames: int = 0,
        progress_callback: Optional[ProgressCallback] = _default_progress,
        edge_detector: str = "canny",
        color_space: str = "bgr",
    ) -> dict:
        """
        Cartoonize a video file and write the result to disk.

        Parameters
        ----------
        input_path : str | Path
            Source video file.
        output_path : str | Path
            Destination video file.
        max_frames : int
            0 = process the entire video.
        progress_callback : callable
            Called each frame with (frame_idx, total_frames, current_fps).
            Pass None to suppress.
        edge_detector : str
            'canny' or 'adaptive'.
        color_space : str
            'bgr' or 'lab'.

        Returns
        -------
        dict
            Summary with frame count, elapsed time, average fps.
        """
        vcfg = self.cfg.video
        input_path = Path(input_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = vcfg.output_fps if vcfg.output_fps > 0 else src_fps
        limit = (max_frames or vcfg.max_frames) or total_frames

        fourcc = cv2.VideoWriter_fourcc(*vcfg.fourcc)
        writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (src_w, src_h))
        if not writer.isOpened():
            cap.release()
            raise IOError(f"Cannot open VideoWriter for: {output_path}")

        prev_cartoon: Optional[np.ndarray] = None
        t_start = time.perf_counter()
        frame_idx = 0

        try:
            while frame_idx < limit:
                ok, frame = cap.read()
                if not ok:
                    break

                result = self.image_pipeline.run(
                    frame,
                    edge_detector=edge_detector,
                    color_space=color_space,
                )
                cartoon = result.cartoon

                # Temporal smoothing — exponential moving average
                if vcfg.temporal_smooth and prev_cartoon is not None:
                    alpha = vcfg.temporal_alpha
                    cartoon = cv2.addWeighted(
                        prev_cartoon, alpha, cartoon, 1.0 - alpha, 0
                    )
                prev_cartoon = cartoon

                writer.write(cartoon)

                if vcfg.show_preview:
                    cv2.imshow("Cartoonizer Preview", cartoon)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\nInterrupted by user.")
                        break

                elapsed = time.perf_counter() - t_start
                current_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
                if progress_callback:
                    progress_callback(frame_idx, limit, current_fps)

                frame_idx += 1

        finally:
            cap.release()
            writer.release()
            if vcfg.show_preview:
                cv2.destroyAllWindows()

        elapsed = time.perf_counter() - t_start
        if progress_callback:
            print()  # newline after progress bar

        return {
            "frames_processed": frame_idx,
            "elapsed_seconds": elapsed,
            "average_fps": frame_idx / elapsed if elapsed > 0 else 0.0,
            "output_path": str(output_path),
        }

    def frame_generator(
        self,
        input_path: str | Path,
        *,
        edge_detector: str = "canny",
        color_space: str = "bgr",
    ) -> Iterator[tuple[np.ndarray, np.ndarray, PipelineResult]]:
        """
        Yield (original_frame, cartoon_frame, pipeline_result) for each frame.
        Useful for custom downstream processing or streaming.

        Example
        -------
        for orig, cartoon, result in vp.frame_generator("clip.mp4"):
            # analyse or display cartoon
        """
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = self.image_pipeline.run(
                    frame,
                    edge_detector=edge_detector,
                    color_space=color_space,
                )
                yield frame, result.cartoon, result
        finally:
            cap.release()


__all__ = ["VideoPipeline"]