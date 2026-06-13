#!/usr/bin/env python3
"""
cartoonize.py – Command-line interface for the unified cartoonization pipeline.
"""
import argparse
import sys
import os
import time
import cv2
from pipeline import process_image, CLASSICAL_STYLES

def parse_args():
    p = argparse.ArgumentParser(
        description="Cartoonize an image using Classical CV or CartoonGAN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Styles (Classical): comic, anime, sketch, neon, stained_glass
Styles (GAN): cartoongan (requires --use-gan)""")
    
    p.add_argument("input", help="Path to input image")
    p.add_argument("output", nargs="?", help="Path to output image (optional)")
    p.add_argument("--style", default="comic", help="Cartoon style (default: comic)")
    p.add_argument("--use-gan", action="store_true", help="Use CartoonGAN deep learning model")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Compute device for GAN")
    p.add_argument("--all", action="store_true", help="Generate all classical styles")
    return p.parse_args()

def auto_output(input_path: str, style: str) -> str:
    base, ext = os.path.splitext(input_path)
    ext = ext or ".png"
    return f"{base}_{style}{ext}"

def main():
    args = parse_args()
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: cannot read '{args.input}'", file=sys.stderr)
        sys.exit(1)
        
    h, w = img.shape[:2]
    print(f"Input : {args.input}  ({w}×{h})")
    
    if args.all:
        styles_to_run = list(CLASSICAL_STYLES)
        use_gan = False
    else:
        styles_to_run = [args.style]
        use_gan = args.use_gan

    for style in styles_to_run:
        out_path = auto_output(args.input, style) if (args.all or not args.output) else args.output
            
        t0 = time.perf_counter()
        try:
            result = process_image(img, style=style, use_gan=use_gan, device=args.device)
            elapsed = (time.perf_counter() - t0) * 1000
            cv2.imwrite(out_path, result)
            method = "GAN" if use_gan else "CV "
            print(f"  [{method} | {style:>13}]  →  {out_path}  ({elapsed:.0f} ms)")
        except Exception as e:
            print(f"  [ERROR | {style:>13}]  →  {e}", file=sys.stderr)
            
    print("Done.")

if __name__ == "__main__":
    main()