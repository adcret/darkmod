#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import imageio.v3 as iio
import numpy as np

# --- CONFIGURATION & REGEX ---
# Default fallback regex
DEFAULT_INDEX_RE = re.compile(r"_k(\d+)\.png$")
# Heuristic regexes for auto-detection
K_SUFFIX_RE = re.compile(r"_k(\d+)(?:\D*$)")
DILS_RE = re.compile(r"dislocation_density_(\d+)_log")

def find_images_sorted_by_index(input_dir, index_re, glob_pattern):
    all_candidates = sorted(input_dir.glob(glob_pattern))
    indexed = []
    for p in all_candidates:
        m = index_re.search(p.name)
        if m:
            k = int(m.group(1))
            indexed.append((k, p))
    indexed.sort(key=lambda t: t[0])
    return [p for _, p in indexed]

def write_video(images, output_path, fps):
    if not images:
        raise RuntimeError("No images to write")

    # 1. READ FIRST IMAGE TO SET THE STANDARD
    first = iio.imread(images[0])
    base_h, base_w = first.shape[:2]
    
    # H.264 requires even dimensions
    target_w = base_w + (base_w % 2)
    target_h = base_h + (base_h % 2)

    print(f"Target Resolution: {target_w}x{target_h} (derived from {images[0].name})")

    with iio.imopen(output_path, "w", plugin="pyav") as out:
        out.init_video_stream(fps=fps, codec="h264")
        
        for i, img_path in enumerate(images):
            frame = iio.imread(img_path)
            
            # --- NORMALIZATION ---
            # 1. Fix Channels (Gray->RGB, RGBA->RGB)
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=2)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            
            # 2. Fix Dtype (must be uint8)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255)
                if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
                    frame = frame * 255.0
                frame = frame.astype(np.uint8)

            # --- ROBUST RESIZING (Fixes the Mismatch) ---
            curr_h, curr_w = frame.shape[:2]
            
            # If frame is larger, CENTER CROP it
            if curr_h > target_h:
                start = (curr_h - target_h) // 2
                frame = frame[start : start + target_h, :, ...]
            if curr_w > target_w:
                start = (curr_w - target_w) // 2
                frame = frame[:, start : start + target_w, ...]

            # If frame is smaller (or after crop), PAD it
            curr_h, curr_w = frame.shape[:2]
            pad_h = target_h - curr_h
            pad_w = target_w - curr_w
            
            if pad_h > 0 or pad_w > 0:
                frame = np.pad(
                    frame,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            out.write_frame(frame)
            if i % 10 == 0:
                print(f"Processing frame {i}/{len(images)}...", end="\r")
        print(f"\nDone. Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--glob", type=str, default="*.png")
    parser.add_argument("--regex", type=str, required=True)

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    try:
        index_re = re.compile(args.regex)
    except re.error as exc:
        raise SystemExit(f"Invalid --regex: {exc}")

    images = find_images_sorted_by_index(args.input_dir, index_re, args.glob)
    
    if not images:
        raise SystemExit("No images matched.")

    print(f"Found {len(images)} images.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_video(images, args.output, fps=args.fps)

if __name__ == "__main__":
    main()