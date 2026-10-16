#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np


DEFAULT_INDEX_RE = re.compile(r"_k(\d+)\.png$")
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


def extract_index_auto(filename):
    m = K_SUFFIX_RE.search(filename)
    if m:
        return int(m.group(1)), 3
    m = DILS_RE.search(filename)
    if m:
        return int(m.group(1)), 2
    nums = re.findall(r"\d+", filename)
    if nums:
        return int(nums[-1]), 1
    return None, None


def auto_collect_sorted(input_dir, glob_pattern):
    pngs = sorted(input_dir.glob(glob_pattern))
    chosen = {}
    skipped_no_index = 0
    duplicates_resolved = 0
    for p in pngs:
        idx, weight = extract_index_auto(p.name)
        if idx is None:
            skipped_no_index += 1
            continue
        # De-prioritize filenames that look like 'part\d' (penalty)
        penalty = 1 if re.search(r"(?:^|[_-])part\d+", p.name, re.IGNORECASE) else 0
        score = (weight * 10) - penalty
        existing = chosen.get(idx)
        if existing is None:
            chosen[idx] = (score, p)
        else:
            duplicates_resolved += 1
            if score > existing[0] or (
                score == existing[0] and p.name < existing[1].name
            ):
                chosen[idx] = (score, p)
    ordered = [pair[1] for _, pair in sorted(chosen.items(), key=lambda t: t[0])]
    report = (
        f"Scanned {len(pngs)} PNG files; kept {len(ordered)} frames; "
        f"skipped {skipped_no_index} without index; resolved {duplicates_resolved} duplicates."
    )
    return ordered, report


def check_frames(images):
    if not images:
        print("No images selected.")
        return
    sizes = {}
    channels = {}
    dtypes = {}
    odd_dim_examples = []
    max_examples = 8
    for p in images:
        arr = iio.imread(p)
        h, w = arr.shape[:2]
        sizes[(w, h)] = sizes.get((w, h), 0) + 1
        ch = 1 if arr.ndim == 2 else arr.shape[2]
        channels[ch] = channels.get(ch, 0) + 1
        dtypes[str(arr.dtype)] = dtypes.get(str(arr.dtype), 0) + 1
        if (w % 2 != 0) or (h % 2 != 0):
            if len(odd_dim_examples) < max_examples:
                odd_dim_examples.append(f"{p.name} -> {w}x{h}")
    print(f"Total frames: {len(images)}")
    print(
        f"Unique sizes (WxH): {len(sizes)} -> {sorted([f'{w}x{h}({c})' for (w,h),c in sizes.items()])}"
    )
    print(f"Channels counts: {sorted(channels.items())}  (1=gray, 3=RGB, 4=RGBA)")
    print(f"Dtypes: {sorted(dtypes.items())}")
    if odd_dim_examples:
        print("Found frames with odd dimensions (H.264 yuv420p needs even):")
        for ex in odd_dim_examples:
            print(f"  - {ex}")
    else:
        print("All frame dimensions are even.")


def write_video(images, output_path, fps):
    if not images:
        raise RuntimeError("No images to write in the input directory")

    first = iio.imread(images[0])
    height, width = first.shape[:2]
    # H.264 (yuv420p) requires even width/height. Pad to even if needed.
    target_width = width + (width % 2)
    target_height = height + (height % 2)

    with iio.imopen(output_path, "w", plugin="pyav") as out:
        # Let the plugin infer frame size from the first written frame
        out.init_video_stream(fps=fps, codec="h264")
        # Ensure all frames match size; if not, center-crop or resize could be added here.
        for img_path in images:
            frame = iio.imread(img_path)
            if frame.shape[0] != height or frame.shape[1] != width:
                raise RuntimeError(
                    f"Frame size mismatch for '{img_path}': got {frame.shape[1]}x{frame.shape[0]}, expected {width}x{height}."
                )
            # Normalize channels: grayscale->RGB, RGBA->RGB (drop alpha)
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=2)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            elif frame.ndim == 3 and frame.shape[2] == 3:
                pass
            else:
                raise RuntimeError(
                    f"Unsupported frame shape for '{img_path}': {frame.shape}"
                )

            # Ensure dtype uint8
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255)
                if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
                    frame = frame * 255.0
                frame = frame.astype(np.uint8)
            # Pad to even dimensions if required by codec
            pad_h = target_height - frame.shape[0]
            pad_w = target_width - frame.shape[1]
            if pad_h != 0 or pad_w != 0:
                # pad on bottom and right with zeros (black)
                frame = np.pad(
                    frame,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            out.write_frame(frame)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a movie from PNGs in a folder, sorted by trailing _k{index} number. "
            "Filenames like 'part1_..._k12.png' and 'part2_..._k7.png' are both included; "
            "sorting ignores the part number and only uses k."
        )
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path, help="Directory containing PNG frames"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output video file path (e.g., out.mp4)",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second (default: 10)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only analyze frames and print diagnostics; do not write video",
    )
    parser.add_argument(
        "--start-k",
        type=int,
        default=None,
        help="Optional: start at this k index (inclusive)",
    )
    parser.add_argument(
        "--end-k",
        type=int,
        default=None,
        help="Optional: end at this k index (inclusive)",
    )
    parser.add_argument(
        "--regex",
        type=str,
        default=r"_k(\\d+)\\.png$",
        help=(
            "Regex with one capturing group for the frame index (default: '_k(\\d+)\\.png$'). "
            "Example for your other dataset: 'dislocation_density_(\\d+)_log\\.png$'"
        ),
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.png",
        help="Glob to select candidate files within the directory (default: '*.png')",
    )

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    try:
        index_re = re.compile(args.regex)
    except re.error as exc:
        raise SystemExit(f"Invalid --regex: {exc}")

    # Auto mode when regex is empty string
    if args.regex == "":
        images, report = auto_collect_sorted(args.input_dir, args.glob)
        if not images:
            all_candidates = sorted(args.input_dir.glob(args.glob))
            examples = (
                ", ".join(p.name for p in all_candidates[:8])
                if all_candidates
                else "<none>"
            )
            raise SystemExit(
                "Auto-detect found no index-bearing images. "
                f"Directory: {args.input_dir}\n"
                f"Glob: {args.glob} (found {len(all_candidates)} files)\n"
                f"Examples: {examples}"
            )
        print(report)
    else:
        images = find_images_sorted_by_index(
            args.input_dir, index_re=index_re, glob_pattern=args.glob
        )
        if not images:
            all_candidates = sorted(args.input_dir.glob(args.glob))
            examples = (
                ", ".join(p.name for p in all_candidates[:8])
                if all_candidates
                else "<none>"
            )
            raise SystemExit(
                "No images matched the provided --regex in the input directory. "
                f"Directory: {args.input_dir}\n"
                f"Glob: {args.glob} (found {len(all_candidates)} files)\n"
                f"Regex: {args.regex}\n"
                f"Examples: {examples}"
            )

    if args.start_k is not None or args.end_k is not None:
        filtered = []
        for p in images:
            m = index_re.search(p.name)
            k_val = int(m.group(1)) if m else None
            if k_val is None:
                continue
            if args.start_k is not None and k_val < args.start_k:
                continue
            if args.end_k is not None and k_val > args.end_k:
                continue
            filtered.append(p)
        images = filtered

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        check_frames(images)
        return
    write_video(images, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
