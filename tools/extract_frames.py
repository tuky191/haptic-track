#!/usr/bin/env python3
"""Extract frames from a video file at a given interval.

Usage:
    python extract_frames.py <video_path> <output_dir> [--fps 2]

Extracts frames as PNGs into output_dir. Default: 2 frames per second.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def extract(video_path: str, output_dir: str, fps: float = 2.0):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(out / "frame_%04d.png"),
        "-y",
    ]
    print(f"Extracting frames at {fps} fps → {out}/")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    frames = sorted(out.glob("frame_*.png"))
    print(f"Extracted {len(frames)} frames")
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("output_dir", help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract")
    args = parser.parse_args()
    extract(args.video, args.output_dir, args.fps)
