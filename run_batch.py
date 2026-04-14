#!/usr/bin/env python3
"""
Batch pipeline runner — processes a list of videos through run_mono_standalone.

Usage:
    # By video ID (looks up /ceph/Dataset/QEVD-FIT-COACH/long_range_videos/<id>.mp4):
    python run_batch.py 0000 0001 0002

    # By absolute path:
    python run_batch.py /ceph/Dataset/QEVD-FIT-COACH/long_range_videos/0000.mp4

    # Mix of both:
    python run_batch.py 0000 /some/other/video.mp4

    # Rerun even if cached:
    python run_batch.py --rerun 0000 0001
"""

import argparse
import os
import sys
import traceback

from loguru import logger

from run_mono_standalone import METADATA_PATH, run_mono_standalone

VIDEO_DIR = "/ceph/Dataset/QEVD-FIT-COACH/long_range_videos/"
CALIB_PATH = ""  # no calibration file for this dataset


def resolve_video_path(arg: str) -> str:
    """Accept either a 4-digit ID or an absolute/relative path."""
    if os.path.isfile(arg):
        return arg
    candidate = os.path.join(VIDEO_DIR, arg + ".mp4")
    if os.path.isfile(candidate):
        return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Batch mono pipeline runner")
    parser.add_argument(
        "videos",
        nargs="+",
        help="Video IDs (e.g. 0000) or absolute paths to process",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        default=False,
        help="Reprocess even if cached results exist",
    )
    parser.add_argument(
        "--metadata",
        default=METADATA_PATH,
        help=f"Path to body info JSON (default: {METADATA_PATH})",
    )
    args = parser.parse_args()

    # Resolve all paths up front so we fail early on missing files
    video_paths = []
    for v in args.videos:
        path = resolve_video_path(v)
        if path is None:
            logger.error(f"Video not found: '{v}' (tried as path and as ID in {VIDEO_DIR})")
            sys.exit(1)
        video_paths.append(path)

    # Load WHAM once for all videos
    try:
        from WHAM.demo import initialize_wham
        logger.info("Loading WHAM model ...")
        initialize_wham()
        logger.info("WHAM model loaded.")
    except Exception as e:
        logger.warning(f"Could not initialise WHAM model: {e}")

    results = []
    for i, video_path in enumerate(video_paths):
        logger.info(f"[{i+1}/{len(video_paths)}] Processing {video_path} ...")
        try:
            result = run_mono_standalone(
                video_path=video_path,
                metadata_path=args.metadata,
                calib_path=CALIB_PATH,
                estimate_local_only=True,
                rerun=args.rerun,
            )
            results.append({"video": video_path, "status": "ok", "result": result})
            logger.info(f"[{i+1}/{len(video_paths)}] Done — case_id: {result.get('case_id')}")

        except Exception:
            tb = traceback.format_exc()
            logger.error(f"[{i+1}/{len(video_paths)}] FAILED — {video_path}\n{tb}")
            results.append({"video": video_path, "status": "error", "traceback": tb})

    # Summary
    ok    = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "error"]
    logger.info(f"\nBatch complete: {len(ok)}/{len(video_paths)} succeeded, {len(failed)} failed.")
    for r in failed:
        logger.error(f"  FAILED: {r['video']}")


if __name__ == "__main__":
    main()
