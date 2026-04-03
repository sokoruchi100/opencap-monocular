import requests
import logging
import os
import subprocess
import cv2
from loguru import logger
from decouple import config as env_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _get_video_mime_type(video_path):
    """Get MIME type based on file extension."""
    ext = os.path.splitext(video_path)[1].lower()
    mime_types = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    return mime_types.get(ext, "video/mp4")  # Default to mp4 if unknown


def predict_activity_from_video(video_path):
    """
    Predicts the activity from a video by calling the activity classification API.
    Also determines if a flat floor assumption is valid based on the activity.

    Args:
        video_path (str): The path to the video file.

    Returns:
        tuple: A tuple containing:
            - str: The predicted activity (or None if prediction fails).
            - bool: True if a flat floor can be assumed, False otherwise.
    """
    predicted_activity = None
    flat_floor = False  # Default to False

    # List of activities where a flat floor can be assumed
    flat_floor_activities = [
        "squatting",
        "sit-to-stand",
        "walking",
        "running",
        "standing",
        "sitting",
    ]

    try:
        videollama_base = env_config("VIDEOLLAMA_URL", default="http://localhost:8400").rstrip("/")
        activity_api_url = f"{videollama_base}/predict"
        logger.info(
            f"Requesting activity prediction from {activity_api_url} for: {video_path}"
        )

        # Send video path as a form field — the VideoLLaMA container reads the
        # file directly from the shared volume (same path in both containers).
        data = {"video_path": os.path.abspath(video_path), "fps": "3", "max_frames": "20"}
        response = requests.post(activity_api_url, data=data, timeout=120)

        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        activity_data = response.json()
        predicted_activity = activity_data.get("predicted_activity")
        logger.info(f"Predicted activity: {predicted_activity}")

        # Handle activities with slashes (e.g., "Squatting/Sit-to-stand")
        # Split by "/" and check if any part matches the allowed activities
        activity_parts = [
            part.strip().lower() for part in predicted_activity.split("/")
        ]
        matches_allowed_activity = any(
            part in flat_floor_activities for part in activity_parts
        )
        has_one_leg = "1 leg" in predicted_activity.lower()

        if matches_allowed_activity and not has_one_leg:
            flat_floor = True
            logger.info(
                f"Activity '{predicted_activity}' allows for flat floor assumption."
            )
        else:
            logger.info(
                f"Activity '{predicted_activity}' does not allow for flat floor assumption."
            )

    except requests.exceptions.ConnectionError:
        logger.error(
            f"Connection to activity classifier at {activity_api_url} failed. Is the service running?"
        )
    except FileNotFoundError:
        logger.error(f"Video file not found: {video_path}")
    except requests.exceptions.HTTPError as e:
        # HTTPError is raised by response.raise_for_status() for 4xx/5xx status codes
        response = e.response
        if response is not None:
            status_code = response.status_code
            try:
                response_text = response.text
            except Exception:
                response_text = "Could not read response text"
            logger.error(
                f"HTTP error {status_code} from activity classifier at {activity_api_url}: {e}"
            )
            logger.error(f"Response text: {response_text}")
        else:
            logger.error(
                f"HTTP error from activity classifier at {activity_api_url}: {e}"
            )
    except requests.exceptions.RequestException as e:
        # For other request exceptions, try to get response details if available
        if hasattr(e, "response") and e.response is not None:
            status_code = e.response.status_code
            response_text = e.response.text
            logger.error(
                f"Request error {status_code} from activity classifier at {activity_api_url}: {e}"
            )
            logger.error(f"Response text: {response_text}")
        else:
            logger.error(f"Could not get activity prediction: {e}")

    return predicted_activity, flat_floor


import subprocess
import json
import requests
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session video segmentation
# ---------------------------------------------------------------------------

def _get_video_fps_and_duration(video_path):
    """Return (fps, duration_seconds) for a video file."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    duration_s = frame_count / fps if fps > 0 else 0.0
    return fps, duration_s


def _trim_clip(video_path, start_s, end_s, output_path):
    """Extract a clip from start_s to end_s using ffmpeg (stream copy, fast)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-i", video_path,
            "-t", str(end_s - start_s),
            "-c", "copy",
            output_path,
        ],
        check=True,
        capture_output=True,
    )


def _concat_clip_sources(clip_sources, output_path, clips_dir):
    """
    Concatenate (and trim where necessary) a list of clip sources.

    Args:
        clip_sources: list of (clip_path, rel_start_s, rel_end_s) — rel times
                      are relative to the start of that clip file
        output_path:  destination file for the concatenated result
        clips_dir:    directory used for temporary trim files (same filesystem
                      as clips, avoids cross-device move errors)
    """
    working_files = []  # files to pass to the concat demuxer
    temp_files = []     # intermediate trim files to delete after concat

    for clip_path, rel_start, rel_end in clip_sources:
        cap = cv2.VideoCapture(clip_path)
        clip_fps = cap.get(cv2.CAP_PROP_FPS)
        clip_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        clip_dur = clip_frames / clip_fps if clip_fps > 0 else 0

        needs_trim = rel_start > 0.05 or (clip_dur - rel_end) > 0.05
        if needs_trim:
            tmp_path = os.path.join(clips_dir, f"_tmp_{os.path.basename(clip_path)}")
            _trim_clip(clip_path, rel_start, rel_end, tmp_path)
            working_files.append(tmp_path)
            temp_files.append(tmp_path)
        else:
            working_files.append(clip_path)

    list_path = os.path.join(clips_dir, "_concat_list.txt")
    temp_files.append(list_path)
    with open(list_path, "w") as f:
        for p in working_files:
            f.write(f"file '{os.path.abspath(p)}'\n")

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", output_path],
            check=True, capture_output=True,
        )
        logger.info(f"[segmentation] Merged clip → {output_path}")
    finally:
        for p in temp_files:
            if os.path.exists(p):
                os.remove(p)


def _merge_window_labels(window_labels):
    """
    Convert window classifications into activity segments, handling compound
    labels (e.g. "walk,jump") by splitting the window proportionally.

    A compound label with n activities divides the window into n equal parts,
    one per activity in order of appearance.  This means boundary accuracy is
    window_size / n — the smaller the window, the more precise the boundary.

    Example with 10s windows:
        Window  0–10s  "walk"       → walk  0–10s
        Window 10–20s  "walk/jump"  → walk 10–15s, jump 15–20s
        Window 20–30s  "jump"       → jump 20–30s

    After merging consecutive same-label parts:
        walk:  0–15s   (from clip_0000[0:10s] + clip_0001[0:5s])
        jump: 15–30s   (from clip_0001[5:10s] + clip_0002[0:10s])

    Args:
        window_labels: list of (start_s, end_s, label, clip_path)

    Returns:
        list of (start_s, end_s, label, clip_sources) where
        clip_sources = [(clip_path, rel_start_s, rel_end_s), ...]
    """
    if not window_labels:
        return []

    # Expand each window into one sub-segment per activity in its label,
    # recording which portion of the clip file each sub-segment corresponds to.
    sub_segments = []
    for start_s, end_s, label, clip_path in window_labels:
        parts = [p.strip().lower() for p in label.split(",") if p.strip()]
        n = len(parts)
        l = end_s - start_s
        for i, part in enumerate(parts):
            sub_segments.append((
                start_s + i * l / n,        # absolute start
                start_s + (i + 1) * l / n,  # absolute end
                part,
                clip_path,
                i * l / n,                  # rel_start within clip
                (i + 1) * l / n,            # rel_end within clip
            ))

    # Merge consecutive sub-segments with the same label
    segments = []
    seg_start_s = sub_segments[0][0]
    seg_label = sub_segments[0][2]
    seg_clips = [(sub_segments[0][3], sub_segments[0][4], sub_segments[0][5])]

    for start_s, end_s, label, clip_path, rel_start, rel_end in sub_segments[1:]:
        if label != seg_label:
            segments.append((seg_start_s, start_s, seg_label, seg_clips))
            seg_start_s = start_s
            seg_label = label
            seg_clips = [(clip_path, rel_start, rel_end)]
        else:
            seg_clips.append((clip_path, rel_start, rel_end))

    segments.append((seg_start_s, sub_segments[-1][1], seg_label, seg_clips))
    return segments


def segment_session_video(
    video_path,
    window_size_s=10,
    min_segment_s=5,
):
    """
    Segment a session video into labelled exercise clips.

    Divides the video into non-overlapping fixed-length windows and classifies
    each with the VideoLLaMA activity classifier. Consecutive windows with the
    same label are merged into a single segment. Short 'other' segments
    (< min_segment_s) are dropped.

    Args:
        video_path:     path to the input video
        window_size_s:  duration of each classification window in seconds
        min_segment_s:  minimum duration to keep an 'other' segment

    Returns:
        list of dicts, each with keys:
            start_s, end_s       — segment time range in seconds
            start_frame, end_frame — corresponding frame indices
            label                — predicted activity string

    Raises:
        RuntimeError: if the classifier is unreachable (fail fast — all
                      segments must be labelled for the pipeline to proceed)
    """
    fps, duration_s = _get_video_fps_and_duration(video_path)
    logger.info(
        f"[segmentation] Video: {duration_s:.1f}s at {fps:.1f}fps | window={window_size_s}s"
    )

    # Short video — classify as a single segment without windowing
    if duration_s < window_size_s:
        logger.info("[segmentation] Video shorter than one window, treating as single segment")
        label, _ = predict_activity_from_video(video_path)
        if label is None:
            raise RuntimeError(
                "Activity classifier unreachable. Cannot segment video. "
                "Check that the VideoLLaMA service is running."
            )
        return [{
            "start_s": 0.0,
            "end_s": duration_s,
            "start_frame": 0,
            "end_frame": int(duration_s * fps),
            "label": label,
        }]

    # Non-overlapping windows; include a trailing partial window if long enough
    starts = []
    t = 0.0
    while t + window_size_s <= duration_s:
        starts.append(t)
        t += window_size_s
    if t < duration_s and (duration_s - t) >= min_segment_s:
        starts.append(t)

    logger.info(f"[segmentation] Classifying {len(starts)} windows ...")

    clips_dir = os.path.join("output_videos", "clips")
    if os.path.exists(clips_dir):
        import glob
        for f in glob.glob(os.path.join(clips_dir, "*.mp4")):
            os.remove(f)
    os.makedirs(clips_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    window_labels = []
    for i, start_s in enumerate(starts):
        end_s = min(start_s + window_size_s, duration_s)
        clip_path = os.path.join(clips_dir, f"{video_stem}_clip_{i:04d}_{int(start_s)}s.mp4")
        _trim_clip(video_path, start_s, end_s, clip_path)

        label, _ = predict_activity_from_video(clip_path)

        if label is None:
            raise RuntimeError(
                f"Activity classifier unreachable at window {i} "
                f"({start_s:.1f}–{end_s:.1f}s). Cannot segment video. "
                "Check that the VideoLLaMA service is running."
            )

        logger.info(f"[segmentation] Window {i:3d}: {start_s:6.1f}–{end_s:6.1f}s → '{label}' ({clip_path})")
        window_labels.append((start_s, end_s, label, clip_path))

    # Merge consecutive same-label windows into segments
    raw_segments = _merge_window_labels(window_labels)

    # Collect all window clip paths for cleanup after segments are built
    all_window_clips = [clip_path for _, _, _, clip_path in window_labels]

    # Filter, concatenate clips, and convert to output format
    segments = []
    seg_idx = 0
    for start_s, end_s, label, clip_sources in raw_segments:
        duration = end_s - start_s
        is_other = label.lower() in ("other", "none", "unknown")
        if is_other and duration < min_segment_s:
            logger.info(
                f"[segmentation] Dropping short '{label}' segment "
                f"{start_s:.1f}–{end_s:.1f}s ({duration:.1f}s < {min_segment_s}s)"
            )
            continue

        merged_clip = os.path.join(
            clips_dir, f"{video_stem}_seg_{seg_idx:02d}_{label}.mp4"
        )
        _concat_clip_sources(clip_sources, merged_clip, clips_dir)
        seg_idx += 1

        segments.append({
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(start_s * fps),
            "end_frame": int(end_s * fps),
            "label": label,
            "clip_path": merged_clip,
        })

    # Delete all window clips now that segments have been merged
    for clip_path in all_window_clips:
        if os.path.exists(clip_path):
            os.remove(clip_path)
    logger.info(f"[segmentation] Deleted {len(all_window_clips)} window clip(s)")

    logger.info(f"[segmentation] {len(segments)} segment(s) after filtering:")
    for i, seg in enumerate(segments):
        logger.info(
            f"[segmentation]   [{i}] {seg['start_s']:.1f}–{seg['end_s']:.1f}s "
            f"({seg['end_s']-seg['start_s']:.1f}s) '{seg['label']}'"
        )

    return segments


def get_rotation_metadata(video_path):
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream_tags=rotate",
            "-of",
            "json",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        rotate_tag = data.get("streams", [{}])[0].get("tags", {}).get("rotate")
        if rotate_tag:
            rotation_angle = int(rotate_tag)
            logger.info(f"Rotation metadata found: {rotation_angle} degrees")
            return rotation_angle
    except Exception as e:
        logger.warning(f"No rotation metadata found or ffprobe failed: {e}")
    return None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.convert_to_avi import prepare_video

    video_path = "/ceph/Dataset/QEVD-FIT-COACH/long_range_videos/0000.mp4"
    video_path, rotation = prepare_video(video_path, output_dir="output_videos")
    print(f"Video ready: {video_path} (rotation={rotation})")

    print("\nSegmenting session video:")
    segments = segment_session_video(video_path)
    for i, seg in enumerate(segments):
        print(f"  [{i}] {seg['start_s']:.1f}–{seg['end_s']:.1f}s  '{seg['label']}'  → {seg['clip_path']}")
