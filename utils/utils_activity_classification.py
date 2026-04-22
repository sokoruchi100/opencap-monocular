import requests
import logging
import os
import subprocess
import json
import numpy as np
from loguru import logger
from decouple import config as env_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def predict_activity_from_feedbacks(feedbacks):
    """
    Predict the activity label from a list of feedback strings by calling
    the VideoLLaMA /predict-with-feedbacks endpoint.

    Args:
        feedbacks (list): feedback strings recorded during the exercise segment

    Returns:
        tuple: (predicted_activity, flat_floor)
            predicted_activity — str label or None if the call fails
            flat_floor         — always True (dataset videos have flat floors)
    """
    predicted_activity = None
    flat_floor = True

    try:
        videollama_base = env_config("VIDEOLLAMA_URL", default="http://localhost:8400").rstrip("/")
        activity_api_url = f"{videollama_base}/predict-with-feedbacks"
        data = {"feedbacks": ", ".join(feedbacks)}
        response = requests.post(activity_api_url, data=data, timeout=120)
        response.raise_for_status()
        predicted_activity = response.json().get("predicted_activity")
        logger.info(f"Predicted activity: {predicted_activity}")

    except requests.exceptions.ConnectionError:
        logger.error(f"Connection to activity classifier at {activity_api_url} failed. Is the service running?")
    except requests.exceptions.HTTPError as e:
        resp = e.response
        logger.error(f"HTTP error {resp.status_code if resp else '?'} from activity classifier: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Could not get activity prediction: {e}")

    return predicted_activity, flat_floor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_video_fps_and_duration(video_path):
    """Return (fps, duration_seconds) using ffprobe (cv2 frame count is unreliable for AVI)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,nb_frames:format=duration",
            "-of", "json",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)
    stream = info.get("streams", [{}])[0]

    num, den = (int(x) for x in stream.get("r_frame_rate", "0/1").split("/"))
    fps = num / den if den else 0.0

    duration_s = float(info.get("format", {}).get("duration", 0) or 0)
    if duration_s == 0 and fps > 0:
        duration_s = int(stream.get("nb_frames", 0) or 0) / fps

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


# ---------------------------------------------------------------------------
# Timestamp / event extraction
# ---------------------------------------------------------------------------

def get_video_events(video_id_str, json_path, base_npy_path):
    """
    Extract event timestamps and associated feedback from the dataset metadata.

    Args:
        video_id_str:  4-digit string identifying the video (e.g. "0000")
        json_path:     path to feedbacks_long_range.json
        base_npy_path: directory containing {video_id_str}_timestamps.npy files

    Returns:
        (events, timestamp_duration)
            events — list of {start_s, end_s, unique_feedbacks} dicts,
                     with times relative to the first npy timestamp
            timestamp_duration — total span of the npy timestamps in seconds
    """
    with open(json_path, "r") as f:
        all_json_data = json.load(f)

    data_item = next(
        (item for item in all_json_data
         if isinstance(item, dict)
         and item.get("long_range_video_file") == f"./long_range_videos/{video_id_str}.mp4"),
        None,
    )
    if data_item is None:
        logging.error(f"Data for video ID '{video_id_str}' not found in JSON.")
        return [], 0

    is_transition_list      = data_item.get("is_transition", [])
    feedback_timestamps_list = data_item.get("feedback_timestamps", [])
    feedbacks_list           = data_item.get("feedbacks", [])

    if not is_transition_list or not feedback_timestamps_list or not feedbacks_list:
        logging.error(f"Missing fields for video ID '{video_id_str}'.")
        return [], 0

    npy_path = os.path.join(base_npy_path, f"{video_id_str}_timestamps.npy")
    if not os.path.exists(npy_path):
        logging.error(f"NPY file not found: {npy_path}")
        return [], 0

    timestamps_raw = np.load(npy_path)
    video_ts_sec = timestamps_raw / 1e9 + 28800
    timestamp_duration = float(video_ts_sec[-1] - video_ts_sec[0]) if len(video_ts_sec) > 0 else 0

    # Build (start, end) pairs from consecutive True entries in is_transition
    event_timestamps = []
    start_ts = None
    for is_trans, ts in zip(is_transition_list, feedback_timestamps_list):
        if is_trans:
            if start_ts is None:
                start_ts = ts
            else:
                event_timestamps.append((start_ts, ts))
                start_ts = ts

    events = []
    frame_rate = 30.0
    for i, (start_ts, end_ts) in enumerate(event_timestamps):
        closest_start_frame_idx = np.argmin(np.abs(start_ts - video_ts_sec))
        closest_end_frame_idx = np.argmin(np.abs(end_ts - video_ts_sec))

        event_feedbacks = feedbacks_list[closest_start_frame_idx : closest_end_frame_idx + 1]
        unique_non_empty_feedbacks = sorted(list(set(f for f in event_feedbacks if f)))

        start_time_video_sec = closest_start_frame_idx / frame_rate
        end_time_video_sec = closest_end_frame_idx / frame_rate

        events.append({
            'start_time_sec': start_time_video_sec,
            'end_time_sec': end_time_video_sec,
            'unique_feedbacks': unique_non_empty_feedbacks
        })
        
    logging.info(f"Video ID '{video_id_str}': {len(events)} events, timestamp span {timestamp_duration:.2f}s")
    return events, timestamp_duration


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_session_video(
    video_path,
    video_id_str="0000",
    json_path="/ceph/Dataset/QEVD-FIT-COACH/feedbacks_long_range.json",
    npy_base_path="/ceph/Dataset/QEVD-FIT-COACH/long_range_videos/",
):
    """
    Segment a session video into labelled exercise clips using dataset metadata.

    Event boundaries come from the transition timestamps in the npy file.

    Returns:
        list of (start_s, end_s, start_frame, end_frame, label, clip_path) tuples
    """
    video_events, target_duration = get_video_events(video_id_str, json_path, npy_base_path)

    fps, duration_s = _get_video_fps_and_duration(video_path)
    logger.info(
        f"[segmentation] Video: {duration_s:.1f}s at {fps:.1f}fps | "
        f"timestamp span: {target_duration:.1f}s"
    )

    clips_dir = os.path.join("output_videos", "clips")
    if os.path.exists(clips_dir):
        import glob
        for f in glob.glob(os.path.join(clips_dir, "*.mp4")):
            os.remove(f)
    os.makedirs(clips_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    if not video_events:
        logger.warning(f"No video events found for video ID '{video_id_str}'.")
        return []

    logger.info(f"[segmentation] Processing {len(video_events)} segment(s) ...")
    segments = []
    for i, event in enumerate(video_events):
        start_s   = event["start_time_sec"]
        end_s     = event["end_time_sec"]
        feedbacks = event["unique_feedbacks"]
        feedback_str = ", ".join(feedbacks) if feedbacks else "no feedback"
        logger.info(f"[segmentation] Event {i}: {start_s:.1f}–{end_s:.1f}s | {feedback_str}")

        clip_path = os.path.join(clips_dir, f"{video_stem}_clip_{i:04d}_{int(start_s)}s.mp4")
        _trim_clip(video_path, start_s, end_s, clip_path)

        label, _ = predict_activity_from_feedbacks(feedbacks)
        if label is None:
            raise RuntimeError(
                f"Activity classifier unreachable at event {i} "
                f"({start_s:.1f}–{end_s:.1f}s). Check that the VideoLLaMA service is running."
            )

        logger.info(f"[segmentation] Event {i:3d}: {start_s:6.1f}–{end_s:6.1f}s → '{label}'")
        segments.append({
            "start_s":     start_s,
            "end_s":       end_s,
            "start_frame": int(start_s * fps),
            "end_frame":   int(end_s * fps),
            "label":       label,
            "clip_path":   clip_path,
        })

    logger.info(f"[segmentation] {len(segments)} segment(s):")
    for i, seg in enumerate(segments):
        logger.info(f"[segmentation]   [{i}] {seg['start_s']:.1f}–{seg['end_s']:.1f}s ({seg['end_s']-seg['start_s']:.1f}s) '{seg['label']}'")

    return segments


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
