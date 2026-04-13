#!/usr/bin/env python3
"""
Standalone script to run the mono pipeline without the API.
This is useful for debugging with breakpoints.

Usage:
    python run_mono_standalone.py
"""

import os
import time
import yaml
import hashlib
import sys
from loguru import logger
from hashlib import md5
from pathlib import Path
from typing import Optional
import json

# Import the same modules used in mono_api.py
from optimization import run_optimization
from WHAM.demo import main_wham
from visualization.utils import generateVisualizerJson
from visualization.automation import automate_recording
from utils.convert_to_avi import convert_to_avi, prepare_video
from utils.utilsCameraPy3 import getVideoRotation
from utils.tracking_filters import InsufficientFullBodyKeypointsError
from utils.utils_activity_classification import segment_session_video

# Used to extract the height_m, mass_kg, and sex, of Subjects
METADATA_PATH = "/ceph/Dataset/QEVD-FIT-COACH/QEVD-FIT-COACH_body_info.json"

# Import enhanced logging (optional)
try:
    from deployment.logging_config import setup_logging

    setup_logging("api")
except ImportError:
    logger.warning("Enhanced logging not available, using basic logging")

# Get repo path
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def generate_request_hash(video_path, metadata):
    """Generate a unique hash to identify a processing request based on its parameters."""
    # Create a string with all relevant parameters
    # Use 'sex' key consistently, providing defaults if missing
    sex = metadata.get("sex", "unknown")
    height_m = metadata.get("height_m", 0.0)
    mass_kg = metadata.get("mass_kg", 0.0)
    key_string = f"{video_path}_{height_m}_{mass_kg}_{sex}"
    # Generate a hash
    return hashlib.md5(key_string.encode()).hexdigest()


def resolve_intrinsics_from_metadata(metadata: dict, repo_path: str) -> str:
    """Resolve device-specific intrinsics from metadata (same logic as mono_api)."""
    device_model = None
    try:
        device_model = metadata.get("iphoneModel", {}).get("Cam0", "")
        device_model = device_model.replace("iphone", "iPhone").replace(
            "ipad", "iPad"
        )
    except Exception:
        pass

    default_intrinsics = os.path.join(
        repo_path, "camera_intrinsics/iPhone13,3/Deployed/cameraIntrinsics.pickle"
    )
    if device_model:
        device_intrinsics = os.path.join(
            repo_path,
            f"camera_intrinsics/{device_model}/Deployed/cameraIntrinsics.pickle",
        )
        if os.path.exists(device_intrinsics):
            logger.info(
                f"Using device-specific intrinsics for {device_model}: {device_intrinsics}"
            )
            return device_intrinsics
        logger.warning(
            f"No intrinsics found for device '{device_model}' at {device_intrinsics}. "
            f"Falling back to {default_intrinsics}"
        )
        return default_intrinsics

    logger.warning(
        f"Could not determine device model from metadata. "
        f"Falling back to {default_intrinsics}"
    )
    return default_intrinsics


def run_mono_standalone(
    video_path: str,
    metadata_path: str,
    calib_path: str,
    intrinsics_path: Optional[str] = None,
    estimate_local_only: bool = True,
    rerun: bool = False,
    session_id: Optional[str] = None,
    activity: Optional[str] = None,
):
    """
    Run the mono pipeline standalone (without API).

    Args:
        video_path: Path to the input video file
        metadata_path: Path to the metadata JSON file
        calib_path: Path to the calibration file
        intrinsics_path: Path to intrinsics pickle; if None or empty, resolved from
            metadata iphoneModel.Cam0 like the API (camera_intrinsics/.../cameraIntrinsics.pickle).
        estimate_local_only: Whether to estimate local only
        rerun: Whether to rerun even if cached results exist
        session_id: Optional session ID to use as case_id
        activity: Optional activity type (e.g., "walking", "sitting")

    Returns:
        Dictionary with results similar to the API response
    """
    # Load metadata
    metadata = {}
    video_name = os.path.basename(video_path).split(".")[0]
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding='utf-8') as f:
            metadata = json.load(f).get(video_name)

    # Rename keys based on mapping; keep original if not in mapping
    mapping = {'height': 'height_m', 'mass': 'mass_kg', 'gender': 'sex'}
    metadata = {mapping.get(k, k): v for k, v in metadata.items()}

    if not intrinsics_path:
        intrinsics_path = resolve_intrinsics_from_metadata(metadata, repo_path)

    # Defaults
    height_m = metadata.get("height_m", 1.70)  # Default height_m if not found
    mass_kg = metadata.get("mass_kg", 70.0)  # Default mass_kg if not found
    sex = metadata.get("sex", "male")  # Default sex if not found
    logger.info(f"height: {height_m} m, mass: {mass_kg} kg, sex: {sex}")

    # If rerun is False, check for cached results
    if not rerun:
        request_hash = generate_request_hash(video_path, metadata)
        results_dir = os.path.join(repo_path, "results")

        if os.path.exists(results_dir):
            for case_dir in os.listdir(results_dir):
                cache_file = os.path.join(results_dir, case_dir, "request_hash.txt")
                if os.path.exists(cache_file):
                    with open(cache_file, "r") as f:
                        stored_hash = f.read().strip()

                    if stored_hash == request_hash:
                        logger.info(f"Found cached results in {case_dir}")
                        results_path = os.path.join(results_dir, case_dir, video_name)

                        # Check if key files exist
                        ik_file_path = os.path.join(results_path, "ik_results.pkl")
                        output_mono_json_path = os.path.join(results_path, "mono.json")
                        output_video_path = os.path.join(
                            results_path, "viewer_mono.webm"
                        )

                        if os.path.exists(ik_file_path) and os.path.exists(
                            output_mono_json_path
                        ):
                            return {
                                "message": "Mono pipeline completed successfully (cached result)!",
                                "ik_file_path": ik_file_path,
                                "case_id": case_dir,
                                "case_dir": os.path.join(results_dir, case_dir),
                                "visualization": {
                                    "created": True,
                                    "json_path": output_mono_json_path,
                                    "video_path": (
                                        output_video_path
                                        if os.path.exists(output_video_path)
                                        else None
                                    ),
                                },
                            }

    # Create a case directory based on timestamp and request parameters
    if session_id:
        case_num = session_id
        logger.info(f"Using provided session_id as case_id: {case_num}")
    else:
        timestamp = int(time.time())
        # Hash includes parameters that might affect processing but aren't part of the input 'identity' for caching
        case_hash = md5(f"{timestamp}_{estimate_local_only}".encode()).hexdigest()[
            :8
        ]  # rerun is handled by the cache check logic
        case_num = f"{timestamp}_{case_hash}"
        logger.info(f"Generated new case_id: {case_num}")

    # Create case directory structure
    case_dir = os.path.join(repo_path, "results", case_num)
    os.makedirs(case_dir, exist_ok=True)

    logger.info(f"Created case directory: {case_dir}")

    # Create output directory
    video_name = os.path.basename(video_path).split(".")[0]  # Remove file extension
    # trial_path = os.path.join(case_dir, video_name)
    trial_path = case_dir
    logger.info(f"trial_path: {trial_path}")
    if not os.path.exists(trial_path):
        os.makedirs(trial_path)

    start_time = time.time()

    logger.info(f"video_path: {video_path}")
    logger.info(f"metadata_path: {metadata_path}")
    logger.info(f"calib_path: {calib_path}")
    logger.info(f"intrinsics_path: {intrinsics_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    # Detect rotation and convert to AVI once before segmentation.
    # Each segment clip is then trimmed from this AVI.
    video_path, rotation = prepare_video(video_path, output_dir="output_videos")
    logger.info(f"Rotation: {rotation} | video ready at: {video_path}")

    # --- Session segmentation -------------------------------------------------
    # Classify overlapping windows and split the video into per-exercise segments.
    # Each segment is processed independently through WHAM + optimization.
    logger.info("Segmenting session video into exercise clips ...")
    segments = segment_session_video(video_path, video_id_str=video_name)  # raises RuntimeError if classifier down
    logger.info(f"Found {len(segments)} segment(s) to process.")
    # --------------------------------------------------------------------------

    segment_results = []

    for seg_idx, seg in enumerate(segments):
        # WHAM names the output dir after the clip stem, so seg_dir is derived from it
        seg_clip_path = seg["clip_path"]
        clip_stem = os.path.splitext(os.path.basename(seg_clip_path))[0]
        seg_dir = os.path.join(case_dir, clip_stem)

        logger.info(
            f"--- Segment {seg_idx}: '{seg['label']}' "
            f"{seg['start_s']:.1f}–{seg['end_s']:.1f}s → {seg_dir}"
        )

        # Run WHAM — it creates case_dir/{clip_stem}/ from the clip filename
        inputs_wham = {
            "calib_path": calib_path,
            "video_path": seg_clip_path,
            "output_path": case_dir,
            "visualize": True,
            "save_pkl": True,
            "run_smplify": True,
            "rerun": rerun,
            "estimate_local_only": estimate_local_only,
        }

        logger.info(
            f"Starting WHAM for segment {seg_idx} ..."
        )
        try:
            main_wham(**inputs_wham)
        except InsufficientFullBodyKeypointsError as e:
            logger.warning(
                f"Segment {seg_idx} skipped after WHAM keypoint gate: {e}"
            )
            segment_results.append({
                "segment": seg_idx,
                "label": seg["label"],
                "aborted": True,
                "aborted_stage": "WHAM_preprocess",
                "error": str(e),
            })
            continue
        logger.info(f"WHAM done for segment {seg_idx}. Time so far: {time.time() - start_time:.1f}s")

        # WHAM writes output to case_dir/seg_name/ (clip stem == seg_name)
        results_path = seg_dir

        # Run optimization
        logger.info(f"Running optimization for segment {seg_idx} ...")
        inputs_optimization = {
            "data_dir": results_path,
            "trial_name": os.path.basename(results_path),
            "height_m": height_m,
            "mass_kg": mass_kg,
            "sex": sex,
            "intrinsics_pth": intrinsics_path,
            "run_opensim_original_wham": True,
            "run_opensim_opt2": True,
            "use_gpu": True,
            "static_cam": True,
            "n_iter_opt2": 75,
            "print_loss_terms": False,
            "plotting": True,
            "save_video_debug": False,
            "output_path": results_path,
            "video_path": seg_clip_path,
            "activity": seg["label"],  # pass the classified label directly
            "rotation": rotation,
            "create_contact_visualizations": False,
        }

        output_paths = run_optimization(**inputs_optimization)
        logger.info(f"Optimization done for segment {seg_idx}. Time so far: {time.time() - start_time:.1f}s")

        # Generate visualization
        logger.info(f"Generating visualization for segment {seg_idx} ...")
        model_mono_sub_folder = os.path.join(results_path, "OpenSim", "Model")
        model_mono_file = None
        ik_motion_file = None

        if os.path.exists(model_mono_sub_folder):
            model_folders = [x for x in os.listdir(model_mono_sub_folder) if "wham" not in x]
            if model_folders:
                model_mono_file = os.path.join(
                    results_path, "OpenSim", "Model", model_folders[0],
                    "LaiUhlrich2022_scaled_no_patella.osim",
                )
                if not os.path.exists(model_mono_file):
                    model_mono_file = None

        ik_motion_sub_folder = os.path.join(results_path, "OpenSim", "IK")
        if os.path.exists(ik_motion_sub_folder):
            ik_folders = [x for x in os.listdir(ik_motion_sub_folder) if "wham" not in x]
            if ik_folders:
                ik_motion_folder = os.path.join(results_path, "OpenSim", "IK", ik_folders[0])
                mot_files = [x for x in os.listdir(ik_motion_folder) if x.endswith(".mot")]
                if mot_files:
                    ik_motion_file = os.path.join(ik_motion_folder, mot_files[0])
                else:
                    logger.error(f"No .mot file in {ik_motion_folder}")

        output_mono_json_path = os.path.join(results_path, "mono.json")
        output_video_path = os.path.join(results_path, "viewer_mono.webm")
        jsonOutputPath = None
        visualization_created = False

        if model_mono_file and ik_motion_file:
            try:
                jsonOutputPath = generateVisualizerJson(
                    modelPath=model_mono_file,
                    ikPath=ik_motion_file,
                    jsonOutputPath=output_mono_json_path,
                    vertical_offset=0,
                )
                logger.info(f"Visualization JSON: {jsonOutputPath}")
                visualization_created = True
            except Exception as e:
                logger.error(f"Visualization failed for segment {seg_idx}: {e}")
        else:
            logger.warning(f"Skipping visualization for segment {seg_idx}: IK or model missing")

        segment_results.append({
            "segment": seg_idx,
            "label": seg["label"],
            "start_s": seg["start_s"],
            "end_s": seg["end_s"],
            "ik_file_path": output_paths.get("ik_results_file"),
            "trc_file_path": output_paths.get("trc_file"),
            "scaled_model_file_path": output_paths.get("scaled_model_file"),
            "json_file_path": jsonOutputPath,
            "video_file_path": output_paths.get("trimmed_video"),
            "seg_dir": seg_dir,
            "visualization": {
                "created": visualization_created,
                "json_path": output_mono_json_path if visualization_created else None,
                "video_path": output_video_path if visualization_created and os.path.exists(output_video_path) else None,
            },
        })

    # Store request hash
    request_hash = generate_request_hash(video_path, metadata)
    with open(os.path.join(case_dir, "request_hash.txt"), "w") as f:
        f.write(request_hash)
    logger.info(f"Stored request hash {request_hash}")

    return {
        "message": "Mono pipeline completed successfully!",
        "case_id": case_num,
        "case_dir": case_dir,
        "metadata_path": metadata_path,
        "segments": segment_results,
    }


if __name__ == "__main__":
    # Example usage - modify these paths as needed
    # You can set breakpoints anywhere in the run_mono_standalone function

    video_dir = "/ceph/Dataset/QEVD-FIT-COACH/long_range_videos/"
    video_name = sys.argv[1] if len(sys.argv) > 1 else "0000"
    video_path = os.path.join(video_dir, video_name + ".mp4") if os.path.isfile(os.path.join(video_dir, video_name + ".mp4")) else None
    if not video_path:
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    calib_path = "" # none exists
    # intrinsics_path: omit or None to resolve from metadata iphoneModel.Cam0 (mono_api behavior)

    # Optional: Initialize WHAM model (similar to API startup)
    try:
        from WHAM.demo import initialize_wham

        logger.info("Loading WHAM model...")
        initialize_wham()
        logger.info("WHAM model loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not initialize WHAM model: {e}")

    # Run the pipeline
    result = run_mono_standalone(
        video_path=video_path,
        metadata_path=METADATA_PATH,
        calib_path=calib_path,
        estimate_local_only=True,
        rerun=False,
        session_id=None,  # Set to a string to use a specific session ID
        activity=None,  # Set to "walking", "sitting", etc. if needed
    )

    # Print results
    logger.info("Pipeline completed!")
    logger.info(f"Results: {result}")
