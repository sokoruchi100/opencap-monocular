import os
import cv2
import sys


def convert_to_avi(inputPath, outputPath=None, frameRate=None, quality=0, rotation=None):
    """
    Convert video format and optionally rotate frames
    
    Args:
        inputPath:  Path to input video
        outputPath: Path to output video (if none, will be the same as inputPath but with .avi extension)
        frameRate: Optional frame rate limit (e.g., 60)
        quality: Quality setting for output (0=best, default)
        rotation: Rotation angle in degrees (0, 90, 180, 270). If None, uses ffmpeg conversion.
                  If specified, uses OpenCV to rotate frames during conversion.
    """
    if outputPath is None:
        outputPath = inputPath.replace(".mov", ".avi").replace(".mp4", ".avi")
    
    # If rotation is specified and needs frame rotation, use OpenCV to rotate frames (like old working version)
    # Old version: should_rotate=True for 90/270 meant frames needed rotation to upright
    # New version: rotation=90/270 means frames are rotated, need to rotate them to upright
    # Rotation 0/180: frames are already in correct orientation, no rotation needed
    print(f'{rotation=}')
    if rotation is not None and rotation in [90, 270]:
        return _convert_with_rotation(inputPath, outputPath, rotation, frameRate, quality)
    else:    
        # Use ffmpeg for simple conversion (no rotation needed for 0, 180, or None)
        cmd_fr = '' if frameRate is None else f' -r {frameRate} '
        CMD = f"ffmpeg -loglevel error -y -i {inputPath}{cmd_fr} -q:v {quality} {outputPath}"
        
        if not os.path.exists(outputPath):
            os.system(CMD)
        
        return outputPath


def _convert_with_rotation(input_path, output_path, rotation, frame_rate=None, quality=0):
    """
    Convert video using OpenCV and rotate frames
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        rotation: Rotation angle in degrees (90, 180, 270)
        frame_rate: Optional frame rate limit
        quality: Quality setting (0=best)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate is not None:
        fps = min(fps, frame_rate)
    
    # Determine output dimensions based on rotation
    if rotation == 90:
        output_width = height
        output_height = width
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif rotation == 270:
        output_width = height
        output_height = width
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    elif rotation == 180:
        output_width = width
        output_height = height
        rotate_code = cv2.ROTATE_180
    else:
        output_width = width
        output_height = height
        rotate_code = None
    
    # Use XVID codec (same as old version)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video file: {output_path}")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if rotate_code is not None:
            rotated_frame = cv2.rotate(frame, rotate_code)
            out.write(rotated_frame)
        else:
            out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return output_path

def prepare_video(video_path, output_dir="output_videos"):
    """
    Detect rotation and convert to AVI if the input is MOV or MP4.

    Call this once before segmentation so all downstream clip trimming works
    on a consistent AVI file.

    Args:
        video_path:  path to the original video file
        output_dir:  directory where the converted AVI is written

    Returns:
        tuple: (video_path, rotation)
            video_path — original path if already AVI, otherwise path to the
                         newly created AVI
            rotation   — rotation angle in degrees detected from metadata
                         (0, 90, 180, 270, or None)
    """
    from utils.utilsCameraPy3 import getVideoRotation

    rotation = getVideoRotation(video_path)

    if video_path.lower().endswith((".mov", ".mp4")):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        avi_path = os.path.join(output_dir, f"{video_name}.avi")
        os.makedirs(output_dir, exist_ok=True)
        video_path = convert_to_avi(video_path, outputPath=avi_path, rotation=rotation)

    return video_path, rotation


if __name__ == "__main__":
    video_num = sys.argv[1] if len(sys.argv) > 1 else "0000"

    video_path = f"/ceph/Dataset/QEVD-FIT-COACH/long_range_videos/{video_num}.mp4"
    if not os.path.exists(video_path):
        print(f"{video_path} does not exist")
        exit()

    output_video_path = f"/storage/Daniel/opencap-monocular/output_videos/{video_num}.avi"
    prepare_video(video_path)