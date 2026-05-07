import sys
import time
import subprocess
import argparse
import os

# --- Constants ---
ASPECT_RATIO = 9 / 16

# Lazy-loaded models — initialized on first use so that importing the module
# or running --help doesn't trigger heavyweight model loading.
_model = None
_face_cascade = None

def get_yolo_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO('yolov8n.pt')
    return _model

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _face_cascade

def analyze_scene_content(video_path, scene_start_time, scene_end_time):
    """
    Analyzes the middle frame of a scene to detect people and faces.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    start_frame = scene_start_time.get_frames()
    end_frame = scene_end_time.get_frames()
    middle_frame_number = int(start_frame + (end_frame - start_frame) / 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)

    # Run detection per sampled frame, keeping per-frame results so we can
    # track each person across samples and measure their motion.
    per_frame_detections = []
    for fn in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue
        per_frame_detections.append(_detect_people_in_frame(frame))
    cap.release()

    if not per_frame_detections:
        return []

    # Greedy IoU tracking: link detections across consecutive sampled frames
    # so we can build per-person trajectories.
    tracks = []  # each: person_box (latest), face_box, boxes[], centers[]
    for frame_dets in per_frame_detections:
        used = [False] * len(frame_dets)
        # Match existing tracks to detections in this frame
        for track in tracks:
            best_iou, best_idx = 0.0, -1
            for i, det in enumerate(frame_dets):
                if used[i]:
                    continue
                iou = _iou(track['person_box'], det['person_box'])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= 0.3 and best_idx >= 0:
                det = frame_dets[best_idx]
                used[best_idx] = True
                track['person_box'] = det['person_box']
                track['boxes'].append(det['person_box'])
                cx = (det['person_box'][0] + det['person_box'][2]) / 2
                cy = (det['person_box'][1] + det['person_box'][3]) / 2
                track['centers'].append((cx, cy))
                if track['face_box'] is None and det['face_box'] is not None:
                    track['face_box'] = det['face_box']
        # Unmatched detections start new tracks
        for i, det in enumerate(frame_dets):
            if used[i]:
                continue
            cx = (det['person_box'][0] + det['person_box'][2]) / 2
            cy = (det['person_box'][1] + det['person_box'][3]) / 2
            tracks.append({
                'person_box': list(det['person_box']),
                'face_box': det['face_box'],
                'boxes': [list(det['person_box'])],
                'centers': [(cx, cy)],
            })

    # Compute motion score for each track. Normalized by the person's average
    # box width so a small fidgety subject and a tall pacing subject are
    # comparable. Score ~ "body widths travelled across the scene samples".
    for t in tracks:
        if len(t['centers']) < 2:
            t['motion'] = 0.0
        else:
            total = 0.0
            for i in range(1, len(t['centers'])):
                dx = t['centers'][i][0] - t['centers'][i - 1][0]
                dy = t['centers'][i][1] - t['centers'][i - 1][1]
                total += (dx * dx + dy * dy) ** 0.5
            widths = [max(1, b[2] - b[0]) for b in t['boxes']]
            avg_w = sum(widths) / len(widths)
            t['motion'] = total / avg_w

    # Strip tracking metadata down to the public shape; keep `motion`.
    return [{
        'person_box': t['person_box'],
        'face_box': t['face_box'],
        'motion': t['motion'],
    } for t in tracks]


def detect_scenes(video_path, downscale=0, frame_skip=0):
    """Detect scene boundaries.

    Args:
        video_path: Path to the video file.
        downscale: Downscale factor for processing (0 = auto-detect based on
                   resolution).  Higher values are faster but may miss subtle cuts.
        frame_skip: Number of frames to skip between each processed frame.
                    0 = process every frame (default, most accurate).
    """
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    if downscale > 0:
        video_manager.set_downscale_factor(downscale)
    else:
        video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=True,
                                frame_skip=frame_skip)
    scene_list = scene_manager.get_scene_list()
    fps = video_manager.get_framerate()
    video_manager.release()
    return scene_list, fps

def get_enclosing_box(boxes):
    if not boxes:
        return None
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return [min_x, min_y, max_x, max_y]

def decide_cropping_strategy(scene_analysis, frame_height, motion_threshold=0.5):
    """Pick a cropping strategy for a scene.

    Args:
        scene_analysis: list of detections from analyze_scene_content. Each
            entry may include a `motion` score (body-widths travelled across
            sampled frames).
        frame_height: source frame height, used to size the crop window.
        motion_threshold: minimum motion score for a single subject to "win"
            the scene over the group bounding box. Lower = more aggressive
            tracking of the most active person. 0 = always track the most
            active person; very high = legacy group-only behavior.
    """
    num_people = len(scene_analysis)
    if num_people == 0:
        return 'LETTERBOX', None
    if num_people == 1:
        target_box = scene_analysis[0]['face_box'] or scene_analysis[0]['person_box']
        return 'TRACK', target_box

    # Multiple people: if one is clearly more active than the others, focus
    # on them rather than averaging the whole group.
    motions = [obj.get('motion', 0.0) for obj in scene_analysis]
    max_motion = max(motions)
    if max_motion >= motion_threshold:
        idx = motions.index(max_motion)
        target = scene_analysis[idx]
        target_box = target['face_box'] or target['person_box']
        return 'TRACK', target_box

    # Otherwise, fall back to enclosing the group (or letterboxing if too wide).
    person_boxes = [obj['person_box'] for obj in scene_analysis]
    group_box = get_enclosing_box(person_boxes)
    group_width = group_box[2] - group_box[0]
    max_width_for_crop = frame_height * ASPECT_RATIO
    if group_width < max_width_for_crop:
        return 'TRACK', group_box
    else:
        return 'LETTERBOX', None

def calculate_crop_box(target_box, frame_width, frame_height):
    target_center_x = (target_box[0] + target_box[2]) / 2
    crop_height = frame_height
    crop_width = int(crop_height * ASPECT_RATIO)
    x1 = int(target_center_x - crop_width / 2)
    y1 = 0
    x2 = int(target_center_x + crop_width / 2)
    y2 = frame_height
    if x1 < 0:
        x1 = 0
        x2 = crop_width
    if x2 > frame_width:
        x2 = frame_width
        x1 = frame_width - crop_width
    return x1, y1, x2, y2

def get_video_properties(video_path):
    """Returns (width, height, fps) from OpenCV — the same backend that reads frames."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

def get_media_info(video_path):
    """Returns a dict with human-readable info about the input file."""
    info = {}
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries',
             'format=duration,size',
             '-show_entries', 'stream=codec_name,codec_type,width,height,r_frame_rate',
             '-of', 'json', video_path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            fmt = data.get('format', {})
            info['duration'] = float(fmt.get('duration', 0))
            info['size_bytes'] = int(fmt.get('size', 0))
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video' and 'video_codec' not in info:
                    info['video_codec'] = stream.get('codec_name', 'unknown')
                    info['width'] = stream.get('width', 0)
                    info['height'] = stream.get('height', 0)
                    rate = stream.get('r_frame_rate', '0/1')
                    parts = rate.split('/')
                    if len(parts) == 2 and int(parts[1]) != 0:
                        info['fps'] = round(int(parts[0]) / int(parts[1]), 2)
                    else:
                        info['fps'] = float(parts[0])
                elif stream.get('codec_type') == 'audio' and 'audio_codec' not in info:
                    info['audio_codec'] = stream.get('codec_name', 'unknown')
    except (FileNotFoundError, ValueError, KeyError):
        pass
    return info

def format_duration(seconds):
    """Formats seconds into a human-readable string like '1h 32m 15s'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"

def format_file_size(size_bytes):
    """Formats bytes into a human-readable string."""
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    elif size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"

def has_audio_stream(video_path):
    """Uses ffprobe to check whether the file contains an audio stream."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a',
             '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        return result.returncode == 0 and 'audio' in result.stdout
    except FileNotFoundError:
        # ffprobe not available — assume audio exists and let ffmpeg handle it
        return True

def get_stream_start_time(video_path, stream_type='v:0'):
    """Returns the start_time of a stream in seconds (0.0 if unavailable)."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', stream_type,
             '-show_entries', 'stream=start_time', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass
    return 0.0

def is_variable_frame_rate(video_path):
    """Uses ffprobe to check if the video has a variable frame rate."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return False
        # ffprobe returns "num/den" for both r_frame_rate and avg_frame_rate
        parts = result.stdout.strip().split(',')
        if len(parts) < 2:
            return False
        def parse_rate(s):
            nums = s.strip().split('/')
            if len(nums) == 2 and int(nums[1]) != 0:
                return int(nums[0]) / int(nums[1])
            return float(nums[0])
        r_fps = parse_rate(parts[0])
        avg_fps = parse_rate(parts[1])
        # If the real frame rate and average frame rate differ significantly, it's VFR
        return abs(r_fps - avg_fps) > 0.5
    except (FileNotFoundError, ValueError, ZeroDivisionError):
        return False

def run_ffmpeg_with_progress(command, total_duration, desc="Processing"):
    """Runs an FFmpeg command and shows a tqdm progress bar based on stderr output.
    Returns (returncode, stderr_text) so callers can print errors."""
    from tqdm import tqdm
    import re
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        universal_newlines=True
    )
    pbar = tqdm(total=int(total_duration), desc=desc, unit="s", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]')
    time_pattern = re.compile(r'time=(\d+):(\d+):(\d+)\.(\d+)')
    last_seconds = 0
    stderr_lines = []
    for line in process.stderr:
        stderr_lines.append(line)
        match = time_pattern.search(line)
        if match:
            h, m, s, _ = match.groups()
            current_seconds = int(h) * 3600 + int(m) * 60 + int(s)
            if current_seconds > last_seconds:
                pbar.update(current_seconds - last_seconds)
                last_seconds = current_seconds
    pbar.update(max(0, int(total_duration) - last_seconds))
    pbar.close()
    process.wait()
    return process.returncode, ''.join(stderr_lines)

def normalize_to_cfr(video_path, output_path, total_duration=0):
    """Re-muxes a VFR video to constant frame rate."""
    print("  Normalizing variable frame rate to constant frame rate...")
    command = [
        'ffmpeg', '-y', '-i', video_path,
        '-vsync', 'cfr', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'copy', output_path
    ]
    if total_duration > 0:
        returncode, stderr_text = run_ffmpeg_with_progress(command, total_duration, desc="VFR → CFR")
    else:
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Warning: VFR normalization failed, proceeding with original file.")
            print("  Stderr:", e.stderr.decode())
            return False
    if returncode != 0:
        print(f"  Warning: VFR normalization failed, proceeding with original file.")
        return False
    return True

def detect_hw_encoder():
    """Probes FFmpeg for available hardware H.264 encoders.

    Returns (encoder_name, encoder_type) where encoder_type is one of
    'videotoolbox', 'nvenc', or 'libx264'.
    """
    candidates = [
        ('h264_videotoolbox', 'videotoolbox'),
        ('h264_nvenc',        'nvenc'),
    ]
    for encoder, etype in candidates:
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True, text=True
            )
            if encoder in result.stdout:
                return encoder, etype
        except FileNotFoundError:
            break
    return 'libx264', 'libx264'

def resolve_encoder(requested, hw_encoder_name, hw_encoder_type):
    """Resolves which encoder to use based on user request.

    requested: 'auto' (default, always libx264 for quality), 'hw' (use hardware
               if available), or a specific encoder name like 'h264_videotoolbox'.
    Returns (encoder_name, encoder_type).
    """
    if requested == 'auto':
        return 'libx264', 'libx264'
    elif requested == 'hw':
        return hw_encoder_name, hw_encoder_type
    else:
        # User specified an explicit encoder
        if requested == hw_encoder_name:
            return hw_encoder_name, hw_encoder_type
        return requested, requested

def build_encoder_args(encoder_type, quality_level, crf_override=None, preset_override=None):
    """Returns a list of FFmpeg encoder arguments for the given encoder and quality.

    quality_level is one of 'fast', 'balanced', 'high'.
    crf_override and preset_override allow user to force specific values (libx264 only).
    """
    presets = {
        'libx264': {
            'fast':     ['-crf', '28', '-preset', 'veryfast'],
            'balanced': ['-crf', '23', '-preset', 'fast'],
            'high':     ['-crf', '18', '-preset', 'slow'],
        },
        'videotoolbox': {
            'fast':     ['-b:v', '3M', '-allow_sw', '1', '-realtime', '0'],
            'balanced': ['-b:v', '6M', '-allow_sw', '1', '-realtime', '0'],
            'high':     ['-b:v', '12M', '-allow_sw', '1', '-realtime', '0'],
        },
        'nvenc': {
            'fast':     ['-cq', '28', '-preset', 'p1'],
            'balanced': ['-cq', '23', '-preset', 'p4'],
            'high':     ['-cq', '18', '-preset', 'p7'],
        },
    }

    args = list(presets[encoder_type][quality_level])

    # Allow user overrides for libx264
    if encoder_type == 'libx264':
        if crf_override is not None:
            args[args.index('-crf') + 1] = str(crf_override)
        if preset_override is not None:
            args[args.index('-preset') + 1] = preset_override

    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Smartly crops a horizontal video into a vertical one.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output video file.")
    parser.add_argument('--ratio', type=str, default='9:16',
                        help="Output aspect ratio as W:H (default: 9:16). Examples: 9:16, 4:5, 1:1")
    parser.add_argument('--quality', type=str, default='balanced', choices=['fast', 'balanced', 'high'],
                        help="Encoding quality preset (default: balanced). fast=quick encode, balanced=good quality, high=best quality/slow")
    parser.add_argument('--crf', type=int, default=None,
                        help="Override CRF value directly (0-51, lower=better quality). Overrides --quality.")
    parser.add_argument('--preset', type=str, default=None,
                        help="Override FFmpeg x264 preset directly (ultrafast..veryslow). Overrides --quality.")
    parser.add_argument('--plan-only', action='store_true',
                        help="Only run scene detection and analysis (Steps 1-3), then print the processing plan without encoding.")
    parser.add_argument('--frame-skip', type=int, default=0,
                        help="Frames to skip during scene detection (default: 0 = every frame, most accurate). "
                             "1 = every other frame (~2x faster). Higher = faster but may miss quick cuts.")
    parser.add_argument('--downscale', type=int, default=0,
                        help="Downscale factor for scene detection (default: 0 = auto). "
                             "Higher values (2-4) are faster but may miss subtle scene changes.")
    parser.add_argument('--scene-detector', type=str, default='content',
                        choices=['content', 'adaptive'],
                        help="Scene-detection algorithm. 'content' (default) uses a fixed "
                             "threshold; 'adaptive' adapts to camera motion / gradual lighting "
                             "changes and usually catches more cuts in dynamic footage.")
    parser.add_argument('--scene-threshold', type=float, default=None,
                        help="Cut sensitivity. For --scene-detector content, default is 27 "
                             "(PySceneDetect default); LOWER values (e.g. 15-20) detect more "
                             "subtle cuts (dialogue, talk shows, similar lighting). "
                             "For --scene-detector adaptive, default is 3.0.")
    parser.add_argument('--min-scene-len', type=int, default=15,
                        help="Minimum scene length in frames (default 15). Lower this to "
                             "catch very fast cuts (montages, action edits).")
    parser.add_argument('--analysis-samples', type=int, default=3,
                        help="Number of frames to sample per scene for person detection "
                             "(default 3). Higher = more reliable people detection at the "
                             "cost of more YOLO inference. 1 = legacy middle-frame-only.")
    parser.add_argument('--motion-threshold', type=float, default=0.5,
                        help="In multi-person scenes, focus on the most active subject "
                             "if their motion score (body-widths travelled across sampled "
                             "frames) is at least this value. Default 0.5. Lower (e.g. 0.2) "
                             "= more aggressive single-subject tracking; very high (e.g. 999) "
                             "= legacy behavior (always group bounding box). Requires "
                             "--analysis-samples >= 2 to have any effect.")
    parser.add_argument('--encoder', type=str, default='auto',
                        help="Video encoder: 'auto' (libx264, default), 'hw' (auto-detect hardware encoder), "
                             "or a specific encoder name like 'h264_videotoolbox' or 'h264_nvenc'.")
    args = parser.parse_args()

    # Parse aspect ratio
    try:
        ratio_parts = args.ratio.split(':')
        ASPECT_RATIO = int(ratio_parts[0]) / int(ratio_parts[1])
    except (ValueError, IndexError, ZeroDivisionError):
        print(f"❌ Invalid aspect ratio '{args.ratio}'. Use format W:H (e.g. 9:16, 4:5, 1:1)")
        sys.exit(1)

    # Resolve encoder: default is libx264 for best quality; --encoder hw for hardware
    hw_encoder_name, hw_encoder_type = detect_hw_encoder()
    encoder_name, encoder_type = resolve_encoder(args.encoder, hw_encoder_name, hw_encoder_type)
    enc_args = build_encoder_args(encoder_type, args.quality,
                                  crf_override=args.crf, preset_override=args.preset)

    # Defer heavy imports until after arg parsing so --help is instant
    import cv2
    import numpy as np
    from tqdm import tqdm

    script_start_time = time.time()

    input_video = args.input
    final_output_video = args.output

    # Ensure the output filename has a video extension so FFmpeg can determine the format
    _, ext = os.path.splitext(final_output_video)
    if not ext:
        final_output_video += '.mp4'
    
    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.mkv"
    temp_cfr_input = f"{base_name}_temp_cfr_input.mp4"
    
    def cleanup_temp_files():
        """Remove any leftover temporary files."""
        for f in [temp_video_output, temp_audio_output, temp_cfr_input]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    # Clean up previous temp files if they exist
    cleanup_temp_files()
    if os.path.exists(final_output_video): os.remove(final_output_video)

    # Print input file summary
    media_info = get_media_info(input_video)
    if media_info:
        print(f"\n📄 Input: {os.path.basename(args.input)}")
        parts = []
        if 'width' in media_info:
            parts.append(f"{media_info['width']}x{media_info['height']}")
        if 'fps' in media_info:
            parts.append(f"{media_info['fps']} fps")
        if 'video_codec' in media_info:
            parts.append(media_info['video_codec'])
        if 'audio_codec' in media_info:
            parts.append(media_info['audio_codec'])
        if 'duration' in media_info:
            parts.append(format_duration(media_info['duration']))
        if 'size_bytes' in media_info:
            parts.append(format_file_size(media_info['size_bytes']))
        print(f"   {' | '.join(parts)}")
        total_frames_est = int(media_info.get('duration', 0) * media_info.get('fps', 0))
        if total_frames_est > 0:
            print(f"   ~{total_frames_est:,} frames to process")
    enc_label = f"{encoder_name} ({' '.join(enc_args)})"
    print(f"   Ratio: {args.ratio} | Quality: {args.quality} | Encoder: {enc_label}")
    print()

    # Pre-processing: normalize VFR to CFR if needed
    if is_variable_frame_rate(input_video):
        print("⚠️  Variable frame rate detected — normalizing to constant frame rate first...")
        duration = media_info.get('duration', 0) if media_info else 0
        if normalize_to_cfr(input_video, temp_cfr_input, total_duration=duration):
            input_video = temp_cfr_input
            print("✅ VFR normalization complete.")
        else:
            print("⚠️  Proceeding with original VFR file (audio sync may be affected).")

    print("🎬 Step 1: Detecting scenes...")
    step_start_time = time.time()
    scenes, _ = detect_scenes(input_video, downscale=args.downscale, frame_skip=args.frame_skip)
    step_end_time = time.time()
    
    if not scenes:
        print("❌ No scenes were detected. Aborting.")
        sys.exit(1)
    
    print(f"✅ Found {len(scenes)} scenes in {step_end_time - step_start_time:.2f}s. Here is the breakdown:")
    for i, (start, end) in enumerate(scenes):
        print(f"  - Scene {i+1}: {start.get_timecode()} -> {end.get_timecode()}")


    print("\n🧠 Step 2: Analyzing scene content and determining strategy...")
    step_start_time = time.time()
    # Get fps from OpenCV — the same backend that reads the frames — to avoid
    # frame-rate mismatches between the reader and encoder that cause audio drift.
    original_width, original_height, fps = get_video_properties(input_video)
    
    OUTPUT_HEIGHT = original_height
    if OUTPUT_HEIGHT % 2 != 0:
        OUTPUT_HEIGHT += 1
    OUTPUT_WIDTH = int(OUTPUT_HEIGHT * ASPECT_RATIO)
    if OUTPUT_WIDTH % 2 != 0:
        OUTPUT_WIDTH += 1

    scenes_analysis = []
    for i, (start_time, end_time) in enumerate(tqdm(scenes, desc="Analyzing Scenes")):
        analysis = analyze_scene_content(input_video, start_time, end_time,
                                         samples=args.analysis_samples)
        strategy, target_box = decide_cropping_strategy(
            analysis, original_height, motion_threshold=args.motion_threshold)
        scenes_analysis.append({
            'start_frame': start_time.get_frames(),
            'end_frame': end_time.get_frames(),
            'start_seconds': start_time.get_seconds(),
            'end_seconds': end_time.get_seconds(),
            'analysis': analysis,
            'strategy': strategy,
            'target_box': target_box
        })
    step_end_time = time.time()
    print(f"✅ Scene analysis complete in {step_end_time - step_start_time:.2f}s.")

    print("\n📋 Step 3: Generated Processing Plan")
    for i, scene_data in enumerate(scenes_analysis):
        num_people = len(scene_data['analysis'])
        strategy = scene_data['strategy']
        start_time = scenes[i][0].get_timecode()
        end_time = scenes[i][1].get_timecode()
        motions = [obj.get('motion', 0.0) for obj in scene_data['analysis']]
        motion_str = ""
        if motions:
            top = max(motions)
            motion_str = f", max motion: {top:.2f}"
            if num_people > 1 and top >= args.motion_threshold:
                motion_str += " ⚡ (focused on most active)"
        print(f"  - Scene {i+1} ({start_time} -> {end_time}): "
              f"Found {num_people} person(s){motion_str}. Strategy: {strategy}")

    if args.plan_only:
        track_count = sum(1 for s in scenes_analysis if s['strategy'] == 'TRACK')
        letterbox_count = sum(1 for s in scenes_analysis if s['strategy'] == 'LETTERBOX')
        elapsed = time.time() - script_start_time
        print(f"\n📊 Plan summary: {track_count} TRACK / {letterbox_count} LETTERBOX scenes")
        print(f"⏱️  Analysis took {elapsed:.1f}s. Run without --plan-only to encode.")
        sys.exit(0)

    print("\n✂️ Step 4: Processing video frames...")
    step_start_time = time.time()

    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-',
        '-c:v', encoder_name, *enc_args,
        '-pix_fmt', 'yuv420p',
        '-r', str(fps), '-vsync', 'cfr',
        '-an', temp_video_output
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0
    current_scene_index = 0
    dropped_frames = 0
    last_output_frame = None

    num_scenes = len(scenes_analysis)
    with tqdm(total=total_frames, desc=f"Processing [scene 1/{num_scenes}]",
              unit="fr", dynamic_ncols=True,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_scene_index < len(scenes_analysis) - 1 and \
               frame_number >= scenes_analysis[current_scene_index + 1]['start_frame']:
                current_scene_index += 1
                pbar.set_description(f"Processing [scene {current_scene_index + 1}/{num_scenes}]")

            scene_data = scenes_analysis[current_scene_index]
            strategy = scene_data['strategy']
            target_box = scene_data['target_box']

            try:
                if strategy == 'TRACK':
                    crop_box = calculate_crop_box(target_box, original_width, original_height)
                    processed_frame = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                    output_frame = cv2.resize(processed_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                else:  # LETTERBOX
                    scale_factor = OUTPUT_WIDTH / original_width
                    scaled_height = int(original_height * scale_factor)
                    scaled_frame = cv2.resize(frame, (OUTPUT_WIDTH, scaled_height))

                    output_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
                    y_offset = (OUTPUT_HEIGHT - scaled_height) // 2
                    output_frame[y_offset:y_offset + scaled_height, :] = scaled_frame
                last_output_frame = output_frame
            except Exception:
                dropped_frames += 1
                if last_output_frame is not None:
                    output_frame = last_output_frame
                else:
                    output_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)

            ffmpeg_process.stdin.write(output_frame.tobytes())
            frame_number += 1
            pbar.update(1)

    if dropped_frames > 0:
        print(f"  ⚠️  {dropped_frames} frame(s) could not be processed and were duplicated from the previous frame.")

    ffmpeg_process.stdin.close()
    stderr_output = ffmpeg_process.stderr.read().decode()
    ffmpeg_process.wait()
    cap.release()

    if ffmpeg_process.returncode != 0:
        print("\n❌ FFmpeg frame processing failed.")
        print("Stderr:", stderr_output)
        cleanup_temp_files()
        sys.exit(1)
    step_end_time = time.time()
    print(f"✅ Video processing complete in {step_end_time - step_start_time:.2f}s.")

    input_has_audio = has_audio_stream(input_video)

    if input_has_audio:
        print("\n🔊 Step 5: Extracting original audio...")
        step_start_time = time.time()

        # Some files have a non-zero video start_time (e.g. audio starts at 0s
        # but video starts at 1.8s). OpenCV ignores this offset and reads frames
        # from the first video frame, so the processed video starts at 0s.
        # We must trim the audio to match: skip audio before the video started,
        # and limit to the video's duration.
        video_start = get_stream_start_time(input_video, 'v:0')
        audio_extract_command = [
            'ffmpeg', '-y', '-ss', str(video_start),
            '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
        ]
        try:
            subprocess.run(audio_extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            step_end_time = time.time()
            print(f"✅ Audio extracted in {step_end_time - step_start_time:.2f}s.")
        except subprocess.CalledProcessError as e:
            print("\n❌ Audio extraction failed.")
            print("Stderr:", e.stderr.decode())
            cleanup_temp_files()
            sys.exit(1)

        print("\n✨ Step 6: Merging video and audio...")
        step_start_time = time.time()
        merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output, '-i', temp_audio_output,
            '-c:v', 'copy', '-c:a', 'copy', '-shortest', final_output_video
        ]
        try:
            subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            step_end_time = time.time()
            print(f"✅ Final video merged in {step_end_time - step_start_time:.2f}s.")
        except subprocess.CalledProcessError as e:
            print("\n❌ Final merge failed.")
            print("Stderr:", e.stderr.decode())
            cleanup_temp_files()
            sys.exit(1)

        cleanup_temp_files()
    else:
        print("\n🔇 Step 5: No audio stream detected, skipping audio extraction.")
        # Just rename the temp video as the final output
        os.rename(temp_video_output, final_output_video)
        cleanup_temp_files()

    script_end_time = time.time()
    total_time = script_end_time - script_start_time

    # Final summary
    print(f"\n{'─' * 50}")
    print(f"🎉 All done! Final video saved to {final_output_video}")
    print(f"{'─' * 50}")
    output_info = get_media_info(final_output_video)
    if output_info:
        out_parts = []
        if 'width' in output_info:
            out_parts.append(f"{output_info['width']}x{output_info['height']}")
        if 'duration' in output_info:
            out_parts.append(format_duration(output_info['duration']))
        if 'size_bytes' in output_info:
            out_parts.append(format_file_size(output_info['size_bytes']))
        print(f"   Output: {' | '.join(out_parts)}")
    if media_info and output_info and media_info.get('size_bytes') and output_info.get('size_bytes'):
        ratio = output_info['size_bytes'] / media_info['size_bytes'] * 100
        print(f"   Size:   {format_file_size(media_info['size_bytes'])} → {format_file_size(output_info['size_bytes'])} ({ratio:.0f}% of original)")
    print(f"   Time:   {format_duration(total_time)} ({total_time:.1f}s)")
    if media_info and media_info.get('duration'):
        speed = media_info['duration'] / total_time if total_time > 0 else 0
        print(f"   Speed:  {speed:.1f}x real-time")
