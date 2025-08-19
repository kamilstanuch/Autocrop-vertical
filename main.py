import cv2
import scenedetect
import subprocess
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
import os
import numpy as np

# --- Constants ---
ASPECT_RATIO = 9 / 16
OUTPUT_WIDTH = 360
OUTPUT_HEIGHT = 640

# Load the YOLO model once
model = YOLO('yolov8n.pt')

# Load the Haar Cascade for face detection once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_scene_content(video_path, scene_start_time, scene_end_time):
    """
    Analyzes the middle frame of a scene to detect people and faces.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the middle frame number of the scene
    start_frame = scene_start_time.get_frames()
    end_frame = scene_end_time.get_frames()
    middle_frame_number = int(start_frame + (end_frame - start_frame) / 2)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []

    # --- Person Detection using YOLO ---
    results = model([frame], verbose=False)
    
    detected_objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls[0] == 0:
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                person_box = [x1, y1, x2, y2]
                
                # --- Face Detection within the person box ---
                person_roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(person_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                face_box = None
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    face_box = [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh]

                detected_objects.append({'person_box': person_box, 'face_box': face_box})
                
    cap.release()
    return detected_objects


def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
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

def decide_cropping_strategy(scene_analysis, frame_height):
    num_people = len(scene_analysis)
    if num_people == 0:
        return 'LETTERBOX', None
    if num_people == 1:
        target_box = scene_analysis[0]['face_box'] or scene_analysis[0]['person_box']
        return 'TRACK', target_box
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

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

if __name__ == '__main__':
    input_video = 'video_churchil.mp4'
    temp_video_output = 'temp_video.mp4'
    temp_audio_output = 'temp_audio.aac'
    final_output_video = 'output.mp4'
    
    # Clean up previous temp files if they exist
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(temp_audio_output): os.remove(temp_audio_output)
    if os.path.exists(final_output_video): os.remove(final_output_video)

    print("Step 1: Detecting scenes...")
    scenes, fps = detect_scenes(input_video)
    
    if not scenes:
        print("No scenes were detected. Aborting.")
        exit()

    print("\nStep 2: Analyzing content of each scene...")
    original_width, original_height = get_video_resolution(input_video)
    scenes_analysis = []
    for i, (start_time, end_time) in enumerate(scenes):
        analysis = analyze_scene_content(input_video, start_time, end_time)
        scenes_analysis.append({
            'start_frame': start_time.get_frames(),
            'end_frame': end_time.get_frames(),
            'analysis': analysis
        })
        print(f"  - Scene {i+1}: Found {len(analysis)} person(s).")

    print("\nStep 3: Processing video frames and piping to FFmpeg...")
    
    # FFmpeg command to receive frames from stdin
    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-an', # No audio in this temporary file
        temp_video_output
    ]

    # Open FFmpeg process
    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    frame_number = 0
    current_scene_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % 100 == 0:
            print(f"  - Processing frame {frame_number}/{total_frames}")

        # Determine which scene this frame is in
        if current_scene_index < len(scenes_analysis) - 1 and \
           frame_number >= scenes_analysis[current_scene_index + 1]['start_frame']:
            current_scene_index += 1

        scene_data = scenes_analysis[current_scene_index]
        strategy, target_box = decide_cropping_strategy(scene_data['analysis'], original_height)

        if strategy == 'TRACK':
            crop_box = calculate_crop_box(target_box, original_width, original_height)
            processed_frame = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            output_frame = cv2.resize(processed_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        else: # LETTERBOX
            scale_factor = OUTPUT_WIDTH / original_width
            scaled_height = int(original_height * scale_factor)
            scaled_frame = cv2.resize(frame, (OUTPUT_WIDTH, scaled_height))
            
            output_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
            y_offset = (OUTPUT_HEIGHT - scaled_height) // 2
            output_frame[y_offset:y_offset + scaled_height, :] = scaled_frame
        
        # Write frame to FFmpeg's stdin
        ffmpeg_process.stdin.write(output_frame.tobytes())
        frame_number += 1
    
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    cap.release()
    
    print("\nStep 4: Extracting original audio...")
    audio_extract_command = [
        'ffmpeg', '-y', '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
    ]
    subprocess.run(audio_extract_command, check=True, capture_output=True)

    print("\nStep 5: Merging processed video and original audio...")
    merge_command = [
        'ffmpeg', '-y', '-i', temp_video_output, '-i', temp_audio_output,
        '-c:v', 'copy', '-c:a', 'copy', final_output_video
    ]
    subprocess.run(merge_command, check=True, capture_output=True)

    # Clean up temp files
    os.remove(temp_video_output)
    os.remove(temp_audio_output)

    print(f"\nAll done! Final video saved to {final_output_video}")
