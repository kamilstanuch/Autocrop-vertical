import cv2
import scenedetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
from moviepy.editor import VideoFileClip, AudioFileClip

# --- Constants ---
ASPECT_RATIO = 9 / 16
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

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
    # The model expects a list of images
    results = model([frame], verbose=False)
    
    detected_objects = []

    # Process results
    for result in results:
        # We are interested in the bounding boxes
        boxes = result.boxes
        for box in boxes:
            # class '0' is 'person' in the COCO dataset which yolov8n was trained on
            if box.cls[0] == 0:
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                person_box = [x1, y1, x2, y2]
                
                # --- Face Detection within the person box ---
                person_roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(person_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                face_box = None
                if len(faces) > 0:
                    # We only care about the first (and likely largest) face found
                    fx, fy, fw, fh = faces[0]
                    # The face coordinates are relative to the person crop, so we convert them back to frame coordinates
                    face_box = [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh]

                detected_objects.append({'person_box': person_box, 'face_box': face_box})
                
    cap.release()
    return detected_objects


def detect_scenes(video_path):
    # Create a video manager and add the video to it
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()

    # Add the ContentDetector algorithm to the scene_manager
    scene_manager.add_detector(ContentDetector())

    # Base video manager is required to accurately split scenes
    video_manager.set_downscale_factor()
    video_manager.start()

    # Perform scene detection
    scene_manager.detect_scenes(frame_source=video_manager)

    # Get the list of scenes
    scene_list = scene_manager.get_scene_list()

    # We also need the framerate for frame calculations
    fps = video_manager.get_framerate()

    print("Detected scenes:")
    if not scene_list:
        print("No scenes detected.")
    else:
        for i, scene in enumerate(scene_list):
            print(
                f"Scene {i+1}: "
                f"Start {scene[0].get_timecode()} / "
                f"End {scene[1].get_timecode()}"
            )
    
    video_manager.release()
    return scene_list, fps

def get_enclosing_box(boxes):
    """
    Calculates a single bounding box that encloses a list of boxes.
    """
    if not boxes:
        return None
    
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    
    return [min_x, min_y, max_x, max_y]

def decide_cropping_strategy(scene_analysis, frame_height):
    """
    Decides the cropping strategy based on the scene analysis.
    Returns the strategy ('TRACK', 'LETTERBOX') and the target box.
    """
    num_people = len(scene_analysis)
    
    if num_people == 0:
        return 'LETTERBOX', None
        
    if num_people == 1:
        # If there's a face, track the face. Otherwise, track the person.
        target_box = scene_analysis[0]['face_box'] or scene_analysis[0]['person_box']
        return 'TRACK', target_box

    # More than one person
    person_boxes = [obj['person_box'] for obj in scene_analysis]
    group_box = get_enclosing_box(person_boxes)
    
    group_width = group_box[2] - group_box[0]
    
    # Heuristic: If the group width is less than ~80% of the vertical frame's
    # width (which is frame_height * 9/16), we can probably fit them.
    max_width_for_crop = (frame_height * ASPECT_RATIO) * 0.8
    
    if group_width < max_width_for_crop:
        return 'TRACK', group_box
    else:
        return 'LETTERBOX', None


def calculate_crop_box(target_box, frame_width, frame_height):
    """
    Calculates a 9:16 crop box centered on the target_box.
    """
    target_center_x = (target_box[0] + target_box[2]) / 2
    
    crop_height = frame_height
    crop_width = int(crop_height * ASPECT_RATIO)
    
    # Calculate crop box coordinates
    x1 = int(target_center_x - crop_width / 2)
    y1 = 0
    x2 = int(target_center_x + crop_width / 2)
    y2 = frame_height
    
    # Ensure the crop box is within the frame bounds
    if x1 < 0:
        x1 = 0
        x2 = crop_width
    if x2 > frame_width:
        x2 = frame_width
        x1 = frame_width - crop_width
        
    return x1, y1, x2, y2

def process_video(input_path, output_path, scenes_analysis):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    
    frame_number = 0
    current_scene_index = 0
    
    # Get the first scene's strategy
    strategy, target_box = decide_cropping_strategy(scenes_analysis[0]['analysis'], frame_height)
    if strategy == 'TRACK':
        crop_box = calculate_crop_box(target_box, frame_width, frame_height)

    print("Processing video with smart cropping...")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we have moved to a new scene
        if current_scene_index < len(scenes_analysis) - 1:
            next_scene_start_frame = scenes_analysis[current_scene_index + 1]['start_frame']
            if frame_number >= next_scene_start_frame:
                current_scene_index += 1
                strategy, target_box = decide_cropping_strategy(scenes_analysis[current_scene_index]['analysis'], frame_height)
                print(f"  - New Scene ({current_scene_index+1}): Strategy is {strategy}")
                if strategy == 'TRACK':
                    crop_box = calculate_crop_box(target_box, frame_width, frame_height)

        # Apply the transformation for the current scene
        if strategy == 'TRACK':
            cropped_frame = frame[:, crop_box[0]:crop_box[2]]
            output_frame = cv2.resize(cropped_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        else: # LETTERBOX
            # Calculate letterbox dimensions
            scale_factor = OUTPUT_WIDTH / frame_width
            scaled_height = int(frame_height * scale_factor)
            
            scaled_frame = cv2.resize(frame, (OUTPUT_WIDTH, scaled_height))
            
            # Create a black canvas
            output_frame =  torch.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=torch.uint8).numpy()
            
            # Paste the scaled frame in the center
            y_offset = (OUTPUT_HEIGHT - scaled_height) // 2
            output_frame[y_offset:y_offset + scaled_height, :] = scaled_frame
        
        out.write(output_frame)
        frame_number += 1
        
    cap.release()
    out.release()
    print("Video processing complete.")


if __name__ == '__main__':
    input_video = 'video_churchil.mp4'
    temp_output_video = 'temp_output.mp4'
    final_output_video = 'output.mp4'
    
    print("Step 1: Detecting scenes...")
    scenes, fps = detect_scenes(input_video)
    
    scenes_analysis = []
    if scenes:
        print("\nStep 2: Analyzing content of each scene...")
        for i, (start_time, end_time) in enumerate(scenes):
            analysis = analyze_scene_content(input_video, start_time, end_time)
            scenes_analysis.append({
                'scene': i+1,
                'start_frame': start_time.get_frames(),
                'end_frame': end_time.get_frames(),
                'analysis': analysis
            })
            print(f"  - Scene {i+1}: Found {len(analysis)} person(s).")
            
        print("\nStep 3: Processing video and applying smart cropping...")
        process_video(input_video, temp_output_video, scenes_analysis)
        
        print("\nStep 4: Merging audio...")
        original_video_clip = VideoFileClip(input_video)
        processed_video_clip = VideoFileClip(temp_output_video)
        
        final_clip = processed_video_clip.set_audio(original_video_clip.audio)
        final_clip.write_videofile(final_output_video, codec='libx264', audio_codec='aac')
        
        print(f"\nAll done! Final video saved to {final_output_video}")
    else:
        print("No scenes were detected. Aborting.")
