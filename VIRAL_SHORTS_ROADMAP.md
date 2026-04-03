# 🚀 Viral Shorts Implementation Roadmap

This document outlines the step-by-step implementation plan to transform our AI Video Clipping Bot into a comprehensive viral shorts generator, based on the analysis of [mutonby/openshorts](https://github.com/mutonby/openshorts).

---

## 📊 Current State vs Target State

### What We Have ✅
- Horizontal → Vertical (9:16) conversion
- AI face tracking (YOLOv8 + MediaPipe)
- Smart cropping strategies (TRACK vs LETTERBOX)
- Scene detection
- Audio preservation

### What We're Adding 🎯
- AI viral moment detection (Gemini)
- Auto-generated subtitles (faster-whisper)
- Hook text overlays
- AI video effects
- Multi-platform metadata
- Batch clip processing

---

## 🏗️ Implementation Phases

## Phase 1: Core Infrastructure (Week 1-2)

### Stage 1.1: Add Transcript Generation
**Estimated Time:** 2 days  
**Priority:** HIGH

#### Tasks:
- [ ] Install `faster-whisper` package
- [ ] Create `transcribe.py` module
- [ ] Implement word-level timestamp extraction
- [ ] Test transcription accuracy on sample videos

#### Implementation Details:
```python
# transcribe.py
from faster_whisper import WhisperModel

def transcribe_video(video_path):
    """
    Transcribe audio with word-level timestamps.
    Returns: {
        'text': 'full transcript',
        'segments': [...],
        'language': 'en'
    }
    """
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(video_path, word_timestamps=True)
    # Process and return structured data
```

#### Files to Create:
- `transcribe.py`

#### Files to Modify:
- `requirements.txt` - add `faster-whisper>=1.0.0`

#### Success Criteria:
- Successfully transcribe 5-minute video in < 2 minutes
- Word-level timestamps accurate within 0.1s
- Handles multiple languages

---

### Stage 1.2: Integrate Gemini API
**Estimated Time:** 2 days  
**Priority:** HIGH

#### Tasks:
- [ ] Set up Google Gemini API credentials
- [ ] Create `.env` file for API key storage
- [ ] Create `viral_detector.py` module
- [ ] Implement prompt template for viral moment detection
- [ ] Parse and validate JSON responses

#### Implementation Details:
```python
# viral_detector.py
from google import genai
import json

VIRAL_PROMPT_TEMPLATE = """
You are a senior short-form video editor. Analyze this transcript 
and identify 3-15 MOST VIRAL moments for TikTok/Reels/Shorts.

Requirements:
- Each clip: 15-60 seconds
- Strong hooks (first 3 seconds critical)
- Natural cuts at silence/pauses
- Include viral_hook_text (max 10 words)
- Platform-specific titles/descriptions

Video Duration: {duration}s
Transcript: {transcript}
Words (with timestamps): {words}

Return JSON only:
{
  "shorts": [
    {
      "start": 12.340,
      "end": 37.900,
      "viral_hook_text": "Did you know this?",
      "video_title_for_youtube_short": "...",
      "video_description_for_tiktok": "...",
      "video_description_for_instagram": "..."
    }
  ]
}
"""

def detect_viral_clips(transcript, video_duration):
    # Implement Gemini API call
    pass
```

#### Files to Create:
- `viral_detector.py`
- `.env.example`
- `.env` (gitignored)

#### Files to Modify:
- `.gitignore` - add `.env`
- `requirements.txt` - add `google-genai>=0.7.0`, `python-dotenv>=1.0.0`

#### Success Criteria:
- Successfully detect 3-15 clips from 30-min video
- Each clip has valid timestamps and metadata
- API calls complete in < 30 seconds

---

## Phase 2: Viral Clip Detection (Week 2-3)

### Stage 2.1: Implement Clip Extraction
**Estimated Time:** 2 days  
**Priority:** HIGH

#### Tasks:
- [ ] Create `clip_extractor.py` module
- [ ] Implement precise FFmpeg clip cutting
- [ ] Maintain audio sync during extraction
- [ ] Handle edge cases (clip boundaries, audio offset)

#### Implementation Details:
```python
# clip_extractor.py
import subprocess

def extract_clip(input_video, start, end, output_path):
    """
    Extract precise clip using FFmpeg re-encoding.
    Ensures frame-accurate cuts and audio sync.
    """
    command = [
        'ffmpeg', '-y',
        '-ss', str(start),
        '-to', str(end),
        '-i', input_video,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-c:a', 'aac',
        output_path
    ]
    subprocess.run(command, check=True, 
                   stdout=subprocess.DEVNULL, 
                   stderr=subprocess.PIPE)
```

#### Files to Create:
- `clip_extractor.py`

#### Success Criteria:
- Clips extracted within ±0.1s of target timestamps
- No audio desync
- Process 10 clips from 1-hour video in < 5 minutes

---

### Stage 2.2: Create Metadata Generator
**Estimated Time:** 1 day  
**Priority:** MEDIUM

#### Tasks:
- [ ] Create `metadata_generator.py` module
- [ ] Generate comprehensive JSON metadata per clip
- [ ] Include platform-specific data
- [ ] Save transcript segments with each clip

#### Implementation Details:
```python
# metadata_generator.py
import json

def generate_clip_metadata(clip_info, transcript_segment, effects_applied):
    """
    Generate metadata JSON for a single clip.
    """
    metadata = {
        "clip_number": clip_info['index'],
        "timestamps": {
            "start": clip_info['start'],
            "end": clip_info['end'],
            "duration": clip_info['end'] - clip_info['start']
        },
        "viral_hook_text": clip_info['viral_hook_text'],
        "platforms": {
            "tiktok": {
                "description": clip_info['video_description_for_tiktok'],
                "hashtags": extract_hashtags(clip_info['video_description_for_tiktok'])
            },
            "instagram": {
                "description": clip_info['video_description_for_instagram']
            },
            "youtube": {
                "title": clip_info['video_title_for_youtube_short']
            }
        },
        "transcript_segment": transcript_segment,
        "effects_applied": effects_applied,
        "created_at": datetime.now().isoformat()
    }
    return metadata
```

#### Files to Create:
- `metadata_generator.py`

#### Success Criteria:
- Valid JSON for each clip
- All platform data included
- Metadata file < 100KB per clip

---

## Phase 3: Subtitle Generation (Week 3-4)

### Stage 3.1: Create SRT Generator
**Estimated Time:** 2 days  
**Priority:** HIGH (Subtitles = 40% engagement boost)

#### Tasks:
- [ ] Create `subtitle_generator.py` module
- [ ] Implement word grouping algorithm (max 20 chars, 2s duration)
- [ ] Handle clip time offsets correctly
- [ ] Support manual line breaks
- [ ] Format SRT with proper timecodes

#### Implementation Details:
```python
# subtitle_generator.py

def generate_srt(transcript, clip_start, clip_end, output_path, 
                 max_chars=20, max_duration=2.0):
    """
    Generate SRT file with short, readable subtitle blocks.
    Optimized for vertical video (TikTok/Reels style).
    """
    # Extract words within clip timeframe
    words = extract_words_in_range(transcript, clip_start, clip_end)
    
    # Group words intelligently
    subtitle_blocks = []
    current_block = []
    block_start = None
    
    for word in words:
        # Adjust times relative to clip start
        start = max(0, word['start'] - clip_start)
        end = max(0, word['end'] - clip_start)
        
        if should_close_block(current_block, word, max_chars, max_duration):
            # Finalize current block
            subtitle_blocks.append(format_block(current_block))
            current_block = [word]
        else:
            current_block.append(word)
    
    # Write SRT file
    write_srt_file(subtitle_blocks, output_path)
```

#### Files to Create:
- `subtitle_generator.py`

#### Success Criteria:
- Subtitles readable on mobile (max 20 chars per line)
- No word cutoffs mid-sentence
- Timing accurate within 0.1s

---

### Stage 3.2: Burn Subtitles with FFmpeg
**Estimated Time:** 2 days  
**Priority:** HIGH

#### Tasks:
- [ ] Implement subtitle burning with ASS styling
- [ ] Create TikTok-style preset (bold, white text, black outline)
- [ ] Support multiple positions (top/center/bottom)
- [ ] Add outline mode vs box mode
- [ ] Make font/color customizable

#### Implementation Details:
```python
# subtitle_generator.py (continued)

def burn_subtitles(video_path, srt_path, output_path,
                   alignment='bottom', fontsize=16,
                   font_name='Verdana', font_color='#FFFFFF',
                   border_color='#000000', border_width=2,
                   bg_color='#000000', bg_opacity=0.0):
    """
    Burn subtitles with customizable styling.
    
    Modes:
    - Outline mode (bg_opacity=0): White text + black border
    - Box mode (bg_opacity>0): Text with semi-transparent background
    """
    # Map alignment
    ass_alignment = {
        'top': 6,
        'middle': 10,
        'bottom': 2
    }[alignment]
    
    # Build ASS style string
    style = build_ass_style(
        alignment=ass_alignment,
        fontsize=fontsize,
        font_name=font_name,
        primary_color=hex_to_ass_color(font_color),
        outline_color=hex_to_ass_color(border_color),
        border_width=border_width,
        bg_opacity=bg_opacity
    )
    
    # FFmpeg command
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f"subtitles='{srt_path}':force_style='{style}'",
        '-c:a', 'copy',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        output_path
    ]
    
    subprocess.run(cmd, check=True)
```

#### Files to Modify:
- `subtitle_generator.py` - add burning function

#### Success Criteria:
- Subtitles visible on all devices
- Text readable against any background
- No performance degradation (< 2x video length to process)

---

## Phase 4: Hook Overlays (Week 4-5)

### Stage 4.1: Hook Text Image Generator
**Estimated Time:** 3 days  
**Priority:** HIGH (Hooks = 30% engagement boost)

#### Tasks:
- [ ] Download NotoSerif-Bold font (or similar)
- [ ] Create `hook_generator.py` module
- [ ] Implement PIL-based image generation
- [ ] Add rounded corners (20px radius)
- [ ] Add drop shadow effect
- [ ] Implement intelligent text wrapping (pixel-based)
- [ ] Make box size responsive to text length

#### Implementation Details:
```python
# hook_generator.py
import os
import urllib.request
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_URL = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSerif/NotoSerif-Bold.ttf"
FONT_DIR = "fonts"
FONT_PATH = os.path.join(FONT_DIR, "NotoSerif-Bold.ttf")

def download_font_if_needed():
    """Download serif font for hook text if not present."""
    if not os.path.exists(FONT_DIR):
        os.makedirs(FONT_DIR)
    if not os.path.exists(FONT_PATH):
        print(f"⬇️ Downloading font from {FONT_URL}...")
        req = urllib.request.Request(FONT_URL, 
                                     headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, \
             open(FONT_PATH, 'wb') as out_file:
            out_file.write(response.read())

def create_hook_image(text, target_width, output_path, font_scale=1.0):
    """
    Generate white rounded box with black serif text.
    
    Features:
    - Rounded corners (20px)
    - Drop shadow (5px offset, gaussian blur)
    - Auto-wrap text to fit width
    - Padding: 30px horizontal, 25px vertical
    - Line spacing: 20px
    """
    download_font_if_needed()
    
    # Configuration
    padding_x = 30
    padding_y = 25
    line_spacing = 20
    corner_radius = 20
    shadow_offset = (5, 5)
    
    # Font size: ~5% of target width
    base_font_size = int(target_width * 0.05)
    font_size = int(base_font_size * font_scale)
    font = ImageFont.truetype(FONT_PATH, font_size)
    
    # Wrap text (pixel-based)
    max_text_width = target_width - (2 * padding_x)
    lines = wrap_text_to_width(text, font, max_text_width)
    
    # Calculate box dimensions
    max_line_width = max(get_text_width(line, font) for line in lines)
    box_width = max(max_line_width + 2 * padding_x, int(target_width * 0.3))
    
    text_heights = [get_text_height(line, font) for line in lines]
    total_text_height = sum(text_heights) + (len(lines) - 1) * line_spacing
    box_height = total_text_height + 2 * padding_y
    
    # Create canvas with space for shadow
    canvas_w = box_width + 40
    canvas_h = box_height + 40
    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw shadow
    shadow_box = [(20 + shadow_offset[0], 20 + shadow_offset[1]),
                  (20 + box_width + shadow_offset[0], 
                   20 + box_height + shadow_offset[1])]
    draw.rounded_rectangle(shadow_box, radius=corner_radius, 
                          fill=(0, 0, 0, 100))
    
    # Blur shadow
    img = img.filter(ImageFilter.GaussianBlur(5))
    
    # Draw white box
    draw_final = ImageDraw.Draw(img)
    main_box = [(20, 20), (20 + box_width, 20 + box_height)]
    draw_final.rounded_rectangle(main_box, radius=corner_radius, 
                                 fill=(255, 255, 255, 240))
    
    # Draw text (centered)
    current_y = 20 + padding_y
    for i, line in enumerate(lines):
        line_w = get_text_width(line, font)
        x = 20 + (box_width - line_w) // 2
        draw_final.text((x, current_y), line, font=font, fill="black")
        current_y += text_heights[i] + line_spacing
    
    img.save(output_path)
    return output_path, canvas_w, canvas_h
```

#### Files to Create:
- `hook_generator.py`
- `fonts/` directory (auto-created)

#### Files to Modify:
- `requirements.txt` - add `Pillow>=10.0.0`

#### Success Criteria:
- Hook images render correctly at any resolution
- Text wraps intelligently without word breaks
- Readable on mobile screens
- Shadows look professional

---

### Stage 4.2: Overlay Hooks on Video
**Estimated Time:** 1 day  
**Priority:** HIGH

#### Tasks:
- [ ] Implement FFmpeg overlay filter
- [ ] Position at top 20% of frame (or configurable)
- [ ] Center horizontally
- [ ] Display for 3-5 seconds at clip start

#### Implementation Details:
```python
# hook_generator.py (continued)

def add_hook_to_video(video_path, hook_text, output_path, 
                      position='top', font_scale=1.0, duration=5.0):
    """
    Overlay hook text on video.
    
    Args:
        position: 'top' (20%), 'center' (50%), 'bottom' (70%)
        font_scale: Multiplier for font size
        duration: How long to show hook (seconds)
    """
    # Get video dimensions
    video_width, video_height = get_video_dimensions(video_path)
    
    # Generate hook image
    target_width = int(video_width * 0.9)  # 90% of video width
    hook_img_path = f"temp_hook_{os.path.basename(video_path)}.png"
    img_path, box_w, box_h = create_hook_image(
        hook_text, target_width, hook_img_path, font_scale
    )
    
    # Calculate position
    overlay_x = (video_width - box_w) // 2  # Center horizontally
    
    if position == 'center':
        overlay_y = (video_height - box_h) // 2
    elif position == 'bottom':
        overlay_y = int(video_height * 0.70)
    else:  # top
        overlay_y = int(video_height * 0.20)
    
    # FFmpeg overlay with fade out
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', img_path,
        '-filter_complex',
        f"[1:v]fade=out:st={duration-0.5}:d=0.5:alpha=1[fg];"
        f"[0:v][fg]overlay={overlay_x}:{overlay_y}:enable='between(t,0,{duration})'",
        '-c:a', 'copy',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        output_path
    ]
    
    subprocess.run(ffmpeg_cmd, check=True)
    
    # Cleanup temp image
    if os.path.exists(hook_img_path):
        os.remove(hook_img_path)
```

#### Success Criteria:
- Hook appears at correct position
- Fades out smoothly
- No audio glitches
- Process time < 1.5x clip duration

---

## Phase 5: AI Video Effects (Week 5-6)

### Stage 5.1: Gemini Filter Generation
**Estimated Time:** 3 days  
**Priority:** MEDIUM (Nice to have, complex)

#### Tasks:
- [ ] Create `ai_effects.py` module
- [ ] Design comprehensive Gemini prompt for FFmpeg filters
- [ ] Implement filter string sanitization
- [ ] Add safety checks (validate syntax, preserve resolution)
- [ ] Handle filter generation errors gracefully

#### Implementation Details:
```python
# ai_effects.py
from google import genai
import re

EFFECTS_PROMPT_TEMPLATE = """
You are an expert FFmpeg video editor. Generate a video filter string 
to make this clip MORE VIRAL with contextual effects.

Video: {width}x{height}, {fps} fps, {duration}s
Transcript: {transcript}

RULES:
1. Analyze transcript for key moments
2. Apply effects ONLY when contextually relevant:
   - Zoom (zoompan) for emphasis at punch lines
   - Contrast/saturation for mood changes
   - NO random effects
3. Use timeline editing: enable='between(t,start,end)'
4. NEVER use comparison operators (<, >, <=, >=)
5. USE functions: lt(), lte(), gt(), gte(), between()
6. Preserve exact resolution: {width}x{height}
7. Set zoompan: s={width}x{height}:fps={fps}:d=1

Example:
zoompan=z='if(between(on,0,75),1.1,if(between(on,76,150),1.3,1.15))':s={width}x{height}:fps={fps}:d=1,eq=contrast=1.2:enable='between(t,5,8)'

Return JSON:
{{
  "filter_string": "..."
}}
"""

def generate_video_effects(video_path, transcript, duration, 
                          width, height, fps):
    """
    Ask Gemini to generate contextual FFmpeg filters.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    prompt = EFFECTS_PROMPT_TEMPLATE.format(
        width=width, height=height, fps=fps,
        duration=duration, transcript=transcript
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    
    # Parse and sanitize
    filter_data = json.loads(response.text)
    filter_string = filter_data.get('filter_string', '')
    
    # Sanitize (convert <, >, <=, >= to functions)
    filter_string = sanitize_filter_string(filter_string)
    
    # Enforce zoompan output size
    filter_string = enforce_zoompan_output_size(filter_string, width, height)
    
    return filter_string

def sanitize_filter_string(filter_string):
    """
    Convert comparison operators to FFmpeg functions.
    t>=5 → gte(t,5)
    on<100 → lt(on,100)
    """
    patterns = [
        (r'([A-Za-z_]\w*)\s*>=\s*(-?\d+(?:\.\d+)?)', r'gte(\1,\2)'),
        (r'([A-Za-z_]\w*)\s*<=\s*(-?\d+(?:\.\d+)?)', r'lte(\1,\2)'),
        (r'([A-Za-z_]\w*)\s*>\s*(-?\d+(?:\.\d+)?)', r'gt(\1,\2)'),
        (r'([A-Za-z_]\w*)\s*<\s*(-?\d+(?:\.\d+)?)', r'lt(\1,\2)'),
    ]
    
    for pattern, replacement in patterns:
        filter_string = re.sub(pattern, replacement, filter_string)
    
    return filter_string
```

#### Files to Create:
- `ai_effects.py`

#### Success Criteria:
- Generate valid FFmpeg filters 90% of the time
- Effects align with video content
- No crashes from malformed filters
- Fallback to no effects on failure

---

### Stage 5.2: Apply Effects Safely
**Estimated Time:** 2 days  
**Priority:** MEDIUM

#### Tasks:
- [ ] Implement effect application with error handling
- [ ] Add dry-run mode (validate filter without applying)
- [ ] Log all generated filters for debugging
- [ ] Create fallback: copy video if effects fail

#### Implementation Details:
```python
# ai_effects.py (continued)

def apply_effects(input_path, output_path, filter_string, 
                  width, height, dry_run=False):
    """
    Apply AI-generated effects with safety checks.
    """
    if not filter_string:
        # No effects, just copy
        shutil.copy(input_path, output_path)
        return True
    
    # Add setsar to ensure square pixels
    if 'setsar=' not in filter_string:
        filter_string = f"{filter_string},setsar=1"
    
    print(f"🎬 AI Filter: {filter_string}")
    
    if dry_run:
        # Validate syntax only
        return validate_filter_syntax(filter_string)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vf', filter_string,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '-c:a', 'copy',
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, 
                               text=True, timeout=300)
        if result.returncode != 0:
            print(f"⚠️ Effects failed, copying original")
            shutil.copy(input_path, output_path)
            return False
        return True
    except Exception as e:
        print(f"⚠️ Effects error: {e}, copying original")
        shutil.copy(input_path, output_path)
        return False
```

#### Success Criteria:
- Never crash the pipeline
- Graceful fallback on errors
- Effects enhance (not distract from) content

---

## Phase 6: Integration & Pipeline (Week 6-7)

### Stage 6.1: Update Main Pipeline
**Estimated Time:** 3 days  
**Priority:** HIGH

#### Tasks:
- [ ] Refactor `main.py` to support new workflow
- [ ] Add `--detect-viral` flag
- [ ] Implement batch clip processing
- [ ] Create output directory structure
- [ ] Add progress tracking for multi-clip jobs

#### Implementation Details:
```python
# main.py (updated)

def process_video_with_viral_detection(input_video, output_dir, args):
    """
    Full viral shorts pipeline.
    
    1. Convert to vertical (existing logic)
    2. Transcribe with faster-whisper
    3. Detect viral moments with Gemini
    4. For each clip:
       a. Extract clip
       b. Process to vertical
       c. Generate subtitles
       d. Burn subtitles
       e. Add hook overlay
       f. Apply AI effects (if enabled)
       g. Save metadata
    """
    # Step 1: Transcribe
    print("🎙️ Step 1/6: Transcribing video...")
    transcript = transcribe_video(input_video)
    
    # Step 2: Get video duration
    duration = get_video_duration(input_video)
    
    # Step 3: Detect viral clips
    print("🤖 Step 2/6: Analyzing viral moments...")
    clips_data = detect_viral_clips(transcript, duration)
    
    if not clips_data or 'shorts' not in clips_data:
        print("❌ No viral clips detected. Exiting.")
        return
    
    print(f"🔥 Found {len(clips_data['shorts'])} viral clips!")
    
    # Step 4: Process each clip
    for i, clip_info in enumerate(clips_data['shorts']):
        print(f"\n{'='*60}")
        print(f"🎬 Processing Clip {i+1}/{len(clips_data['shorts'])}")
        print(f"   {clip_info['start']:.1f}s → {clip_info['end']:.1f}s")
        print(f"   Hook: {clip_info['viral_hook_text']}")
        print(f"{'='*60}\n")
        
        # 4a. Extract clip
        clip_temp = os.path.join(output_dir, f"temp_clip_{i+1}.mp4")
        extract_clip(input_video, clip_info['start'], 
                    clip_info['end'], clip_temp)
        
        # 4b. Process to vertical
        clip_vertical = os.path.join(output_dir, f"clip_{i+1}_vertical.mp4")
        process_video_to_vertical(clip_temp, clip_vertical)
        
        # 4c. Add subtitles
        if args.add_subtitles:
            clip_with_subs = os.path.join(output_dir, 
                                         f"clip_{i+1}_subs.mp4")
            add_subtitles_to_clip(clip_vertical, clip_with_subs, 
                                 transcript, clip_info)
            current_video = clip_with_subs
        else:
            current_video = clip_vertical
        
        # 4d. Add hook overlay
        if args.add_hooks:
            clip_with_hook = os.path.join(output_dir, 
                                         f"clip_{i+1}_hook.mp4")
            add_hook_to_video(current_video, 
                            clip_info['viral_hook_text'],
                            clip_with_hook)
            current_video = clip_with_hook
        
        # 4e. Apply AI effects
        if args.ai_effects:
            clip_final = os.path.join(output_dir, f"clip_{i+1}_final.mp4")
            apply_ai_effects(current_video, clip_final, 
                           transcript, clip_info)
            current_video = clip_final
        
        # 4f. Rename to final name
        final_name = sanitize_filename(
            f"clip_{i+1}_{clip_info['viral_hook_text'][:30]}.mp4"
        )
        final_path = os.path.join(output_dir, final_name)
        os.rename(current_video, final_path)
        
        # 4g. Save metadata
        metadata_path = os.path.join(output_dir, 
                                     f"clip_{i+1}_metadata.json")
        save_clip_metadata(clip_info, transcript, metadata_path)
        
        # Cleanup temps
        cleanup_temp_files(output_dir, i+1)
        
        print(f"✅ Clip {i+1} complete: {final_path}\n")
    
    print(f"\n🎉 All {len(clips_data['shorts'])} clips processed!")
```

#### Files to Modify:
- `main.py` - major refactor

#### Success Criteria:
- Process 10 clips without errors
- Proper error handling at each stage
- Clean output directory structure
- Progress visible throughout

---

### Stage 6.2: Enhance CLI
**Estimated Time:** 2 days  
**Priority:** MEDIUM

#### Tasks:
- [ ] Add new command-line flags
- [ ] Update help documentation
- [ ] Add preset modes (quick, balanced, premium)
- [ ] Implement config file support (optional)

#### New CLI Flags:
```bash
# Viral Detection
--detect-viral          # Enable Gemini viral moment detection
--skip-viral            # Process whole video without detection

# Subtitles
--add-subtitles         # Generate and burn subtitles
--subtitle-position     # top, center, bottom (default: bottom)
--subtitle-style        # preset names (tiktok, minimal, bold)
--max-chars            # Max chars per subtitle line (default: 20)

# Hooks
--add-hooks            # Add viral hook text overlays
--hook-position        # top, center, bottom (default: top)
--hook-duration        # How long to show (default: 5s)

# Effects
--ai-effects           # Apply AI-generated video effects
--effects-intensity    # subtle, balanced, dramatic

# Output
--platforms            # Comma-separated: tiktok,instagram,youtube
--output-template      # Naming pattern for clips

# Presets
--preset quick         # Fast processing, minimal effects
--preset balanced      # Default settings
--preset premium       # All features, best quality
```

#### Example Commands:
```bash
# Quick mode: Just vertical + subtitles
python main.py -i video.mp4 -o clips/ --detect-viral --add-subtitles

# Premium mode: Everything
python main.py -i video.mp4 -o clips/ --preset premium

# Custom configuration
python main.py -i video.mp4 -o clips/ \
  --detect-viral \
  --add-subtitles --subtitle-position bottom \
  --add-hooks --hook-duration 4 \
  --ai-effects --effects-intensity subtle \
  --platforms tiktok,instagram
```

#### Files to Modify:
- `main.py` - argparse setup

#### Success Criteria:
- Help text clear and comprehensive
- Presets work correctly
- All flags validated before processing

---

## Phase 7: Metadata & Export (Week 7)

### Stage 7.1: Comprehensive Metadata System
**Estimated Time:** 2 days  
**Priority:** MEDIUM

#### Tasks:
- [ ] Enhance metadata JSON format
- [ ] Add performance predictions (optional, using Gemini)
- [ ] Include video stats (resolution, duration, file size)
- [ ] Track processing stats (time taken per stage)

#### Enhanced Metadata Format:
```json
{
  "clip_number": 1,
  "source_video": "original_video.mp4",
  "timestamps": {
    "start": 12.340,
    "end": 37.900,
    "duration": 25.560
  },
  "viral_hook_text": "Did you know this trick?",
  "platforms": {
    "tiktok": {
      "description": "Game-changing productivity hack! 🚀 #productivity #lifehack #viral",
      "hashtags": ["productivity", "lifehack", "viral"],
      "optimal_post_time": "7-9 PM on weekdays"
    },
    "instagram": {
      "description": "This changed everything for me 💡 Follow for more tips!",
      "hashtags": ["reels", "productivity", "hack"]
    },
    "youtube": {
      "title": "This Productivity Hack Will Change Your Life! 🔥",
      "tags": ["productivity", "tutorial", "lifehack"]
    }
  },
  "transcript_segment": "So here's what I discovered...",
  "effects_applied": {
    "vertical_crop": true,
    "subtitles": true,
    "hook_overlay": true,
    "ai_effects": false
  },
  "video_stats": {
    "resolution": "1080x1920",
    "fps": 30,
    "file_size_mb": 12.5,
    "codec": "h264"
  },
  "processing_stats": {
    "total_time_seconds": 45.2,
    "stages": {
      "extraction": 5.1,
      "vertical_processing": 15.3,
      "subtitles": 12.8,
      "hooks": 8.2,
      "effects": 0.0
    }
  },
  "viral_potential": {
    "score": 8.5,
    "reasoning": "Strong hook, clear value proposition, trending topic"
  },
  "created_at": "2026-04-02T10:30:45Z",
  "tool_version": "2.0.0"
}
```

#### Files to Modify:
- `metadata_generator.py` - expand format

#### Success Criteria:
- All metadata accurate
- JSON validates against schema
- Useful for analytics

---

### Stage 7.2: Batch Processing & Reports
**Estimated Time:** 2 days  
**Priority:** LOW

#### Tasks:
- [ ] Generate summary report after batch processing
- [ ] Create CSV export of all clips
- [ ] Add thumbnail generation (first frame)
- [ ] Create master metadata JSON with all clips

#### Summary Report Example:
```
╔══════════════════════════════════════════════════════════╗
║           VIRAL SHORTS GENERATION REPORT                 ║
╚══════════════════════════════════════════════════════════╝

📹 Source Video: my_podcast_episode.mp4
⏱️  Duration: 1h 32m 15s
📊 Clips Generated: 8

┌──────────────────────────────────────────────────────────┐
│ Clip 1: "Did you know this trick?"                       │
├──────────────────────────────────────────────────────────┤
│ Time: 2:34 → 2:59 (25s)                                  │
│ Viral Score: 9.2/10                                       │
│ File: clip_1_did_you_know_this_trick.mp4 (14.2 MB)      │
│ Effects: ✅ Subtitles ✅ Hook ❌ AI Effects               │
└──────────────────────────────────────────────────────────┘

[... 7 more clips ...]

🎉 Summary:
   • Total clips: 8
   • Total duration: 3m 42s
   • Total size: 112.5 MB
   • Average viral score: 8.1/10
   • Processing time: 6m 23s
   
📁 Output Directory: ./output/my_podcast_episode/
📊 Full Report: ./output/my_podcast_episode/report.txt
📋 Metadata: ./output/my_podcast_episode/master_metadata.json
```

#### Files to Create:
- `report_generator.py`

#### Success Criteria:
- Report generated automatically
- All stats accurate
- Helpful for content planning

---

## 🚦 Milestones & Checkpoints

### Milestone 1: MVP (End of Week 3)
**Deliverables:**
- ✅ Transcription working
- ✅ Gemini viral detection working
- ✅ Basic subtitle generation
- ✅ Clip extraction functional

**Demo:** Process 1 video → get 5 clips with subtitles

---

### Milestone 2: Feature Complete (End of Week 5)
**Deliverables:**
- ✅ All Phase 1-4 features implemented
- ✅ Subtitles + Hooks working
- ✅ CLI flags functional
- ✅ Metadata generation

**Demo:** Process 1 video → get 5 clips with subtitles + hooks + metadata

---

### Milestone 3: Production Ready (End of Week 7)
**Deliverables:**
- ✅ All phases complete (including AI effects)
- ✅ Error handling robust
- ✅ Documentation updated
- ✅ Example videos processed

**Demo:** Process multiple videos → generate 20+ viral clips

---

## 📚 Documentation Updates

As you implement each phase, update these files:

### README.md
- [ ] Add new features section
- [ ] Update usage examples
- [ ] Add before/after screenshots
- [ ] Link to examples

### CHANGELOG.md (create new)
- [ ] Document version 2.0.0 changes
- [ ] List all new features
- [ ] Note breaking changes (if any)

### API_KEYS.md (create new)
- [ ] Document required API keys
- [ ] Show how to get Gemini API key
- [ ] Explain cost structure
- [ ] Add troubleshooting

---

## 🧪 Testing Strategy

### Unit Tests
- [ ] `test_transcribe.py` - transcription accuracy
- [ ] `test_viral_detector.py` - Gemini responses
- [ ] `test_subtitle_generator.py` - SRT formatting
- [ ] `test_hook_generator.py` - image generation
- [ ] `test_metadata.py` - JSON validation

### Integration Tests
- [ ] End-to-end pipeline with sample video
- [ ] Error handling (invalid inputs, API failures)
- [ ] Performance benchmarks

### Test Videos
Create a test suite with:
- [ ] Short video (1 min) - quick tests
- [ ] Medium video (10 min) - typical use case
- [ ] Long video (60 min) - stress test
- [ ] Various aspect ratios
- [ ] Different languages

---

## 🎯 Success Metrics

Track these KPIs to measure success:

### Performance
- Transcription speed: < 0.4x video length
- Clip processing: < 2x clip length per clip
- Total pipeline: < 3x video length

### Quality
- Subtitle accuracy: > 95%
- Viral clip relevance: > 80% (manual review)
- No audio desync in any clip

### Usability
- Setup time: < 5 minutes
- Commands intuitive (no manual needed for basics)
- Error messages helpful

---

## 🚨 Common Issues & Solutions

### Issue: Gemini API Rate Limits
**Solution:** Implement exponential backoff, batch requests

### Issue: FFmpeg Errors with AI Filters
**Solution:** Sanitization + graceful fallback to no effects

### Issue: Subtitle Timing Off
**Solution:** Use word-level timestamps, validate against audio

### Issue: Long Processing Times
**Solution:** 
- Use faster-whisper on CPU
- Parallel processing for multiple clips
- Hardware encoding where available

---

## 🔄 Future Enhancements (Post-Launch)

### Version 2.1
- [ ] Multi-speaker detection (assign colors per speaker)
- [ ] Animated text effects (kinetic typography)
- [ ] Background music integration
- [ ] A/B testing for different hooks

### Version 2.2
- [ ] Web UI (FastAPI + React)
- [ ] Batch video processing (queue system)
- [ ] Direct social media posting (Upload-Post integration)
- [ ] Analytics dashboard (track clip performance)

### Version 3.0
- [ ] Real-time preview
- [ ] Manual clip editing/refinement
- [ ] Custom AI training on your best clips
- [ ] Collaborative features (team workflows)

---

## 📞 Support & Resources

### Documentation
- [Gemini API Docs](https://ai.google.dev/docs)
- [FFmpeg Filter Guide](https://ffmpeg.org/ffmpeg-filters.html)
- [faster-whisper GitHub](https://github.com/guillaumekln/faster-whisper)

### Community
- GitHub Issues: Report bugs
- Discussions: Feature requests, Q&A

---

## ✅ Quick Reference Checklist

Use this to track overall progress:

**Infrastructure:**
- [ ] Transcription module
- [ ] Gemini API integration
- [ ] Clip extraction
- [ ] Metadata system

**Core Features:**
- [ ] Viral moment detection
- [ ] Subtitle generation
- [ ] Subtitle burning
- [ ] Hook image generation
- [ ] Hook overlay
- [ ] AI effects (optional)

**Integration:**
- [ ] Updated main.py pipeline
- [ ] Enhanced CLI
- [ ] Output directory structure
- [ ] Error handling

**Polish:**
- [ ] Documentation updated
- [ ] Tests written
- [ ] Example videos processed
- [ ] README with demos

---

## 🎓 Learning Resources

If you need to understand the underlying tech better:

- **FFmpeg:** [FFmpeg for Beginners](https://img.ly/blog/ultimate-guide-to-ffmpeg/)
- **Whisper:** [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
- **Gemini:** [Google AI Studio](https://aistudio.google.com/)
- **PIL/Pillow:** [Pillow Documentation](https://pillow.readthedocs.io/)

---

## 📝 Notes

- Estimated times are for 1 developer working part-time (4 hours/day)
- Adjust priorities based on your specific needs
- Can skip AI Effects (Phase 5) for faster MVP
- Test after each stage before moving forward
- Keep backups of working versions

---

**Last Updated:** April 2, 2026  
**Version:** 1.0  
**Status:** Ready to implement 🚀
