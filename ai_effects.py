#!/usr/bin/env python3
"""
AI Video Effects Generator

Uses Google Gemini API to generate contextual FFmpeg video filters
based on video content and transcript analysis.
"""

import os
import sys
import json
import re
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import types
import cv2


# Load environment variables
load_dotenv()


EFFECTS_PROMPT_TEMPLATE = """
You are an expert FFmpeg video editor. Generate a video filter string to make this clip MORE VIRAL with contextual effects.

Video Metadata:
- Resolution: {width}x{height}
- FPS: {fps}
- Duration: {duration}s

Clip Transcript:
{transcript}

CRITICAL RULES:
1. Analyze the transcript for key emotional moments
2. Apply effects ONLY when contextually relevant:
   - Zoom (zoompan) for emphasis during punch lines or dramatic reveals
   - Color adjustments (eq) for mood changes
   - DO NOT add random effects
3. Use timeline editing with 'between()' function: enable='between(t,start,end)'
4. NEVER use comparison operators (<, >, <=, >=) - they cause syntax errors
5. ALWAYS use functions: lt(), lte(), gt(), gte(), between()
6. For zoompan filter: MUST set output size: s={width}x{height}:fps={fps}:d=1
7. Preserve exact resolution: {width}x{height}
8. Keep effects subtle and professional

Example Valid Filter:
zoompan=z='if(between(on,0,75),1.1,if(between(on,76,150),1.3,1.15))':s={width}x{height}:fps={fps}:d=1,eq=contrast=1.2:enable='between(t,5,8)'

Translation: Zoom 10% for frames 0-75, zoom 30% for frames 76-150, then 15%. Boost contrast 20% from 5-8 seconds.

If the content doesn't warrant effects, return an empty filter string.

Return ONLY valid JSON:
{{
  "filter_string": "your_filter_here_or_empty_string",
  "reasoning": "Brief explanation of effects applied"
}}
"""


def get_video_metadata(video_path: str) -> Dict:
    """Extract video metadata using OpenCV and ffprobe."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    duration = frame_count / fps if fps > 0 else 0.0
    
    return {
        'width': width,
        'height': height,
        'fps': int(fps),
        'duration': round(duration, 2),
        'frame_count': frame_count
    }


def load_transcript_segment(transcript_path: str, start: float, end: float) -> str:
    """
    Load transcript text for a specific time range.
    
    Args:
        transcript_path: Path to transcript JSON
        start: Start time in seconds
        end: End time in seconds
        
    Returns:
        Transcript text as string
    """
    if not os.path.exists(transcript_path):
        return "No transcript available"
    
    try:
        with open(transcript_path, 'r') as f:
            data = json.load(f)
        
        # Extract text from segments within time range
        segments = data.get('segments', [])
        text_parts = []
        
        for segment in segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check if segment overlaps with our time range
            if seg_start < end and seg_end > start:
                text_parts.append(segment.get('text', '').strip())
        
        return ' '.join(text_parts) if text_parts else "No transcript in this range"
        
    except Exception as e:
        print(f"⚠️ Error loading transcript: {e}")
        return "Error loading transcript"


def sanitize_filter_string(filter_string: str) -> str:
    """
    Convert comparison operators to FFmpeg functions.
    
    Converts:
    - t>=5 → gte(t,5)
    - on<100 → lt(on,100)
    - frame<=200 → lte(frame,200)
    
    Args:
        filter_string: Raw filter string from Gemini
        
    Returns:
        Sanitized filter string
    """
    if not filter_string:
        return ""
    
    # Patterns for sanitization
    patterns = [
        (r'([A-Za-z_]\w*)\s*>=\s*(-?\d+(?:\.\d+)?)', r'gte(\1,\2)'),
        (r'([A-Za-z_]\w*)\s*<=\s*(-?\d+(?:\.\d+)?)', r'lte(\1,\2)'),
        (r'([A-Za-z_]\w*)\s*>\s*(-?\d+(?:\.\d+)?)', r'gt(\1,\2)'),
        (r'([A-Za-z_]\w*)\s*<\s*(-?\d+(?:\.\d+)?)', r'lt(\1,\2)'),
    ]
    
    sanitized = filter_string
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized)
    
    return sanitized


def enforce_zoompan_output_size(filter_string: str, width: int, height: int, fps: int) -> str:
    """
    Ensure zoompan filter has correct output size.
    
    Args:
        filter_string: Filter string to check
        width: Video width
        height: Video height
        fps: Video FPS
        
    Returns:
        Filter string with corrected zoompan parameters
    """
    if not filter_string or 'zoompan' not in filter_string:
        return filter_string
    
    # Check if zoompan already has s= parameter
    if re.search(r'zoompan=.*?s=\d+x\d+', filter_string):
        # Replace with correct dimensions
        corrected = re.sub(
            r'(zoompan=.*?s=)\d+x\d+',
            rf'\g<1>{width}x{height}',
            filter_string
        )
        return corrected
    else:
        # Add s= parameter if missing
        corrected = re.sub(
            r'zoompan=',
            f'zoompan=s={width}x{height}:fps={fps}:d=1,',
            filter_string
        )
        return corrected


def generate_video_effects(video_path: str, transcript_path: str = None,
                          clip_start: float = 0.0, clip_end: float = None) -> Tuple[str, str]:
    """
    Ask Gemini to generate contextual FFmpeg filters.
    
    Args:
        video_path: Path to video file
        transcript_path: Path to transcript JSON (optional)
        clip_start: Clip start time in original video
        clip_end: Clip end time in original video
        
    Returns:
        Tuple of (filter_string, reasoning)
    """
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY not found in environment")
        return "", "No API key available"
    
    # Get video metadata
    metadata = get_video_metadata(video_path)
    
    if clip_end is None:
        clip_end = metadata['duration']
    
    # Load transcript
    transcript_text = ""
    if transcript_path:
        transcript_text = load_transcript_segment(transcript_path, clip_start, clip_end)
    else:
        transcript_text = "No transcript available"
    
    # Build prompt
    prompt = EFFECTS_PROMPT_TEMPLATE.format(
        width=metadata['width'],
        height=metadata['height'],
        fps=metadata['fps'],
        duration=metadata['duration'],
        transcript=transcript_text
    )
    
    try:
        print(f"🤖 Asking Gemini to generate effects...")
        
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7
            )
        )
        
        # Parse response
        result = json.loads(response.text)
        filter_string = result.get('filter_string', '')
        reasoning = result.get('reasoning', 'No reasoning provided')
        
        if not filter_string:
            print("  No effects suggested (content doesn't warrant effects)")
            return "", reasoning
        
        # Sanitize filter string
        filter_string = sanitize_filter_string(filter_string)
        filter_string = enforce_zoompan_output_size(
            filter_string, metadata['width'], metadata['height'], metadata['fps']
        )
        
        print(f"  Generated filter: {filter_string}")
        print(f"  Reasoning: {reasoning}")
        
        return filter_string, reasoning
        
    except Exception as e:
        print(f"⚠️ Gemini API error: {e}")
        return "", f"Error: {e}"


def validate_filter_syntax(filter_string: str, video_path: str) -> bool:
    """
    Validate FFmpeg filter syntax without processing.
    
    Args:
        filter_string: Filter to validate
        video_path: Input video for testing
        
    Returns:
        True if valid, False otherwise
    """
    if not filter_string:
        return True
    
    # Try running ffmpeg with -f null to validate without output
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', filter_string,
        '-f', 'null', '-'
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=5  # Quick validation
        )
        return result.returncode == 0
    except Exception:
        return False


def apply_effects(input_path: str, output_path: str, filter_string: str,
                 dry_run: bool = False) -> bool:
    """
    Apply AI-generated effects with safety checks.
    
    Args:
        input_path: Input video file
        output_path: Output video file
        filter_string: FFmpeg filter string to apply
        dry_run: If True, only validate without applying
        
    Returns:
        True if successful, False otherwise
    """
    if not filter_string:
        # No effects, just copy
        print("  No effects to apply, copying original...")
        shutil.copy(input_path, output_path)
        return True
    
    # Add setsar to ensure square pixels
    if 'setsar=' not in filter_string:
        filter_string = f"{filter_string},setsar=1"
    
    print(f"  Filter: {filter_string}")
    
    if dry_run:
        # Validate syntax only
        print("  [DRY RUN] Validating filter syntax...")
        return validate_filter_syntax(filter_string, input_path)
    
    # Apply effects
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vf', filter_string,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '-c:a', 'copy',
        output_path
    ]
    
    try:
        print("  Applying effects...")
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"  ⚠️ Effects failed: {result.stderr[-500:]}")
            print("  Copying original without effects...")
            shutil.copy(input_path, output_path)
            return False
        
        print("  ✓ Effects applied successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("  ⚠️ Effects processing timed out, copying original...")
        shutil.copy(input_path, output_path)
        return False
    except Exception as e:
        print(f"  ⚠️ Effects error: {e}, copying original...")
        shutil.copy(input_path, output_path)
        return False


def add_effects_from_json(clips_json_path: str, transcript_path: str,
                         input_dir: str = 'vertical_clips',
                         output_dir: str = None) -> int:
    """
    Add AI effects to multiple clips from viral detection JSON.
    
    Args:
        clips_json_path: JSON file with viral clips
        transcript_path: Transcript JSON file
        input_dir: Directory with input clips
        output_dir: Output directory (default: effects/)
        
    Returns:
        Number of clips processed successfully
    """
    if not os.path.exists(clips_json_path):
        print(f"Error: Clips JSON not found: {clips_json_path}")
        return 0
    
    # Load clips JSON
    with open(clips_json_path, 'r') as f:
        data = json.load(f)
    
    clips = data.get('clips', [])
    if not clips:
        print("No clips found in JSON")
        return 0
    
    # Create output directory
    if output_dir is None:
        output_dir = 'effects'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ADDING AI EFFECTS TO {len(clips)} CLIPS")
    print(f"{'='*60}\n")
    
    success_count = 0
    
    for i, clip in enumerate(clips, 1):
        clip_file = f"viral_{i:03d}.mp4"
        input_clip = os.path.join(input_dir, clip_file)
        
        if not os.path.exists(input_clip):
            print(f"⊘ Clip {i}: Not found at {input_clip}, skipping")
            continue
        
        output_clip = os.path.join(output_dir, clip_file)
        
        print(f"\n[{i}/{len(clips)}] Processing: {input_clip}")
        
        # Generate effects
        filter_string, reasoning = generate_video_effects(
            input_clip, transcript_path,
            clip_start=clip.get('start', 0.0),
            clip_end=clip.get('end', 0.0)
        )
        
        # Apply effects
        success = apply_effects(input_clip, output_clip, filter_string)
        
        if success:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{len(clips)} clips processed successfully")
    print(f"{'='*60}\n")
    
    return success_count


def main():
    """Command-line interface for AI effects generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate and apply AI-powered video effects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add AI effects to single video
  python ai_effects.py clip.mp4 transcript.json -o enhanced.mp4
  
  # Batch process from viral detection JSON
  python ai_effects.py --batch viral_clips.json --transcript transcript.json -o effects/
  
  # Dry run (validate only)
  python ai_effects.py clip.mp4 transcript.json --dry-run
        """
    )
    
    parser.add_argument('video', nargs='?', help='Input video file')
    parser.add_argument('transcript', nargs='?', help='Transcript JSON file')
    parser.add_argument('-o', '--output', help='Output video file or directory')
    parser.add_argument('--batch', help='Process multiple clips from viral detection JSON')
    parser.add_argument('--transcript', dest='transcript_file', 
                       help='Transcript JSON (for batch mode)')
    parser.add_argument('--input-dir', default='vertical_clips',
                       help='Input directory for batch mode (default: vertical_clips)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate filters without applying')
    parser.add_argument('-s', '--start', type=float, default=0.0,
                       help='Clip start time (for context)')
    parser.add_argument('-e', '--end', type=float,
                       help='Clip end time (for context)')
    
    args = parser.parse_args()
    
    # Batch mode
    if args.batch:
        if not args.transcript_file:
            print("Error: --transcript required for batch mode")
            sys.exit(1)
        
        success_count = add_effects_from_json(
            args.batch, args.transcript_file,
            input_dir=args.input_dir,
            output_dir=args.output
        )
        sys.exit(0 if success_count > 0 else 1)
    
    # Single video mode
    if not args.video:
        parser.print_help()
        sys.exit(1)
    
    # Generate effects
    filter_string, reasoning = generate_video_effects(
        args.video, args.transcript,
        clip_start=args.start,
        clip_end=args.end
    )
    
    # Apply effects
    output = args.output or args.video.replace('.mp4', '_effects.mp4')
    success = apply_effects(args.video, output, filter_string, dry_run=args.dry_run)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
