"""
Subtitle Generation Module
Author: AI Video Clipping Bot
Purpose: Generate SRT subtitles and burn them onto videos with TikTok-style formatting
"""

import json
import os
import sys
import subprocess
from typing import List, Dict, Any, Tuple
from pathlib import Path


def extract_words_in_range(transcript: Dict, start_time: float, end_time: float) -> List[Dict]:
    """
    Extract words within a specific time range from transcript.
    
    Args:
        transcript: Transcript dictionary with segments and words
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        List of word dictionaries with 'word', 'start', 'end' keys
    """
    words = []
    
    if 'segments' not in transcript:
        return words
    
    for segment in transcript['segments']:
        if 'words' not in segment:
            continue
            
        for word_info in segment['words']:
            word_start = word_info.get('start', 0)
            word_end = word_info.get('end', 0)
            
            # Check if word overlaps with our time range
            if word_end >= start_time and word_start <= end_time:
                words.append({
                    'word': word_info.get('word', '').strip(),
                    'start': word_start,
                    'end': word_end
                })
    
    return words


def group_words_for_subtitles(words: List[Dict], clip_start: float,
                              max_chars: int = 20, max_duration: float = 2.0) -> List[Dict]:
    """
    Group words into subtitle blocks optimized for TikTok/Reels.
    
    Args:
        words: List of word dictionaries with 'word', 'start', 'end'
        clip_start: Clip start time (to adjust timestamps relative to clip)
        max_chars: Maximum characters per subtitle line
        max_duration: Maximum duration per subtitle block in seconds
        
    Returns:
        List of subtitle blocks with 'start', 'end', 'text'
    """
    if not words:
        return []
    
    subtitle_blocks = []
    current_block = []
    current_text = ""
    block_start = None
    block_end = None
    
    for word in words:
        word_text = word['word']
        
        # Skip empty words
        if not word_text:
            continue
        
        # Calculate what the text would be if we add this word
        potential_text = current_text + (' ' if current_text else '') + word_text
        
        # Calculate duration if we add this word
        if block_start is None:
            potential_start = word['start']
        else:
            potential_start = block_start
        potential_end = word['end']
        potential_duration = potential_end - potential_start
        
        # Check if adding this word would exceed limits
        should_close = False
        
        if len(current_block) > 0:
            # Check character limit
            if len(potential_text) > max_chars:
                should_close = True
            # Check duration limit
            elif potential_duration > max_duration:
                should_close = True
        
        if should_close:
            # Finalize current block
            if current_block:
                subtitle_blocks.append({
                    'start': block_start - clip_start,
                    'end': block_end - clip_start,
                    'text': current_text.strip()
                })
            
            # Start new block with current word
            current_block = [word]
            current_text = word_text
            block_start = word['start']
            block_end = word['end']
        else:
            # Add word to current block
            current_block.append(word)
            current_text = potential_text
            if block_start is None:
                block_start = word['start']
            block_end = word['end']
    
    # Add final block
    if current_block:
        subtitle_blocks.append({
            'start': max(0, block_start - clip_start),
            'end': max(0, block_end - clip_start),
            'text': current_text.strip()
        })
    
    return subtitle_blocks


def format_srt_time(seconds: float) -> str:
    """
    Format time in SRT format: HH:MM:SS,mmm
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_ass(subtitle_blocks: List[Dict], output_path: str,
                style: str = 'tiktok', position: str = 'bottom'):
    """
    Generate ASS subtitle file with styling from subtitle blocks.
    
    Args:
        subtitle_blocks: List of subtitle dictionaries with 'start', 'end', 'text'
        output_path: Path to save ASS file
        style: Subtitle style - 'tiktok' or 'box'
        position: Position - 'top', 'center', 'bottom'
    """
    # Position mapping (ASS alignment)
    alignment_map = {
        'top': 8,      # Top center
        'center': 5,   # Middle center  
        'bottom': 2    # Bottom center
    }
    
    alignment = alignment_map.get(position, 2)
    
    # Build style based on preset
    if style == 'tiktok':
        # TikTok style: Bold white text with thick black outline
        style_line = (
            f"Style: Default,Arial Black,18,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            f"1,0,0,0,100,100,0,0,1,3,0,{alignment},10,10,40,1"
        )
    else:  # box style
        # Box style: White text with semi-transparent black background
        style_line = (
            f"Style: Default,Arial,16,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            f"1,0,0,0,100,100,0,0,1,2,2,{alignment},10,10,40,1"
        )
    
    # ASS header
    ass_content = """[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""
    ass_content += style_line + "\n\n"
    
    # Events section
    ass_content += "[Events]\n"
    ass_content += "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    
    for block in subtitle_blocks:
        start_time = format_ass_time(block['start'])
        end_time = format_ass_time(block['end'])
        text = block['text'].replace('\n', '\\N')
        
        ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)


def format_ass_time(seconds: float) -> str:
    """
    Format time in ASS format: H:MM:SS.cc
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def generate_srt(subtitle_blocks: List[Dict], output_path: str):
    """
    Generate SRT subtitle file from subtitle blocks.
    
    Args:
        subtitle_blocks: List of subtitle dictionaries with 'start', 'end', 'text'
        output_path: Path to save SRT file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, block in enumerate(subtitle_blocks, start=1):
            # Subtitle number
            f.write(f"{i}\n")
            
            # Timestamp
            start_time = format_srt_time(block['start'])
            end_time = format_srt_time(block['end'])
            f.write(f"{start_time} --> {end_time}\n")
            
            # Text
            f.write(f"{block['text']}\n")
            
            # Blank line
            f.write("\n")


def burn_subtitles(video_path: str, subtitle_path: str, output_path: str) -> bool:
    """
    Burn subtitles onto video (supports both SRT and ASS files).
    
    Args:
        video_path: Input video file
        subtitle_path: Subtitle file (SRT or ASS)
        output_path: Output video file
        
    Returns:
        True if successful, False otherwise
    """
    import shutil
    import tempfile
    
    # Copy subtitle file to temp location with simple name (FFmpeg has issues with special chars)
    temp_dir = tempfile.gettempdir()
    temp_subtitle = os.path.join(temp_dir, f"temp_sub_{os.getpid()}{os.path.splitext(subtitle_path)[1]}")
    shutil.copy(subtitle_path, temp_subtitle)
    
    try:
        # Use subtitles filter for both ASS and SRT (it handles both)
        # Escape the path properly for FFmpeg filter syntax
        escaped_path = temp_subtitle.replace('\\', '\\\\\\\\').replace(':', '\\\\:')
        filter_str = f"subtitles={escaped_path}"
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', filter_str,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'copy',
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error burning subtitles: {e.stderr}", file=sys.stderr)
        return False
    finally:
        # Clean up temp subtitle file
        if os.path.exists(temp_subtitle):
            os.remove(temp_subtitle)


def add_subtitles_to_clip(video_path: str, transcript: Dict, clip_start: float, clip_end: float,
                         output_path: str = None, style: str = 'tiktok', position: str = 'bottom',
                         max_chars: int = 20, max_duration: float = 2.0) -> bool:
    """
    Complete pipeline: extract words, generate SRT, burn subtitles.
    
    Args:
        video_path: Input video file
        transcript: Full transcript dictionary
        clip_start: Clip start time in original video
        clip_end: Clip end time in original video
        output_path: Output video file (defaults to video_path with '_subtitled' suffix)
        style: Subtitle style ('tiktok' or 'box')
        position: Subtitle position ('top', 'center', 'bottom')
        max_chars: Maximum characters per subtitle line
        max_duration: Maximum duration per subtitle block
        
    Returns:
        True if successful, False otherwise
    """
    if output_path is None:
        base = os.path.splitext(video_path)[0]
        output_path = f"{base}_subtitled.mp4"
    
    # Extract words for this clip
    words = extract_words_in_range(transcript, clip_start, clip_end)
    
    if not words:
        print(f"Warning: No words found for clip {clip_start}-{clip_end}")
        return False
    
    # Group words into subtitle blocks
    subtitle_blocks = group_words_for_subtitles(words, clip_start, max_chars, max_duration)
    
    if not subtitle_blocks:
        print(f"Warning: No subtitle blocks generated for clip")
        return False
    
    # Generate ASS file (with styling built-in)
    ass_path = os.path.splitext(video_path)[0] + '.ass'
    generate_ass(subtitle_blocks, ass_path, style, position)
    
    print(f"  Generated {len(subtitle_blocks)} subtitle blocks")
    
    # Burn subtitles onto video
    success = burn_subtitles(video_path, ass_path, output_path)
    
    # Clean up ASS file
    if os.path.exists(ass_path):
        os.remove(ass_path)
    
    return success


# CLI interface
def main():
    """Command-line interface for subtitle generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate and burn subtitles onto video clips',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add subtitles to entire video
  python subtitle_generator.py clip.mp4 transcript.json
  
  # Add subtitles to specific time range
  python subtitle_generator.py clip.mp4 transcript.json --start 10.5 --end 45.2
  
  # Custom style and position
  python subtitle_generator.py clip.mp4 transcript.json -s 10 -e 40 --style box --position top
  
  # Adjust subtitle parameters
  python subtitle_generator.py clip.mp4 transcript.json --max-chars 25 --max-duration 2.5
        """
    )
    
    parser.add_argument('video', help='Input video file')
    parser.add_argument('transcript', help='Transcript JSON file (from transcribe.py)')
    parser.add_argument('-s', '--start', type=float, default=0.0,
                       help='Clip start time in original video (seconds, default: 0.0)')
    parser.add_argument('-e', '--end', type=float, default=None,
                       help='Clip end time in original video (seconds, default: video duration)')
    parser.add_argument('-o', '--output', help='Output video file')
    parser.add_argument('--style', choices=['tiktok', 'box'], default='tiktok',
                       help='Subtitle style (default: tiktok)')
    parser.add_argument('--position', choices=['top', 'center', 'bottom'], default='bottom',
                       help='Subtitle position (default: bottom)')
    parser.add_argument('--max-chars', type=int, default=20,
                       help='Maximum characters per subtitle line (default: 20)')
    parser.add_argument('--max-duration', type=float, default=2.0,
                       help='Maximum duration per subtitle block in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    # Load transcript
    if not os.path.exists(args.transcript):
        print(f"Error: Transcript file not found: {args.transcript}")
        sys.exit(1)
    
    with open(args.transcript, 'r') as f:
        transcript = json.load(f)
    
    # Check video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Get video duration if end time not specified
    if args.end is None:
        import cv2
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        args.end = frame_count / fps if fps > 0 else 0.0
    
    # Add subtitles
    print(f"Adding subtitles to {args.video}")
    print(f"Clip range: {args.start:.2f}s - {args.end:.2f}s")
    
    success = add_subtitles_to_clip(
        args.video,
        transcript,
        args.start,
        args.end,
        output_path=args.output,
        style=args.style,
        position=args.position,
        max_chars=args.max_chars,
        max_duration=args.max_duration
    )
    
    if success:
        output = args.output or args.video.replace('.mp4', '_subtitled.mp4')
        print(f"✓ Subtitles added successfully!")
        print(f"  Output: {output}")
        sys.exit(0)
    else:
        print("✗ Failed to add subtitles")
        sys.exit(1)


if __name__ == '__main__':
    main()
