"""
Clip Extraction Module
Author: AI Video Clipping Bot
Purpose: Extract precise video clips using FFmpeg based on timestamps from viral detection
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds as float
        
    Raises:
        RuntimeError: If ffprobe fails or duration cannot be determined
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}")
    except ValueError:
        raise RuntimeError(f"Could not parse duration from ffprobe output")


def extract_clip(input_video: str, start: float, end: float, output_path: str,
                quality: str = 'balanced', audio_codec: str = 'aac') -> bool:
    """
    Extract a precise clip from video using FFmpeg re-encoding.
    
    Uses re-encoding (not stream copy) to ensure frame-accurate cuts and 
    avoid audio sync issues. The `-ss` before `-i` enables fast seeking,
    and `-to` ensures precise end time.
    
    Args:
        input_video: Path to source video file
        start: Start time in seconds (float)
        end: End time in seconds (float)
        output_path: Path for output clip
        quality: Quality preset - 'fast' (CRF 28), 'balanced' (CRF 23), 'high' (CRF 18)
        audio_codec: Audio codec to use (default: 'aac')
        
    Returns:
        True if extraction successful, False otherwise
        
    Raises:
        FileNotFoundError: If input video doesn't exist
        ValueError: If start >= end or timestamps are invalid
    """
    # Validate inputs (check logic first, then file existence)
    if start >= end:
        raise ValueError(f"Start time ({start}s) must be before end time ({end}s)")
    
    if start < 0:
        raise ValueError(f"Start time cannot be negative: {start}s")
    
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    
    # Quality presets
    quality_presets = {
        'fast': {'crf': '28', 'preset': 'veryfast'},
        'balanced': {'crf': '23', 'preset': 'fast'},
        'high': {'crf': '18', 'preset': 'slow'}
    }
    
    if quality not in quality_presets:
        quality = 'balanced'
    
    crf = quality_presets[quality]['crf']
    preset = quality_presets[quality]['preset']
    
    # Build FFmpeg command
    # -ss before -i enables fast seeking (input seeking)
    # -to specifies end time (more reliable than -t duration)
    # Re-encode to ensure frame accuracy and audio sync
    command = [
        'ffmpeg', '-y',
        '-ss', str(start),
        '-to', str(end),
        '-i', input_video,
        '-c:v', 'libx264',
        '-crf', crf,
        '-preset', preset,
        '-c:a', audio_codec,
        '-avoid_negative_ts', 'make_zero',  # Ensure timestamps start at 0
        output_path
    ]
    
    try:
        # Run FFmpeg with stderr capture for error reporting
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting clip: {e.stderr}", file=sys.stderr)
        return False


def extract_clips_from_json(video_path: str, clips_json_path: str,
                           output_dir: str = 'clips',
                           quality: str = 'balanced',
                           prefix: str = 'clip',
                           convert_vertical: bool = False) -> List[Dict[str, Any]]:
    """
    Extract multiple clips from a video based on viral clips JSON file.
    
    Args:
        video_path: Path to source video file
        clips_json_path: Path to viral clips JSON file (output from viral_detector.py)
        output_dir: Directory to save extracted clips (created if doesn't exist)
        quality: Quality preset for encoding
        prefix: Prefix for output clip filenames
        convert_vertical: If True, convert clips to 9:16 vertical format using main.py
        
    Returns:
        List of dictionaries with extraction results:
        [
            {
                'clip_number': 1,
                'start': 12.34,
                'end': 37.90,
                'output_path': 'clips/clip_001.mp4',
                'success': True,
                'error': None
            },
            ...
        ]
    """
    # Load clips JSON
    if not os.path.exists(clips_json_path):
        raise FileNotFoundError(f"Clips JSON not found: {clips_json_path}")
    
    with open(clips_json_path, 'r') as f:
        data = json.load(f)
    
    if 'clips' not in data or not data['clips']:
        print("No clips found in JSON file")
        return []
    
    clips = data['clips']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract each clip
    results = []
    total_clips = len(clips)
    
    print(f"\nExtracting {total_clips} clips from: {video_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    for i, clip in enumerate(clips, start=1):
        start = clip['start']
        end = clip['end']
        duration = end - start
        
        # Generate output filename
        output_filename = f"{prefix}_{i:03d}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\nClip {i}/{total_clips}: {start:.2f}s -> {end:.2f}s ({duration:.1f}s)")
        print(f"  Hook: {clip.get('viral_hook_text', 'N/A')}")
        print(f"  Output: {output_path}")
        
        # Extract clip
        try:
            success = extract_clip(video_path, start, end, output_path, quality)
            
            result = {
                'clip_number': i,
                'start': start,
                'end': end,
                'duration': duration,
                'output_path': output_path,
                'success': success,
                'error': None,
                'metadata': clip
            }
            
            if success:
                # Get actual file size
                file_size = os.path.getsize(output_path)
                file_size_mb = file_size / (1024 * 1024)
                result['file_size'] = file_size
                print(f"  ✓ Extracted! ({file_size_mb:.2f} MB)")
                
                # Convert to vertical if requested
                if convert_vertical:
                    print(f"  🔄 Converting to vertical format...")
                    vertical_path = output_path.replace('.mp4', '_vertical.mp4')
                    
                    try:
                        # Call main.py to convert to vertical
                        cmd = [
                            sys.executable, 'main.py',
                            '-i', output_path,
                            '-o', vertical_path,
                            '--quality', quality
                        ]
                        
                        result_conv = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        # Replace horizontal clip with vertical one
                        if os.path.exists(vertical_path):
                            os.remove(output_path)
                            os.rename(vertical_path, output_path)
                            
                            # Update file size
                            file_size = os.path.getsize(output_path)
                            file_size_mb = file_size / (1024 * 1024)
                            result['file_size'] = file_size
                            result['vertical'] = True
                            print(f"  ✓ Vertical conversion complete! ({file_size_mb:.2f} MB)")
                        else:
                            print(f"  ⚠ Vertical conversion failed - keeping horizontal clip")
                            result['vertical'] = False
                            
                    except subprocess.CalledProcessError as e:
                        print(f"  ⚠ Vertical conversion error: {e.stderr[:100]}")
                        print(f"  ⚠ Keeping horizontal clip")
                        result['vertical'] = False
                    except Exception as e:
                        print(f"  ⚠ Vertical conversion failed: {e}")
                        print(f"  ⚠ Keeping horizontal clip")
                        result['vertical'] = False
                else:
                    result['vertical'] = False
                    print(f"  ✓ Success! ({file_size_mb:.2f} MB)")
            else:
                print(f"  ✗ Failed to extract clip")
                
        except Exception as e:
            result = {
                'clip_number': i,
                'start': start,
                'end': end,
                'duration': duration,
                'output_path': output_path,
                'success': False,
                'error': str(e),
                'metadata': clip
            }
            print(f"  ✗ Error: {e}")
        
        results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = total_clips - successful
    
    print("\n" + "=" * 60)
    print(f"Extraction complete: {successful}/{total_clips} successful")
    if failed > 0:
        print(f"Failed clips: {failed}")
    
    return results


def save_extraction_report(results: List[Dict[str, Any]], output_path: str):
    """
    Save extraction results to JSON file.
    
    Args:
        results: List of extraction result dictionaries
        output_path: Path to save report JSON
    """
    report = {
        'total_clips': len(results),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'clips': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nExtraction report saved to: {output_path}")


# CLI interface
def main():
    """Command-line interface for clip extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract video clips based on viral detection timestamps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract clips from viral detection JSON
  python clip_extractor.py video.mp4 video_viral_clips.json
  
  # Custom output directory and quality
  python clip_extractor.py video.mp4 clips.json -o output_clips -q high
  
  # Extract a single clip manually
  python clip_extractor.py video.mp4 -s 10.5 -e 35.2 -o clip.mp4
        """
    )
    
    parser.add_argument('video', help='Input video file')
    parser.add_argument('clips_json', nargs='?', help='Viral clips JSON file (optional if using -s/-e)')
    parser.add_argument('-o', '--output', help='Output directory or file path')
    parser.add_argument('-q', '--quality', choices=['fast', 'balanced', 'high'],
                       default='balanced', help='Encoding quality (default: balanced)')
    parser.add_argument('-s', '--start', type=float, help='Start time in seconds (for single clip)')
    parser.add_argument('-e', '--end', type=float, help='End time in seconds (for single clip)')
    parser.add_argument('--prefix', default='clip', help='Prefix for output filenames (default: clip)')
    parser.add_argument('--report', help='Save extraction report to JSON file')
    parser.add_argument('--vertical', action='store_true', 
                       help='Convert extracted clips to 9:16 vertical format (uses main.py)')
    
    args = parser.parse_args()
    
    # Mode 1: Single clip extraction (manual timestamps)
    if args.start is not None and args.end is not None:
        if not args.output:
            print("Error: -o/--output required when using -s/-e")
            sys.exit(1)
        
        print(f"Extracting single clip: {args.start}s -> {args.end}s")
        success = extract_clip(args.video, args.start, args.end, args.output, args.quality)
        
        if success:
            print(f"✓ Clip saved to: {args.output}")
            sys.exit(0)
        else:
            print("✗ Extraction failed")
            sys.exit(1)
    
    # Mode 2: Batch extraction from JSON
    elif args.clips_json:
        output_dir = args.output or 'clips'
        results = extract_clips_from_json(
            args.video,
            args.clips_json,
            output_dir=output_dir,
            quality=args.quality,
            prefix=args.prefix,
            convert_vertical=args.vertical
        )
        
        # Save report if requested
        if args.report:
            save_extraction_report(results, args.report)
        
        # Exit with error code if any clips failed
        failed = sum(1 for r in results if not r['success'])
        sys.exit(1 if failed > 0 else 0)
    
    else:
        parser.print_help()
        print("\nError: Either provide clips_json file OR use -s/-e for single clip extraction")
        sys.exit(1)


if __name__ == '__main__':
    main()
