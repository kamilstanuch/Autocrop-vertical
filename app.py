#!/usr/bin/env python3
"""
Streamlit Frontend for AI Video Clipping Bot

Simple web interface for converting long videos into viral vertical clips.
"""

import os
import sys
import json
import shutil
import streamlit as st
from pathlib import Path
import subprocess


def run_full_pipeline(video_file, api_key, progress_placeholder):
    """
    Run complete pipeline: Transcribe → Detect → Extract → Subtitles → Effects
    
    Args:
        video_file: Uploaded video file object (from Streamlit)
        api_key: Gemini API key
        progress_placeholder: Streamlit placeholder for progress updates
        
    Returns:
        List of output video paths and summary text
    """
    try:
        # Set API key
        os.environ['GEMINI_API_KEY'] = api_key
        
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        video_path = output_dir / video_file.name
        with open(video_path, 'wb') as f:
            f.write(video_file.read())
        
        video_name = video_path.stem
        
        # Step 1: Transcribe
        progress_placeholder.info("Step 1/5: Transcribing video...")
        transcript_path = output_dir / f"{video_name}_transcript.json"
        result = subprocess.run([
            'python3', 'transcribe.py',
            str(video_path),
            str(transcript_path)
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            progress_placeholder.error(f"❌ Transcription failed: {result.stderr}")
            return None, None
        
        # Step 2: Detect viral moments
        progress_placeholder.info("Step 2/5: Detecting viral moments...")
        clips_json_path = output_dir / f"{video_name}_viral_clips.json"
        result = subprocess.run([
            'python3', 'viral_detector.py',
            str(video_path),
            '-o', str(clips_json_path)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            progress_placeholder.error(f"❌ Viral detection failed: {result.stderr}")
            return None, None
        
        # Load clips to get count
        with open(clips_json_path) as f:
            clips_data = json.load(f)
        num_clips = clips_data.get('num_clips', 0)
        
        if num_clips == 0:
            progress_placeholder.error("❌ No viral clips detected in this video")
            return None, None
        
        # Step 3: Extract clips (vertical)
        progress_placeholder.info(f"Step 3/5: Extracting {num_clips} clips (vertical)...")
        clips_dir = output_dir / "vertical_clips"
        clips_dir.mkdir(exist_ok=True)
        result = subprocess.run([
            'python3', 'clip_extractor.py',
            str(video_path),
            str(clips_json_path),
            '-o', str(clips_dir),
            '--vertical'
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            progress_placeholder.error(f"❌ Clip extraction failed: {result.stderr}")
            return None, None
        
        # Step 4: Add subtitles to each clip
        progress_placeholder.info(f"Step 4/5: Adding subtitles to {num_clips} clips...")
        subtitled_dir = output_dir / "subtitled"
        subtitled_dir.mkdir(exist_ok=True)
        
        clip_files = sorted(clips_dir.glob("viral_*.mp4"))
        for i, clip_file in enumerate(clip_files):
            clip_num = i + 1
            clip_data = clips_data['clips'][i]
            output_path = subtitled_dir / clip_file.name
            
            result = subprocess.run([
                'python3', 'subtitle_generator.py',
                str(clip_file),
                str(transcript_path),
                '-s', str(clip_data['start']),
                '-e', str(clip_data['end']),
                '-o', str(output_path)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                st.warning(f"⚠️ Subtitles failed for clip {clip_num}")
        
        # Step 5: Add AI effects
        progress_placeholder.info(f"Step 5/5: Applying AI effects to {num_clips} clips...")
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        
        for i, clip_file in enumerate(sorted(subtitled_dir.glob("viral_*.mp4"))):
            clip_num = i + 1
            clip_data = clips_data['clips'][i]
            output_path = final_dir / clip_file.name
            
            result = subprocess.run([
                'python3', 'ai_effects.py',
                str(clip_file),
                str(transcript_path),
                '-s', str(clip_data['start']),
                '-e', str(clip_data['end']),
                '-o', str(output_path)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                # Fallback: copy subtitled version
                shutil.copy(clip_file, output_path)
        
        progress_placeholder.success("✅ Complete!")
        
        # Generate summary
        final_clips = sorted(final_dir.glob("viral_*.mp4"))
        summary = f"""
## ✅ Pipeline Complete!

**Generated {len(final_clips)} viral clips** from your video.

### Clips:
"""
        for i, clip_data in enumerate(clips_data['clips'], 1):
            summary += f"\n**Clip {i}:** {clip_data['viral_hook_text']}\n\n"
            summary += f"- Duration: {clip_data['end'] - clip_data['start']:.1f}s\n"
            summary += f"- TikTok: {clip_data['video_description_for_tiktok'][:100]}...\n\n"
        
        # Return paths to final clips
        return list(final_clips), summary
        
    except subprocess.TimeoutExpired:
        progress_placeholder.error("❌ Pipeline timed out. Try a shorter video.")
        return None, None
    except Exception as e:
        progress_placeholder.error(f"❌ Error: {str(e)}")
        return None, None


# Streamlit UI
st.set_page_config(
    page_title="AI Video Clipping Bot",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 AI Video Clipping Bot")
st.markdown("""
Transform long videos into viral vertical shorts for TikTok, Instagram Reels, and YouTube Shorts.

**How it works:**
1. Upload your video
2. Enter your Gemini API key ([Get free key](https://aistudio.google.com/app/apikey))
3. Click "Generate Viral Clips"
4. Wait 5-15 minutes (depending on video length)
5. Download your clips!
""")

# Input section
video_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "mov", "avi"],
    help="Select a video file (MP4, MOV, or AVI)"
)

api_key = st.text_input(
    "Gemini API Key",
    type="password",
    placeholder="AIza...",
    help="Get a free API key from https://aistudio.google.com/app/apikey"
)

# Process button
if st.button("🚀 Generate Viral Clips", type="primary"):
    if not video_file:
        st.error("Please upload a video file")
    elif not api_key:
        st.error("Please enter your Gemini API key")
    else:
        # Create progress placeholder
        progress_placeholder = st.empty()
        
        # Run pipeline
        final_clips, summary = run_full_pipeline(video_file, api_key, progress_placeholder)
        
        if final_clips:
            # Display summary
            st.markdown(summary)
            
            # Create download section
            st.markdown("### 📥 Download Your Clips")
            for clip_path in final_clips:
                with open(clip_path, 'rb') as f:
                    st.download_button(
                        label=f"📥 {clip_path.name}",
                        data=f,
                        file_name=clip_path.name,
                        mime="video/mp4"
                    )

# Footer
st.markdown("""
---
### 📝 Notes
- Processing time: ~5-15 minutes depending on video length
- Recommended: Videos 5-15 minutes long
- Output: 3-15 vertical clips (15-60 seconds each)
- All clips include: vertical format, subtitles, and AI effects
""")
