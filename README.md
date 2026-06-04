# AI Video Clipping Bot

An AI-powered tool that transforms long horizontal videos into viral-ready vertical shorts for TikTok, Instagram Reels, and YouTube Shorts.

The system uses Google Gemini API to analyze video transcripts and identify the most engaging moments, then generates platform-optimized metadata for each clip.

## Architecture

```
Long Video Input
    |
    v
+-------------------------------+
| 1. Transcription              | COMPLETE
|    (faster-whisper)           | word-level timestamps
+-------------------------------+
    |
    v
+-------------------------------+
| 2. AI Viral Detection         | COMPLETE
|    (Google Gemini API)        | identifies 3-15 clips (15-60s)
|                               | generates hooks & metadata
+-------------------------------+
    |
    v
+-------------------------------+
| 3. Clip Extraction            | COMPLETE
|    (FFmpeg)                   | precise timestamp cutting
|                               | + vertical conversion (optional)
+-------------------------------+
    |
    v
+-------------------------------+
| 4. Subtitle Generation        | COMPLETE
|    (SRT/ASS from timestamps)  | TikTok-style captions
+-------------------------------+
    |
    v
+-------------------------------+
| 5. Hook Overlays              | PENDING
|    (PIL + FFmpeg)             | viral text overlays
+-------------------------------+
    |
    v
+-------------------------------+
| 6. AI Effects                 | PENDING
|    (Gemini + FFmpeg filters)  | dynamic zooms, enhancements
+-------------------------------+
    |
    v
Multiple Viral Clips Ready to Post
```

## Setup

```bash
# Clone and install dependencies
git clone <repository-url>
cd "AI Video Clipping Bot"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure Google Gemini API key (free tier available)
echo "GEMINI_API_KEY=your_key_here" > .env
```

Get a free API key at: https://aistudio.google.com/app/apikey

Prerequisites: Python 3.8+, FFmpeg in PATH

## Commands

### Viral Moment Detection

```bash
# Detect viral moments from a video
python3 viral_detector.py video.mp4

# Output: video_viral_clips.json with timestamps and metadata
```

### Clip Extraction

```bash
# Extract all viral clips from JSON (horizontal format)
python3 clip_extractor.py video.mp4 video_viral_clips.json

# Extract and convert to vertical format (9:16) - RECOMMENDED
python3 clip_extractor.py video.mp4 video_viral_clips.json --vertical

# Custom output directory and quality
python3 clip_extractor.py video.mp4 clips.json -o output_clips -q high --vertical

# Extract single clip manually
python3 clip_extractor.py video.mp4 -s 10.5 -e 35.2 -o clip.mp4
```

### Subtitle Generation

```bash
# Add TikTok-style subtitles to a clip
python3 subtitle_generator.py clip.mp4 transcript.json

# Specify time range
python3 subtitle_generator.py clip.mp4 transcript.json -s 10.5 -e 68.7

# Custom output and styling
python3 subtitle_generator.py clip.mp4 transcript.json -o subtitled.mp4 --style tiktok --position bottom

# Advanced options
python3 subtitle_generator.py clip.mp4 transcript.json --max-chars 25 --max-duration 2.5
```

Note: Requires FFmpeg with libass support. Install with: `brew install ffmpeg-full`

### Transcription

```bash
# Generate transcript with word-level timestamps
python3 transcribe.py video.mp4 output.json

# Choose model size (tiny, base, small, medium, large)
python3 transcribe.py video.mp4 output.json --model base
```

### Vertical Video Conversion (Legacy Feature)

```bash
# Convert horizontal to vertical (9:16)
python3 main.py -i video.mp4 -o vertical.mp4

# Custom aspect ratio
python3 main.py -i video.mp4 -o vertical.mp4 --ratio 4:5

# Quality presets: fast, balanced, high
python3 main.py -i video.mp4 -o vertical.mp4 --quality high

# Hardware encoding
python3 main.py -i video.mp4 -o vertical.mp4 --encoder hw
```

## Output Format

Each detected viral clip includes:
- Precise start/end timestamps
- Viral hook text (max 10 words for overlay)
- YouTube Shorts optimized title
- TikTok description with hashtags
- Instagram Reels description with hashtags

Example JSON output:
```json
{
  "clips": [
    {
      "start": 106.02,
      "end": 138.74,
      "viral_hook_text": "Keto beats Ozempic for hunger?",
      "video_title_for_youtube_short": "Keto vs Ozempic: The ULTIMATE Hunger Hormone Showdown!",
      "video_description_for_tiktok": "Keto naturally boosts GLP1 and silences hunger...",
      "video_description_for_instagram": "Learn how keto naturally fixes your hunger..."
    }
  ]
}
```
