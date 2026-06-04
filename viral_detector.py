"""
Viral Clip Detection using Google Gemini API
Author: AI Video Clipping Bot
Purpose: Analyze video transcripts and identify viral moments for short-form content
"""

import json
import os
from typing import Dict, List, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

VIRAL_PROMPT_TEMPLATE = """You are a senior short-form video editor specializing in TikTok, Instagram Reels, and YouTube Shorts.

Analyze the provided transcript and identify 3-15 MOST VIRAL moments suitable for short-form content.

REQUIREMENTS:
- Each clip must be 15-60 seconds long
- Strong hooks required (first 3 seconds are CRITICAL)
- Natural cuts at silence/pauses preferred
- Include viral_hook_text (max 10 words) - catchy overlay text
- Platform-specific titles and descriptions
- Focus on: jaw-dropping facts, emotional moments, plot twists, actionable tips, or controversy

VIDEO DURATION: {duration} seconds
FULL TRANSCRIPT:
{transcript}

WORDS WITH TIMESTAMPS:
{words}

Return ONLY valid JSON in this exact format (no markdown, no code blocks):
{{
  "shorts": [
    {{
      "start": 12.340,
      "end": 37.900,
      "viral_hook_text": "Did you know this?",
      "video_title_for_youtube_short": "Shocking Discovery You Need To Know 🤯",
      "video_description_for_tiktok": "This changed everything 😱 #viral #mindblown #fyp",
      "video_description_for_instagram": "I can't believe this is real 🔥 Tag someone who needs to see this! #reels #viral"
    }}
  ]
}}
"""


class ViralDetector:
    """Detects viral moments in video transcripts using Google Gemini API."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        """
        Initialize the Viral Detector.
        
        Args:
            api_key: Google Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Gemini model to use (gemini-2.5-flash is default, gemini-2.5-pro for better quality)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY in .env file or pass as argument.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
    
    def detect_viral_clips(
        self, 
        transcript: Dict[str, Any], 
        video_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Detect viral moments in a video transcript.
        
        Args:
            transcript: Transcript dict from transcribe.py with keys:
                       - 'text': full transcript text
                       - 'segments': list of segments with word-level timestamps
                       - 'language': detected language
            video_duration: Duration of the video in seconds
        
        Returns:
            List of viral clip dictionaries with fields:
            - start: start time in seconds
            - end: end time in seconds
            - viral_hook_text: catchy overlay text
            - video_title_for_youtube_short: YouTube Shorts title
            - video_description_for_tiktok: TikTok description with hashtags
            - video_description_for_instagram: Instagram description with hashtags
        """
        # Extract words with timestamps for prompt
        words_data = self._extract_words_from_segments(transcript['segments'])
        
        # Format the prompt
        prompt = VIRAL_PROMPT_TEMPLATE.format(
            duration=video_duration,
            transcript=transcript['text'],
            words=json.dumps(words_data[:200], indent=2)  # Limit to first 200 words to save tokens
        )
        
        # Call Gemini API
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=4096,  # Increased for longer responses
                    response_mime_type="application/json",  # Enforce JSON response
                )
            )
            
            # Parse response - handle potential candidates
            if hasattr(response, 'candidates') and response.candidates:
                result_text = response.candidates[0].content.parts[0].text
            else:
                result_text = response.text
            
            result = json.loads(result_text)
            
            # Validate response structure
            clips = self._validate_clips(result.get('shorts', []), video_duration)
            
            return clips
            
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON response from Gemini: {e}")
            print(f"Response was: {result_text}")
            raise
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise
    
    def _extract_words_from_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Extract word-level timestamps from segments.
        
        Args:
            segments: List of segment dictionaries
        
        Returns:
            List of word dictionaries with 'word', 'start', 'end' keys
        """
        words = []
        for segment in segments:
            if 'words' in segment:
                for word_info in segment['words']:
                    words.append({
                        'word': word_info.get('word', ''),
                        'start': word_info.get('start', 0),
                        'end': word_info.get('end', 0)
                    })
        return words
    
    def _validate_clips(
        self, 
        clips: List[Dict], 
        video_duration: float
    ) -> List[Dict]:
        """
        Validate and filter clips based on requirements.
        
        Args:
            clips: List of clip dictionaries from API
            video_duration: Total video duration in seconds
        
        Returns:
            List of validated clips
        """
        validated = []
        
        for clip in clips:
            # Check required fields
            required_fields = [
                'start', 'end', 'viral_hook_text',
                'video_title_for_youtube_short',
                'video_description_for_tiktok',
                'video_description_for_instagram'
            ]
            
            if not all(field in clip for field in required_fields):
                print(f"Warning: Skipping clip with missing fields: {clip}")
                continue
            
            # Validate timing
            start = float(clip['start'])
            end = float(clip['end'])
            duration = end - start
            
            if start < 0 or end > video_duration:
                print(f"Warning: Clip timestamps out of range: {start}-{end}s (video is {video_duration}s)")
                continue
            
            if not (15 <= duration <= 60):
                print(f"Warning: Clip duration {duration}s outside 15-60s range")
                continue
            
            if start >= end:
                print(f"Warning: Invalid clip timing: start={start}, end={end}")
                continue
            
            validated.append(clip)
        
        if len(validated) == 0:
            print("Warning: No valid clips found!")
        
        return validated


def detect_viral_clips(
    transcript: Dict[str, Any], 
    video_duration: float,
    api_key: str = None,
    model: str = "gemini-2.5-flash"
) -> List[Dict[str, Any]]:
    """
    Convenience function to detect viral clips.
    
    Args:
        transcript: Transcript dict from transcribe.py
        video_duration: Duration of the video in seconds
        api_key: Optional Gemini API key (defaults to env var)
        model: Gemini model to use (default: gemini-2.5-flash)
    
    Returns:
        List of viral clip dictionaries
    
    Example:
        >>> from transcribe import transcribe_video
        >>> from viral_detector import detect_viral_clips
        >>> 
        >>> transcript = transcribe_video('video.mp4')
        >>> clips = detect_viral_clips(transcript, video_duration=1800)
        >>> 
        >>> for i, clip in enumerate(clips, 1):
        >>>     print(f"Clip {i}: {clip['start']:.2f}s - {clip['end']:.2f}s")
        >>>     print(f"Hook: {clip['viral_hook_text']}")
    """
    detector = ViralDetector(api_key=api_key, model=model)
    return detector.detect_viral_clips(transcript, video_duration)


# CLI usage
if __name__ == "__main__":
    import sys
    from transcribe import transcribe_video
    
    if len(sys.argv) < 2:
        print("Usage: python viral_detector.py <video_file>")
        print("\nExample:")
        print("  python viral_detector.py sample_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"🎬 Processing: {video_path}")
    print("=" * 60)
    
    # Step 1: Transcribe
    print("\n📝 Step 1: Transcribing video...")
    transcript = transcribe_video(video_path)
    print(f"✅ Transcription complete! Language: {transcript['language']}")
    print(f"📄 Transcript: {transcript['text'][:200]}...")
    
    # Get video duration (rough estimate from last segment)
    video_duration = transcript['segments'][-1]['end'] if transcript['segments'] else 0
    
    # Step 2: Detect viral clips
    print(f"\n🔍 Step 2: Detecting viral moments (Duration: {video_duration:.1f}s)...")
    clips = detect_viral_clips(transcript, video_duration)
    
    print(f"\n✅ Found {len(clips)} viral clips!")
    print("=" * 60)
    
    # Display results
    for i, clip in enumerate(clips, 1):
        duration = clip['end'] - clip['start']
        print(f"\n🎥 Clip {i}: {clip['start']:.2f}s → {clip['end']:.2f}s ({duration:.1f}s)")
        print(f"   Hook: {clip['viral_hook_text']}")
        print(f"   YouTube: {clip['video_title_for_youtube_short']}")
        print(f"   TikTok: {clip['video_description_for_tiktok'][:60]}...")
    
    # Save results
    output_file = video_path.replace('.mp4', '_viral_clips.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'video_file': video_path,
            'duration': video_duration,
            'language': transcript['language'],
            'clips': clips
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
