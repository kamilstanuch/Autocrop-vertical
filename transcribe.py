"""
Transcription module using faster-whisper for audio transcription.
Extracts word-level timestamps for precise subtitle and clip generation.
"""

import os
from faster_whisper import WhisperModel


def transcribe_video(video_path, model_size="base", device="cpu", compute_type="int8"):
    """
    Transcribe audio from a video file using faster-whisper.
    
    Args:
        video_path: Path to the video file
        model_size: Whisper model size (tiny, base, small, medium, large)
        device: Device to run on (cpu, cuda)
        compute_type: Compute type for optimization (int8, float16, float32)
    
    Returns:
        dict: {
            'text': 'full transcript text',
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.5,
                    'text': 'Hello world',
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 0.5},
                        {'word': 'world', 'start': 0.6, 'end': 2.5}
                    ]
                },
                ...
            ],
            'language': 'en'
        }
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"🎙️  Transcribing video with Faster-Whisper...")
    print(f"   Model: {model_size} | Device: {device} | Compute: {compute_type}")
    
    # Initialize the model
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # Transcribe with word-level timestamps
    segments, info = model.transcribe(video_path, word_timestamps=True)
    
    print(f"   Detected language: '{info.language}' with probability {info.language_probability:.2f}")
    
    # Convert to structured format
    transcript_segments = []
    full_text = ""
    
    for segment in segments:
        # Print progress to keep user informed
        print(f"   [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
        seg_dict = {
            'start': segment.start,
            'end': segment.end,
            'text': segment.text,
            'words': []
        }
        
        # Add word-level timestamps if available
        if segment.words:
            for word in segment.words:
                seg_dict['words'].append({
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'probability': word.probability
                })
        
        transcript_segments.append(seg_dict)
        full_text += segment.text + " "
    
    result = {
        'text': full_text.strip(),
        'segments': transcript_segments,
        'language': info.language
    }
    
    print(f" Transcription complete! Total segments: {len(transcript_segments)}")
    
    return result


def extract_words_in_range(transcript, start_time, end_time):
    """
    Extract all words within a specific time range from the transcript.
    
    Args:
        transcript: Transcript dict from transcribe_video()
        start_time: Start time in seconds
        end_time: End time in seconds
    
    Returns:
        list: List of word dicts with timestamps
    """
    words = []
    
    for segment in transcript.get('segments', []):
        for word_info in segment.get('words', []):
            # Check if word overlaps with the time range
            if word_info['end'] > start_time and word_info['start'] < end_time:
                words.append(word_info)
    
    return words


def get_transcript_text_in_range(transcript, start_time, end_time):
    """
    Get the transcript text within a specific time range.
    
    Args:
        transcript: Transcript dict from transcribe_video()
        start_time: Start time in seconds
        end_time: End time in seconds
    
    Returns:
        str: Transcript text in the range
    """
    words = extract_words_in_range(transcript, start_time, end_time)
    return ' '.join([w['word'] for w in words])


if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <video_path> [output_json]")
        print("Example: python transcribe.py video.mp4 transcript.json")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Transcribe the video
    try:
        transcript = transcribe_video(video_path)
        
        # Save to JSON if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)
            print(f"\n Transcript saved to: {output_path}")
        else:
            # Print summary
            print(f"\n Transcript Summary:")
            print(f"   Language: {transcript['language']}")
            print(f"   Segments: {len(transcript['segments'])}")
            print(f"   Total words: {sum(len(s['words']) for s in transcript['segments'])}")
            print(f"\n Full text ({len(transcript['text'])} chars):")
            print(f"   {transcript['text'][:500]}...")
            
    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)
