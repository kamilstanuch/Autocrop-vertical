#!/usr/bin/env python3
"""
Quick test script for Stage 1.1: Transcript Generation
Tests the transcribe.py module with a short audio sample.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcribe import transcribe_video, extract_words_in_range, get_transcript_text_in_range


def test_transcription():
    """
    Test transcription functionality.
    
    Note: For this test to work, you need a sample video file.
    The test will skip if no video file is found.
    """
    print("=" * 60)
    print("Stage 1.1 Test: Transcript Generation")
    print("=" * 60)
    
    # Look for a test video file
    test_videos = [
        'test_video.mp4',
        'sample.mp4',
        '../test_video.mp4',
    ]
    
    video_path = None
    for path in test_videos:
        if os.path.exists(path):
            video_path = path
            break
    
    if not video_path:
        print("\n⚠️  No test video found. Skipping test.")
        print("\nTo test transcription:")
        print("1. Place a short video file (5-30 seconds) in this directory")
        print("2. Name it 'test_video.mp4' or 'sample.mp4'")
        print("3. Run: python test_stage_1_1.py")
        print("\nModule is installed and ready to use!")
        return
    
    print(f"\n✅ Found test video: {video_path}")
    print("\n" + "-" * 60)
    print("Test 1: Basic Transcription")
    print("-" * 60)
    
    try:
        # Test transcription
        transcript = transcribe_video(video_path, model_size="tiny")
        
        print(f"\n✅ Transcription successful!")
        print(f"   Language: {transcript['language']}")
        print(f"   Segments: {len(transcript['segments'])}")
        
        # Calculate total words
        total_words = sum(len(seg['words']) for seg in transcript['segments'])
        print(f"   Total words: {total_words}")
        
        print(f"\n📝 First 200 characters of transcript:")
        print(f"   {transcript['text'][:200]}...")
        
        # Test word extraction
        if transcript['segments']:
            print("\n" + "-" * 60)
            print("Test 2: Word-Level Timestamp Extraction")
            print("-" * 60)
            
            # Get duration from last segment
            last_seg = transcript['segments'][-1]
            duration = last_seg['end']
            
            # Extract words from first 5 seconds
            test_end = min(5.0, duration)
            words = extract_words_in_range(transcript, 0, test_end)
            
            print(f"\n✅ Extracted {len(words)} words from 0-{test_end}s")
            if words:
                print(f"\n   Sample words with timestamps:")
                for i, word in enumerate(words[:5]):
                    print(f"   [{word['start']:.2f}s - {word['end']:.2f}s] {word['word']}")
            
            # Test text extraction
            print("\n" + "-" * 60)
            print("Test 3: Text Range Extraction")
            print("-" * 60)
            
            text = get_transcript_text_in_range(transcript, 0, test_end)
            print(f"\n✅ Text from 0-{test_end}s:")
            print(f"   {text}")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! Stage 1.1 complete.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    test_transcription()
